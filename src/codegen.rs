use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::num::ParseIntError;

use thiserror::Error;

use AsmOperand::{Pseudo, Stack};

use crate::codegen::AsmInstruction::*;
use crate::codegen::AsmOperand::*;
use crate::codegen::Register::*;
use crate::tacky::{Instruction, IRBinaryOperator, IRFunction, IRProgram, IRSymbol, IRUnaryOperator, IRValue};

#[derive(Debug, Clone, PartialEq)]
pub enum Register {
    EAX,
    EDX,
    R10D,
    R11D,

    RSP,
    RBP,
}

impl Display for Register {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", match &self {
            EAX => "eax",
            EDX => "edx",
            R10D => "r10d",
            R11D => "r11d",

            RSP => "rsp",
            RBP => "rbp",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StackOffset(isize);

#[derive(Debug, Clone, PartialEq)]
pub enum AsmOperand {
    Imm32(i32),
    Reg(Register),
    Pseudo(IRSymbol),
    Stack { offset: StackOffset },
}

impl Display for AsmOperand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match &self {
            Imm32(n) => format!("${}", n),
            Reg(r) => r.to_string(),
            Pseudo(p) => format!("<<{}>>", p),
            Stack { offset } => format!("{}(%rbp)", offset.0),
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum AsmInstruction {
    AllocateStack(usize),
    Mov32 { src: AsmOperand, dst: AsmOperand },
    Neg32 { op: AsmOperand },
    Not32 { op: AsmOperand },
    Add32 { src: AsmOperand, dst: AsmOperand },
    Sub32 { src: AsmOperand, dst: AsmOperand },
    IMul32 { src: AsmOperand, dst: AsmOperand },
    IDiv32 { divisor: AsmOperand },
    Cdq, // Sign extend %eax to 64-bit into %edx
    Ret,
}

#[derive(Debug, PartialEq)]
pub struct AsmFunction {
    pub name: IRSymbol,
    pub instructions: Vec<AsmInstruction>,
}

#[derive(Debug, PartialEq)]
pub struct AsmProgram {
    pub functions: Vec<AsmFunction>,
}

#[derive(Error, Debug)]
pub enum CodegenError {
    #[error(transparent)]
    IntImmediateParseError(#[from] ParseIntError),
}

pub fn generate_assembly(p: IRProgram) -> Result<AsmProgram, CodegenError> {
    let mut asm_functions = Vec::with_capacity(p.functions.len());
    for f in p.functions {
        let asm_func = generate_function_assembly(f)?;
        let mut stack_alloc_ctx = StackAllocationContext::new();
        let mut stack_alloced = fixup_asm_instructions(&mut stack_alloc_ctx, asm_func)?;
        let reqd_stack_size = stack_alloc_ctx.stack_size;
        stack_alloced.instructions.insert(0, AllocateStack(reqd_stack_size)); // not-efficient
        asm_functions.push(stack_alloced);
    }
    Ok(AsmProgram { functions: asm_functions })
}

fn generate_function_assembly(f: IRFunction) -> Result<AsmFunction, CodegenError> {
    let mut asm_instructions = Vec::with_capacity(f.body.len());
    for tacky_inst in f.body {
        let asm_instrs = generate_instruction_assembly(tacky_inst)?;
        asm_instructions.extend(asm_instrs);
    }
    Ok(AsmFunction {
        name: f.identifier.clone(),
        instructions: asm_instructions,
    })
}

fn generate_instruction_assembly(ti: Instruction) -> Result<Vec<AsmInstruction>, CodegenError> {
    match ti {
        Instruction::Unary { operator, src, dst } => {
            let asm_dst_operand = from_ir_value(dst);
            Ok(vec![
                Mov32 { src: from_ir_value(src), dst: asm_dst_operand.clone() },
                match operator {
                    IRUnaryOperator::Complement => Not32 { op: asm_dst_operand },
                    IRUnaryOperator::Negate => Neg32 { op: asm_dst_operand },
                },
            ])
        }
        Instruction::Binary { operator, src1, src2, dst } => {
            let asm_dst_operand = from_ir_value(dst);
            let asm_src1_operand = from_ir_value(src1);
            let asm_src2_operand = from_ir_value(src2);
            match operator {
                IRBinaryOperator::Add => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: asm_dst_operand.clone() },
                    Add32 { src: asm_src2_operand, dst: asm_dst_operand },
                ]),
                IRBinaryOperator::Subtract => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: asm_dst_operand.clone() },
                    Sub32 { src: asm_src2_operand, dst: asm_dst_operand },
                ]),
                IRBinaryOperator::Multiply => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: asm_dst_operand.clone() },
                    IMul32 { src: asm_src2_operand, dst: asm_dst_operand },
                ]),
                IRBinaryOperator::Divide => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: Reg(EAX) },
                    Cdq, // Sign extend EAX to EDX as IDivl expects 64-bit dividend
                    IDiv32 { divisor: asm_src2_operand },
                    Mov32 { src: Reg(EAX), dst: asm_dst_operand },
                ]),
                IRBinaryOperator::Modulo => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: Reg(EAX) },
                    Cdq,
                    IDiv32 { divisor: asm_src2_operand },
                    Mov32 { src: Reg(EDX), dst: asm_dst_operand },
                ]),
            }
        }
        Instruction::Return(v) => {
            Ok(vec![
                Mov32 { src: from_ir_value(v), dst: Reg(EAX) },
                Ret,
            ])
        }
    }
}

macro_rules! fixup_binary_expr {
    ($instruction:ident, $ctx:expr, $src:expr, $dst:expr) => {
        match ($src, $dst) {
            (Pseudo(s), Pseudo(d)) => vec![
                Mov32 { src: Stack { offset: $ctx.get_or_allocate_stack(s) }, dst: Reg(R10D) },
                $instruction { src: Reg(R10D), dst: Stack { offset: $ctx.get_or_allocate_stack(d) } },
            ],
            (Pseudo(s), dst) => vec![
                $instruction { src: Stack { offset: $ctx.get_or_allocate_stack(s) }, dst },
            ],
            (src, Pseudo(d)) => vec![
                $instruction { src, dst: Stack { offset: $ctx.get_or_allocate_stack(d) } },
            ],
            (src, dst) => vec![$instruction { src, dst }],
        }
    }
}

macro_rules! fixup_unary_expr {
    ($instruction:ident, $ctx:expr, $operand:expr) => {
        match $operand {
            Pseudo(s) => vec![$instruction { op: Stack { offset: $ctx.get_or_allocate_stack(s) } }],
            op => vec![$instruction { op }],
        }
    }
}

struct StackAllocationContext {
    stack_size: usize,
    cur_offset: StackOffset,
    symbol_offset: HashMap<IRSymbol, StackOffset>,
}

impl StackAllocationContext {
    fn new() -> Self {
        StackAllocationContext {
            stack_size: 0,
            cur_offset: StackOffset(0),
            symbol_offset: HashMap::new(),
        }
    }

    fn get_or_allocate_stack(&mut self, sym: IRSymbol) -> StackOffset {
        if let Some(&offset) = self.symbol_offset.get(&sym) {
            return offset;
        }
        self.stack_size += 8;
        self.cur_offset = StackOffset(self.cur_offset.0 - 8);
        let new_offset = self.cur_offset;
        self.symbol_offset.insert(sym, new_offset);
        new_offset
    }
}

fn fixup_asm_instructions(ctx: &mut StackAllocationContext, f: AsmFunction) -> Result<AsmFunction, CodegenError> {
    let processed_instrs = f.instructions.into_iter().flat_map(|instr| {
        match instr {
            Mov32 { src, dst } => fixup_binary_expr!(Mov32, ctx, src, dst),
            Not32 { op } => fixup_unary_expr!(Not32, ctx, op),
            Neg32 { op } => fixup_unary_expr!(Neg32, ctx, op),
            Add32 { src, dst } => fixup_binary_expr!(Add32, ctx, src, dst),
            Sub32 { src, dst } => fixup_binary_expr!(Sub32, ctx, src, dst),
            IMul32 { src, dst } => fixup_imul32(ctx, src, dst),
            IDiv32 { divisor } => fixup_idiv32(ctx, divisor),
            instr => vec![instr]
        }
    }).collect();
    Ok(AsmFunction {
        name: f.name,
        instructions: processed_instrs,
    })
}

fn fixup_imul32(ctx: &mut StackAllocationContext, src: AsmOperand, dst: AsmOperand) -> Vec<AsmInstruction> {
    let fix_up_pseudo = fixup_binary_expr!(IMul32, ctx, src, dst);
    fix_up_pseudo.into_iter().flat_map(|instr| {
        // If the destination of imul32 is a memory location
        // then it has to be fixed up again regardless of the
        // source operand. Hence, we use R11D as the scratch
        // register for it
        match instr {
            IMul32 { src, dst: dst_operand @ Stack { .. } } => vec![
                Mov32 { src: dst_operand.clone(), dst: Reg(R11D) },
                IMul32 { src, dst: Reg(R11D) },
                Mov32 { src: Reg(R11D), dst: dst_operand },
            ],
            instr => vec![instr],
        }
    }).collect()
}

fn fixup_idiv32(ctx: &mut StackAllocationContext, divisor: AsmOperand) -> Vec<AsmInstruction> {
    match divisor {
        constant_op @ Imm32(_) => vec![
            Mov32 { src: constant_op, dst: Reg(R10D) },
            IDiv32 { divisor: Reg(R10D) },
        ],
        Pseudo(s) => vec![
            IDiv32 { divisor: Stack { offset: ctx.get_or_allocate_stack(s) } }
        ],
        divisor => vec![
            IDiv32 { divisor }
        ]
    }
}

fn from_ir_value(v: IRValue) -> AsmOperand {
    match v {
        IRValue::Constant32(c) => Imm32(c),
        IRValue::Variable(s) => Pseudo(s),
    }
}

#[cfg(test)]
mod test {
    use indoc::indoc;

    use AsmOperand::Reg;

    use crate::codegen::{AsmFunction, AsmOperand, AsmProgram, generate_assembly, StackOffset};
    use crate::codegen::AsmInstruction::{AllocateStack, Mov32, Not32, Ret};
    use crate::codegen::AsmOperand::{Imm32, Stack};
    use crate::codegen::Register::EAX;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::tacky::emit;

    #[test]
    fn test_generate_assembly_for_return_0() {
        let src = indoc! {r#"
        int main(void) {
            return 0;
        }
        "#};
        let expected_asm = AsmProgram {
            functions: vec![AsmFunction {
                name: "main".into(),
                instructions: vec![
                    AllocateStack(0),
                    Mov32 { src: Imm32(0), dst: Reg(EAX) },
                    Ret,
                ],
            }]
        };
        assert_generated_asm_for_source(src, expected_asm);
    }

    #[test]
    fn test_generate_assembly_for_multi_digit_constant_return_value() {
        let src = indoc! {r#"
        int main(void) {
            // test case w/ multi-digit constant
            return 100;
        }
        "#};
        let expected_asm = AsmProgram {
            functions: vec![AsmFunction {
                name: "main".into(),
                instructions: vec![
                    AllocateStack(0),
                    Mov32 { src: Imm32(100), dst: Reg(EAX) },
                    Ret,
                ],
            }],
        };
        assert_generated_asm_for_source(src, expected_asm);
    }

    #[test]
    fn test_generate_assembly_for_bitwise_complement_operator() {
        let src = indoc! {r#"
        int main(void) {
            return ~0;
        }
        "#};
        let expected_asm = AsmProgram {
            functions: vec![AsmFunction {
                name: "main".into(),
                instructions: vec![
                    AllocateStack(8),
                    Mov32 { src: Imm32(0), dst: Stack { offset: StackOffset(-8) } },
                    Not32 { op: Stack { offset: StackOffset(-8) } },
                    Mov32 { src: Stack { offset: StackOffset(-8) }, dst: Reg(EAX) },
                    Ret,
                ],
            }],
        };
        assert_generated_asm_for_source(src, expected_asm);
    }

    fn assert_generated_asm_for_source(source_code: &str, expected_asm: AsmProgram) {
        let lexer = Lexer::new(source_code);
        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("Parsing should be successful");
        let ir = emit(&ast).expect("IR generation must be successful");
        let actual_asm = generate_assembly(ir).expect("Assembly generation must be successful");
        assert_eq!(expected_asm, actual_asm, "expected:{:#?}\nactual:{:#?}", expected_asm, actual_asm);
    }
}