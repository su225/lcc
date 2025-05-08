use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::num::ParseIntError;

use thiserror::Error;

use crate::codegen::AsmInstruction::{Mov, Ret, Unary};
use crate::tacky::{Instruction, IRFunction, IRProgram, IRSymbol, IRUnaryOperator, IRValue};

#[derive(Debug, Clone, PartialEq)]
pub enum Register {
    AX,
    BX,
    CX,
    DX,
    EAX,
    EBX,
    ECX,
    EDX,
    R10D,
    RAX,
    RBX,
    RCX,
    RDX,
    R10,

    RSP,
    RBP,
}

impl Display for Register {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", match &self {
            Register::AX => "ax",
            Register::BX => "bx",
            Register::CX => "cx",
            Register::DX => "dx",

            Register::EAX => "eax",
            Register::EBX => "ebx",
            Register::ECX => "ecx",
            Register::EDX => "edx",

            Register::RAX => "rax",
            Register::RBX => "rbx",
            Register::RCX => "rcx",
            Register::RDX => "rdx",
            Register::R10D => "r10d",

            Register::R10 => "r10",
            Register::RSP => "rsp",
            Register::RBP => "rbp",
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum AsmUnaryOperator {
    Neg,
    Not,
}

impl Display for AsmUnaryOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match &self {
            AsmUnaryOperator::Neg => "negl",
            AsmUnaryOperator::Not => "notl",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StackOffset(isize);

#[derive(Debug, Clone, PartialEq)]
pub enum AsmOperand {
    Imm(i64),
    Register(Register),
    Pseudo(IRSymbol),
    Stack { offset: StackOffset },
}

impl Display for AsmOperand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match &self {
            AsmOperand::Imm(n) => format!("${}", n),
            AsmOperand::Register(r) => r.to_string(),
            AsmOperand::Pseudo(p) => format!("<<{}>>", p),
            AsmOperand::Stack { offset } => format!("{}(%rbp)", offset.0),
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum AsmInstruction {
    Mov { src: AsmOperand, dst: AsmOperand },
    Unary { op: AsmUnaryOperator, dst: AsmOperand },
    AllocateStack(usize),
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
}

pub fn generate_assembly(p: IRProgram) -> Result<AsmProgram, CodegenError> {
    let mut asm_functions = Vec::with_capacity(p.functions.len());
    for f in p.functions {
        let asm_func = generate_function_assembly(f)?;
        let mut stack_alloc_ctx = StackAllocationContext::new();
        let mut stack_alloced = allocate_stack_frame(&mut stack_alloc_ctx, asm_func)?;
        let reqd_stack_size = stack_alloc_ctx.stack_size;
        stack_alloced.instructions.insert(0, AsmInstruction::AllocateStack(reqd_stack_size));
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
                Mov { src: from_ir_value(src), dst: asm_dst_operand.clone() },
                Unary {
                    op: match operator {
                        IRUnaryOperator::Complement => AsmUnaryOperator::Not,
                        IRUnaryOperator::Negate => AsmUnaryOperator::Neg,
                    },
                    dst: asm_dst_operand,
                },
            ])
        }
        Instruction::Return(v) => {
            Ok(vec![
                Mov {
                    src: from_ir_value(v),
                    dst: AsmOperand::Register(Register::EAX),
                },
                Ret,
            ])
        }
    }
}

fn allocate_stack_frame(ctx: &mut StackAllocationContext, f: AsmFunction) -> Result<AsmFunction, CodegenError> {
    let mut res_instrs = Vec::with_capacity(f.instructions.len());
    for instr in f.instructions {
        let alloced = match instr {
            Mov { src, dst } => {
                match (src, dst) {
                    (AsmOperand::Pseudo(s), AsmOperand::Pseudo(d)) => vec![
                        Mov {
                            src: AsmOperand::Stack { offset: get_or_allocate_stack(ctx, s) },
                            dst: AsmOperand::Register(Register::R10D),
                        },
                        Mov {
                            src: AsmOperand::Register(Register::R10D),
                            dst: AsmOperand::Stack { offset: get_or_allocate_stack(ctx, d) },
                        },
                    ],
                    (AsmOperand::Pseudo(s), dst_operand) => vec![
                        Mov {
                            src: AsmOperand::Stack { offset: get_or_allocate_stack(ctx, s) },
                            dst: dst_operand,
                        },
                    ],
                    (src_operand, AsmOperand::Pseudo(d)) => vec![
                        Mov {
                            src: src_operand,
                            dst: AsmOperand::Stack { offset: get_or_allocate_stack(ctx, d) },
                        }
                    ],
                    (src_operand, dst_operand) => vec![
                        Mov { src: src_operand, dst: dst_operand }
                    ],
                }
            }
            Unary { op, dst: AsmOperand::Pseudo(sym) } => vec![
                Unary {
                    op,
                    dst: AsmOperand::Stack { offset: get_or_allocate_stack(ctx, sym) },
                }
            ],
            instr => vec![instr],
        };
        res_instrs.extend(alloced);
    }
    Ok(AsmFunction {
        name: f.name,
        instructions: res_instrs,
    })
}

fn get_or_allocate_stack(ctx: &mut StackAllocationContext, sym: IRSymbol) -> StackOffset {
    if let Some(&offset) = ctx.symbol_offset.get(&sym) {
        return offset;
    }
    ctx.stack_size += 8;
    ctx.cur_offset = StackOffset(ctx.cur_offset.0 - 8);
    let new_offset = ctx.cur_offset;
    ctx.symbol_offset.insert(sym, new_offset);
    new_offset
}

fn from_ir_value(v: IRValue) -> AsmOperand {
    match v {
        IRValue::Constant(c) => AsmOperand::Imm(c),
        IRValue::Variable(s) => AsmOperand::Pseudo(s),
    }
}

#[cfg(test)]
mod test {
    use indoc::indoc;
    use AsmOperand::Register;
    use crate::codegen::{AsmFunction, AsmOperand, AsmProgram, generate_assembly, StackOffset};
    use crate::codegen::AsmInstruction::{AllocateStack, Mov, Ret, Unary};
    use crate::codegen::AsmOperand::{Imm, Stack};
    use crate::codegen::AsmUnaryOperator::Not;
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
                    Mov {
                        src: AsmOperand::Imm(0),
                        dst: Register(EAX),
                    },
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
                    Mov { src: Imm(100), dst: Register(EAX) },
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
                    Mov { src: Imm(0), dst: Stack { offset: StackOffset(-8) } },
                    Unary { op: Not, dst: Stack { offset: StackOffset(-8) } },
                    Mov { src: Stack { offset: StackOffset(-8) }, dst: Register(EAX) },
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