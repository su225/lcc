use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::num::ParseIntError;
use thiserror::Error;
use crate::codegen::x86_64::AsmInstruction::*;
use crate::codegen::x86_64::AsmOperand::*;
use crate::codegen::x86_64::register::Register;
use crate::codegen::x86_64::register::Register::*;
use crate::tacky::{TackyBinaryOperator, TackyFunction, TackyInstruction, TackyProgram, TackySymbol, TackyUnaryOperator, TackyValue};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StackOffset(pub(crate) isize);

#[derive(Debug, Clone, PartialEq)]
pub enum AsmOperand {
    Imm32(i32),
    Reg(Register),
    Pseudo(TackySymbol),
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
pub enum ConditionCode {
    Equal,
    NotEqual,
    Greater,
    Lesser,
    GreaterOrEqual,
    LesserOrEqual,
}

impl Display for ConditionCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConditionCode::Equal => f.write_str("e"),
            ConditionCode::NotEqual => f.write_str("ne"),
            ConditionCode::Greater => f.write_str("g"),
            ConditionCode::Lesser => f.write_str("l"),
            ConditionCode::GreaterOrEqual => f.write_str("ge"),
            ConditionCode::LesserOrEqual => f.write_str("le"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct AsmLabel(String);

impl From<TackySymbol> for AsmLabel {
    fn from(value: TackySymbol) -> Self {
        AsmLabel(value.0)
    }
}

impl Display for AsmLabel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(".L{}", self))
    }
}

#[derive(Debug, PartialEq)]
pub enum AsmInstruction {
    AllocateStack(usize),
    Mov8 { src: AsmOperand, dst: AsmOperand },
    Mov32 { src: AsmOperand, dst: AsmOperand },
    Neg32 { op: AsmOperand },
    Not32 { op: AsmOperand },
    Add32 { src: AsmOperand, dst: AsmOperand },
    Sub32 { src: AsmOperand, dst: AsmOperand },
    IMul32 { src: AsmOperand, dst: AsmOperand },
    IDiv32 { divisor: AsmOperand },
    And32 { src: AsmOperand, dst: AsmOperand },
    Or32 { src: AsmOperand, dst: AsmOperand },
    Xor32 { src: AsmOperand, dst: AsmOperand },
    Sal32 { src: AsmOperand, dst: AsmOperand },
    Sar32 { src: AsmOperand, dst: AsmOperand },
    Shl32 { src: AsmOperand, dst: AsmOperand },
    Shr32 { src: AsmOperand, dst: AsmOperand },
    SignExtendTo64, // Sign extend %eax to 64-bit into %edx
    Cmp32 { op1: AsmOperand, op2: AsmOperand },
    Jmp { target: AsmLabel },
    JmpConditional{ condition_code: ConditionCode, target_if_true: AsmLabel },
    SetCondition { condition_code: ConditionCode, dst: AsmOperand },
    Label(AsmLabel),
    Ret,
}

#[derive(Debug, PartialEq)]
pub struct AsmFunction {
    pub name: TackySymbol,
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

pub fn generate_assembly(p: TackyProgram) -> Result<AsmProgram, CodegenError> {
    let mut asm_functions = Vec::with_capacity(p.functions.len());
    for f in p.functions {
        let asm_func = generate_function_assembly(f)?;
        let mut stack_alloc_ctx = StackAllocationContext::new();
        let mut stack_alloced = fixup_asm_instructions(&mut stack_alloc_ctx, asm_func)?;
        let reqd_stack_size = stack_alloc_ctx.stack_size;
        stack_alloced.instructions.insert(0, AllocateStack(reqd_stack_size)); // not-efficient
        validate_generated_function_assembly(&stack_alloced); // validates generated assembly as sanity check
        asm_functions.push(stack_alloced);
    }
    Ok(AsmProgram { functions: asm_functions })
}

// validate_generated_function_assembly validates generated assembly code as a sanity check to
// help catch issues early on. If any issue was found here, it is likely a compiler bug. It works
// by checking for various x86-64 assembly rules.
fn validate_generated_function_assembly(asm_func: &AsmFunction) {
    todo!()
}

fn generate_function_assembly(f: TackyFunction) -> Result<AsmFunction, CodegenError> {
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

fn generate_instruction_assembly(ti: TackyInstruction) -> Result<Vec<AsmInstruction>, CodegenError> {
    match ti {
        TackyInstruction::Unary { operator, src, dst } => {
            let asm_dst_operand: AsmOperand = dst.into();
            match operator {
                TackyUnaryOperator::Complement => Ok(vec![
                    Mov32 { src: src.into(), dst: asm_dst_operand.clone() },
                    Not32 { op: asm_dst_operand },
                ]),
                TackyUnaryOperator::Negate => Ok(vec![
                    Mov32 { src: src.into(), dst: asm_dst_operand.clone() },
                    Neg32 { op: asm_dst_operand },
                ]),
                TackyUnaryOperator::Not => Ok(vec![
                    Cmp32 { op1: Imm32(0), op2: src.into() },
                    Mov32 { src: Imm32(0), dst: asm_dst_operand.clone() },
                    SetCondition { condition_code: ConditionCode::Equal, dst: asm_dst_operand },
                ]),
            }
        }
        TackyInstruction::Binary { operator, src1, src2, dst } => {
            let asm_dst_operand: AsmOperand = dst.into();
            let asm_src1_operand: AsmOperand = src1.into();
            let asm_src2_operand: AsmOperand = src2.into();
            match operator {
                TackyBinaryOperator::Add => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: asm_dst_operand.clone() },
                    Add32 { src: asm_src2_operand, dst: asm_dst_operand },
                ]),
                TackyBinaryOperator::Subtract => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: asm_dst_operand.clone() },
                    Sub32 { src: asm_src2_operand, dst: asm_dst_operand },
                ]),
                TackyBinaryOperator::Multiply => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: asm_dst_operand.clone() },
                    IMul32 { src: asm_src2_operand, dst: asm_dst_operand },
                ]),
                TackyBinaryOperator::Divide => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: Reg(EAX) },
                    SignExtendTo64, // Sign extend EAX to EDX as IDivl expects 64-bit dividend
                    IDiv32 { divisor: asm_src2_operand },
                    Mov32 { src: Reg(EAX), dst: asm_dst_operand },
                ]),
                TackyBinaryOperator::Modulo => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: Reg(EAX) },
                    SignExtendTo64,
                    IDiv32 { divisor: asm_src2_operand },
                    Mov32 { src: Reg(EDX), dst: asm_dst_operand },
                ]),
                TackyBinaryOperator::BitwiseAnd => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: asm_dst_operand.clone() },
                    And32 { src: asm_src2_operand, dst: asm_dst_operand },
                ]),
                TackyBinaryOperator::BitwiseOr => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: asm_dst_operand.clone() },
                    Or32 { src: asm_src2_operand, dst: asm_dst_operand },
                ]),
                TackyBinaryOperator::BitwiseXor => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: asm_dst_operand.clone() },
                    Xor32 { src: asm_src2_operand, dst: asm_dst_operand },
                ]),
                TackyBinaryOperator::LeftShift => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: asm_dst_operand.clone() },
                    Sal32 { src: asm_src2_operand, dst: asm_dst_operand },
                ]),
                TackyBinaryOperator::RightShift => Ok(vec![
                    Mov32 { src: asm_src1_operand, dst: asm_dst_operand.clone() },
                    Sar32 { src: asm_src2_operand, dst: asm_dst_operand },
                ]),
                TackyBinaryOperator::Equal => Ok(generate_instruction_assembly_for_relational(
                    ConditionCode::Equal, asm_src1_operand, asm_src2_operand, asm_dst_operand)),

                TackyBinaryOperator::NotEqual => Ok(generate_instruction_assembly_for_relational(
                    ConditionCode::NotEqual, asm_src1_operand, asm_src2_operand, asm_dst_operand)),

                TackyBinaryOperator::LessThan => Ok(generate_instruction_assembly_for_relational(
                    ConditionCode::Lesser, asm_src1_operand, asm_src2_operand, asm_dst_operand)),

                TackyBinaryOperator::LessOrEqual => Ok(generate_instruction_assembly_for_relational(
                    ConditionCode::LesserOrEqual, asm_src1_operand, asm_src2_operand, asm_dst_operand)),

                TackyBinaryOperator::GreaterThan => Ok(generate_instruction_assembly_for_relational(
                    ConditionCode::Greater, asm_src1_operand, asm_src2_operand, asm_dst_operand)),

                TackyBinaryOperator::GreaterOrEqual => Ok(generate_instruction_assembly_for_relational(
                    ConditionCode::GreaterOrEqual, asm_src1_operand, asm_src2_operand, asm_dst_operand)),
            }
        }
        TackyInstruction::Return(v) => {
            Ok(vec![
                Mov32 { src: v.into(), dst: Reg(EAX) },
                Ret,
            ])
        },
        TackyInstruction::Copy { src, dst } => {
            let asm_src = src.into();
            let asm_dst = TackyValue::Variable(dst).into();
            Ok(vec![Mov32 { src: asm_src, dst: asm_dst }])
        },
        TackyInstruction::Jump { target: t } => Ok(vec![Jmp {target: AsmLabel::from(t)}]),
        TackyInstruction::JumpIfZero { condition, target } => Ok(vec![
            Cmp32 { op1: Imm32(0), op2: condition.into() },
            JmpConditional { condition_code: ConditionCode::Equal, target_if_true: AsmLabel::from(target) },
        ]),
        TackyInstruction::JumpIfNotZero { condition, target } => Ok(vec![
            Cmp32 { op1: Imm32(0), op2: condition.into() },
            JmpConditional { condition_code: ConditionCode::NotEqual, target_if_true: AsmLabel::from(target) },
        ]),
        TackyInstruction::Label(lbl) => Ok(vec![Label(AsmLabel::from(lbl))]),
    }
}

fn generate_instruction_assembly_for_relational(condition_code: ConditionCode, s1: AsmOperand, s2: AsmOperand, d: AsmOperand) -> Vec<AsmInstruction> {
    vec![
        Cmp32 { op1: s2, op2: s1 },
        Mov32 { src: Imm32(0), dst: d.clone() },
        SetCondition { condition_code, dst: d },
    ]
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
    symbol_offset: HashMap<TackySymbol, StackOffset>,
}

impl StackAllocationContext {
    fn new() -> Self {
        StackAllocationContext {
            stack_size: 0,
            cur_offset: StackOffset(0),
            symbol_offset: HashMap::new(),
        }
    }

    fn get_or_allocate_stack(&mut self, sym: TackySymbol) -> StackOffset {
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
    let mut processed_instrs = vec![];
    for instr in f.instructions {
        match instr {
            Mov8 { src, dst } => processed_instrs.extend(fixup_binary_expr!(Mov8, ctx, src, dst)),
            Mov32 { src, dst } => processed_instrs.extend(fixup_binary_expr!(Mov32, ctx, src, dst)),
            Not32 { op } => processed_instrs.extend(fixup_unary_expr!(Not32, ctx, op)),
            Neg32 { op } => processed_instrs.extend(fixup_unary_expr!(Neg32, ctx, op)),
            Add32 { src, dst } => processed_instrs.extend(fixup_binary_expr!(Add32, ctx, src, dst)),
            Sub32 { src, dst } => processed_instrs.extend(fixup_binary_expr!(Sub32, ctx, src, dst)),
            IMul32 { src, dst } => processed_instrs.extend(fixup_imul32(ctx, src, dst)),
            IDiv32 { divisor } => processed_instrs.extend(fixup_idiv32(ctx, divisor)),
            Sal32 { src, dst } => {
                let fixed_up = fixup_sal32(ctx, src, dst)?;
                processed_instrs.extend(fixed_up);
            }
            Sar32 { src, dst } => {
                let fixed_up = fixup_sar32(ctx, src, dst)?;
                processed_instrs.extend(fixed_up);
            }
            And32 { src, dst } => processed_instrs.extend(fixup_binary_expr!(And32, ctx, src, dst)),
            Or32 { src, dst } => processed_instrs.extend(fixup_binary_expr!(Or32, ctx, src, dst)),
            Xor32 { src, dst } => processed_instrs.extend(fixup_binary_expr!(Xor32, ctx, src, dst)),
            SetCondition { condition_code, dst: Pseudo(v) } => processed_instrs.push(
                SetCondition { condition_code, dst: Stack { offset: ctx.get_or_allocate_stack(v) }}),
            instr => processed_instrs.push(instr),
        }
    }
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

fn fixup_sal32(ctx: &mut StackAllocationContext, src: AsmOperand, dst: AsmOperand) -> Result<Vec<AsmInstruction>, CodegenError> {
    let fix_up_pseudo = fixup_binary_expr!(Sal32, ctx, src, dst);
    let mut processed = vec![];
    for instr in fix_up_pseudo {
        processed.extend(match instr {
            // todo: remove hardcoding later. We need to check for the
            //       width and insert down/upcasting at the IR-level so that
            //       we can lower to proper ASM types.
            Sal32 { src: Imm32(n), dst } =>
                if n < 0 {
                    vec![
                        Mov32 { src: dst.clone(), dst: Reg(R11D) },
                        Mov32 { src: Imm32(-n), dst: Reg(EAX) },
                        Mov8 { src: Reg(AL), dst: Reg(CL) },
                        Mov32 { src: Reg(R11D), dst: dst.clone() },
                        Sal32 { src: Reg(CL), dst: dst.clone() },
                        Neg32 { op: dst.clone() },
                    ]
                } else {
                    vec![
                        Mov32 { src: dst.clone(), dst: Reg(R11D) },
                        Mov32 { src: Imm32(n), dst: Reg(EAX) },
                        Mov8 { src: Reg(AL), dst: Reg(CL) },
                        Mov32 { src: Reg(R11D), dst: dst.clone() },
                        Sal32 { src: Reg(CL), dst: dst.clone() },
                    ]
                },
            Sal32 { src: Reg(r), dst } if r == R10D => vec![
                Mov8 { src: Reg(R10B), dst: Reg(CL) },
                Sal32 { src: Reg(CL), dst: dst.clone() },
            ],
            Sal32 { src: stack @ Stack { .. }, dst } => vec![
                Mov8 { src: stack, dst: Reg(CL) },
                Sal32 { src: Reg(CL), dst: dst.clone() },
            ],
            instr => vec![instr],
        });
    }
    Ok(processed)
}

fn fixup_sar32(ctx: &mut StackAllocationContext, src: AsmOperand, dst: AsmOperand) -> Result<Vec<AsmInstruction>, CodegenError> {
    let fix_up_pseudo = fixup_binary_expr!(Sar32, ctx, src, dst);
    let mut processed = vec![];
    for instr in fix_up_pseudo {
        processed.extend(match instr {
            // todo: remove hardcoding later. We need to check for the
            //       width and insert down/upcasting at the IR-level so that
            //       we can lower to proper ASM types.
            Sar32 { src: Imm32(n), dst } =>
                if n < 0 {
                    vec![
                        Mov32 { src: dst.clone(), dst: Reg(R11D) },
                        Mov32 { src: Imm32(-n), dst: Reg(EAX) },
                        Mov8 { src: Reg(AL), dst: Reg(CL) },
                        Mov32 { src: Reg(R11D), dst: dst.clone() },
                        Sar32 { src: Reg(CL), dst: dst.clone() },
                        Neg32 { op: dst.clone() },
                    ]
                } else {
                    vec![
                        Mov32 { src: dst.clone(), dst: Reg(R11D) },
                        Mov32 { src: Imm32(n), dst: Reg(EAX) },
                        Mov8 { src: Reg(AL), dst: Reg(CL) },
                        Mov32 { src: Reg(R11D), dst: dst.clone() },
                        Sar32 { src: Reg(CL), dst: dst.clone() },
                    ]
                },
            Sar32 { src: Reg(r), dst } if r == R10D => vec![
                Mov8 { src: Reg(R10B), dst: Reg(CL) },
                Sar32 { src: Reg(CL), dst: dst.clone() },
            ],
            Sar32 { src: stack @ Stack { .. }, dst } => vec![
                Mov8 { src: stack, dst: Reg(CL) },
                Sar32 { src: Reg(CL), dst },
            ],
            instr => vec![instr],
        });
    }
    Ok(processed)
}

impl From<TackyValue> for AsmOperand {
    fn from(v: TackyValue) -> Self {
        match v {
            TackyValue::Int32(c) => Imm32(c),
            TackyValue::Variable(s) => Pseudo(s),
        }
    }
}

#[cfg(test)]
mod test {
    use indoc::indoc;

    use crate::codegen::x86_64::*;
    use crate::codegen::x86_64::register::Register::EAX;
    use crate::codegen::x86_64::AsmInstruction::*;
    use crate::codegen::x86_64::AsmOperand::*;
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