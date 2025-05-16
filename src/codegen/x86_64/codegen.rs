use std::collections::HashMap;
use crate::codegen::x86_64::errors::CodegenError;
use crate::codegen::x86_64::register::Register::*;
use crate::codegen::x86_64::types::{AsmFunction, AsmInstruction, AsmOperand, AsmProgram, StackOffset};
use crate::codegen::x86_64::types::AsmInstruction::*;
use crate::codegen::x86_64::types::AsmOperand::*;
use crate::tacky::types::*;

pub fn generate_assembly(p: TackyProgram) -> Result<AsmProgram, CodegenError> {
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
            let asm_dst_operand = from_ir_value(dst);
            Ok(vec![
                Mov32 { src: from_ir_value(src), dst: asm_dst_operand.clone() },
                match operator {
                    TackyUnaryOperator::Complement => Not32 { op: asm_dst_operand },
                    TackyUnaryOperator::Negate => Neg32 { op: asm_dst_operand },
                    _ => todo!(),
                },
            ])
        }
        TackyInstruction::Binary { operator, src1, src2, dst } => {
            let asm_dst_operand = from_ir_value(dst);
            let asm_src1_operand = from_ir_value(src1);
            let asm_src2_operand = from_ir_value(src2);
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
                _ => todo!(),
            }
        }
        TackyInstruction::Return(v) => {
            Ok(vec![
                Mov32 { src: from_ir_value(v), dst: Reg(EAX) },
                Ret,
            ])
        }
        _ => todo!(),
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

fn from_ir_value(v: TackyValue) -> AsmOperand {
    match v {
        TackyValue::Constant32(c) => Imm32(c),
        TackyValue::Variable(s) => Pseudo(s),
    }
}

#[cfg(test)]
mod test {
    use indoc::indoc;

    use crate::codegen::x86_64::generate_assembly;
    use crate::codegen::x86_64::register::Register::EAX;
    use crate::codegen::x86_64::types::*;
    use crate::codegen::x86_64::types::AsmInstruction::*;
    use crate::codegen::x86_64::types::AsmOperand::*;
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