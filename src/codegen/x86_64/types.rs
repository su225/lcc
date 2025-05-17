use std::fmt::{Display, Formatter};
use crate::codegen::x86_64::register::Register;
use crate::tacky::TackySymbol;

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
            AsmOperand::Imm32(n) => format!("${}", n),
            AsmOperand::Reg(r) => r.to_string(),
            AsmOperand::Pseudo(p) => format!("<<{}>>", p),
            AsmOperand::Stack { offset } => format!("{}(%rbp)", offset.0),
        })
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