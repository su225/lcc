use std::fmt::{Display, Formatter};
use crate::codegen::x86_64::register::Register::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Register {
    EDI, ESI, EAX, EBX, ECX, EDX, R8D, R9D, R10D, R11D,

    RSP, RBP, R8, R9,

    AL, BL, CL, DL, R8B, R9B, R10B, R11B,
}

impl Register {
    pub fn width(&self) -> usize {
        match self {
            RSP | RBP | R8 | R9 => 8,
            EDI | ESI | EAX | EBX | ECX | EDX | R8D | R9D | R10D | R11D => 4,
            AL | BL | CL | DL | R8B | R9B | R10B | R11B => 1,
        }
    }

    pub fn least_significant_byte(&self) -> Option<Register> {
        match self {
            EDI => None,
            ESI => None,
            EAX => Some(AL),
            EBX => Some(BL),
            ECX => Some(CL),
            EDX => Some(DL),
            R10D => Some(R10B),
            R11D => Some(R11B),
            RSP | RBP => None,
            AL => Some(AL),
            BL => Some(BL),
            CL => Some(CL),
            DL => Some(DL),
            R10B => Some(R10B),
            R11B => Some(R11B),
            R8D => Some(R8B),
            R9D => Some(R9B),
            R8 => Some(R8B),
            R9 => Some(R9B),
            R8B => Some(R8B),
            R9B => Some(R9B),
        }
    }
}

impl Display for Register {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", match &self {
            EDI => "edi",
            ESI => "esi",
            EAX => "eax",
            EBX => "ebx",
            ECX => "ecx",
            EDX => "edx",
            R10D => "r10d",
            R11D => "r11d",
            R8D => "r8d",
            R9D => "r9d",

            RSP => "rsp",
            RBP => "rbp",
            R8 => "r8",
            R9 => "r9",

            AL => "al",
            BL => "bl",
            CL => "cl",
            DL => "dl",
            R10B => "r10b",
            R11B => "r11b",
            R8B => "r8b",
            R9B => "r9b",
        })
    }
}