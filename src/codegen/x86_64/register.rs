use std::fmt::{Display, Formatter};
use crate::codegen::x86_64::register::Register::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Register {
    EAX, EBX, ECX, EDX, R10D, R11D,

    RSP, RBP,

    AL, BL, CL, DL, R10B, R11B,
}

impl Register {
    pub fn width(&self) -> usize {
        match self {
            RSP | RBP => 8,
            EAX | EBX | ECX | EDX | R10D | R11D => 4,
            AL | BL | CL | DL | R10B | R11B => 1,
        }
    }

    pub fn least_significant_byte(&self) -> Option<Register> {
        match self {
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
        }
    }
}

impl Display for Register {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", match &self {
            EAX => "eax",
            EBX => "ebx",
            ECX => "ecx",
            EDX => "edx",
            R10D => "r10d",
            R11D => "r11d",

            RSP => "rsp",
            RBP => "rbp",

            AL => "al",
            BL => "bl",
            CL => "cl",
            DL => "dl",
            R10B => "r10b",
            R11B => "r11b",
        })
    }
}