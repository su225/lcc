use std::fmt::{Display, Formatter};
use crate::codegen::x86_64::register::Register::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Register {
    EAX, EDX, R10D, R11D,

    RSP, RBP,

    R10B, AL, CL,
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

            R10B => "r10b",
            CL => "cl",
            AL => "al",
        })
    }
}