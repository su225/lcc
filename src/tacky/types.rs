use derive_more::Display;
use crate::parser::types::{BinaryOperator, UnaryOperator};

pub(crate) const COMPILER_GEN_PREFIX: &'static str = "<t>";

#[derive(Debug, Eq, PartialEq, Clone, Hash, Display)]
pub struct TackySymbol(pub String);

impl From<&str> for TackySymbol {
    fn from(value: &str) -> Self {
        TackySymbol(value.to_string())
    }
}

impl TackySymbol {
    pub fn is_generated(&self) -> bool {
        self.0.starts_with(COMPILER_GEN_PREFIX)
    }
}

#[derive(Debug, PartialEq)]
pub struct TackyProgram {
    pub functions: Vec<TackyFunction>,
}

#[derive(Debug, PartialEq)]
pub struct TackyFunction {
    pub identifier: TackySymbol,
    pub body: Vec<TackyInstruction>,
}

#[derive(Debug, PartialEq)]
pub enum TackyUnaryOperator {
    Complement,
    Negate,
    Not,
}

impl From<&UnaryOperator> for TackyUnaryOperator {
    fn from(value: &UnaryOperator) -> Self {
        match value {
            UnaryOperator::Complement => TackyUnaryOperator::Complement,
            UnaryOperator::Negate => TackyUnaryOperator::Negate,
            UnaryOperator::Not => todo!(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum TackyBinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
}

impl From<&BinaryOperator> for TackyBinaryOperator {
    fn from(value: &BinaryOperator) -> Self {
        match value {
            BinaryOperator::Add => TackyBinaryOperator::Add,
            BinaryOperator::Subtract => TackyBinaryOperator::Subtract,
            BinaryOperator::Multiply => TackyBinaryOperator::Multiply,
            BinaryOperator::Divide => TackyBinaryOperator::Divide,
            BinaryOperator::Modulo => TackyBinaryOperator::Modulo,

            BinaryOperator::BitwiseAnd => TackyBinaryOperator::BitwiseAnd,
            BinaryOperator::BitwiseOr => TackyBinaryOperator::BitwiseOr,
            BinaryOperator::BitwiseXor => TackyBinaryOperator::BitwiseXor,
            BinaryOperator::LeftShift => TackyBinaryOperator::LeftShift,
            BinaryOperator::RightShift => TackyBinaryOperator::RightShift,

            _ => todo!(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum TackyValue {
    Constant32(i32),
    Variable(TackySymbol),
}

#[derive(Debug, PartialEq)]
pub enum TackyInstruction {
    Unary {
        operator: TackyUnaryOperator,
        src: TackyValue,
        dst: TackyValue,
    },
    Binary {
        operator: TackyBinaryOperator,
        src1: TackyValue,
        src2: TackyValue,
        dst: TackyValue,
    },
    Return(TackyValue),
    Copy { src: TackyValue, dst: TackyValue },
    Jump { target: TackySymbol },
    JumpIfZero { condition: TackyValue, target: TackySymbol },
    JumpIfNotZero { condition: TackyValue, target: TackySymbol },
    Label(TackySymbol),
}