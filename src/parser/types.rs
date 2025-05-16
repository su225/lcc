use derive_more::Add;
use serde::Serialize;
use crate::common::{Location, Radix};

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Symbol<'a> {
    pub name: &'a str,
    pub(crate) location: Location,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOperator {
    Complement,
    Negate,
    Not,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOperatorAssociativity {
    Left, Right
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOperator {
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

    And,
    Or,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Debug, PartialEq, Ord, PartialOrd, Eq, Add, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct BinaryOperatorPrecedence(pub(crate) u16);

impl BinaryOperator {
    #[inline]
    pub(crate) fn associativity(&self) -> BinaryOperatorAssociativity {
        match self {
            BinaryOperator::Add => BinaryOperatorAssociativity::Left,
            BinaryOperator::Subtract => BinaryOperatorAssociativity::Left,
            BinaryOperator::Multiply => BinaryOperatorAssociativity::Left,
            BinaryOperator::Divide => BinaryOperatorAssociativity::Left,
            BinaryOperator::Modulo => BinaryOperatorAssociativity::Left,
            BinaryOperator::BitwiseAnd => BinaryOperatorAssociativity::Left,
            BinaryOperator::BitwiseOr => BinaryOperatorAssociativity::Left,
            BinaryOperator::BitwiseXor => BinaryOperatorAssociativity::Left,
            BinaryOperator::LeftShift => BinaryOperatorAssociativity::Left,
            BinaryOperator::RightShift => BinaryOperatorAssociativity::Left,
            BinaryOperator::And => BinaryOperatorAssociativity::Left,
            BinaryOperator::Or => BinaryOperatorAssociativity::Left,
            BinaryOperator::Equal => BinaryOperatorAssociativity::Left,
            BinaryOperator::NotEqual => BinaryOperatorAssociativity::Left,
            BinaryOperator::LessThan => BinaryOperatorAssociativity::Left,
            BinaryOperator::LessThanOrEqual => BinaryOperatorAssociativity::Left,
            BinaryOperator::GreaterThan => BinaryOperatorAssociativity::Left,
            BinaryOperator::GreaterThanOrEqual => BinaryOperatorAssociativity::Left,
        }
    }

    #[inline]
    pub(crate) fn precedence(&self) -> BinaryOperatorPrecedence {
        match self {
            BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo => BinaryOperatorPrecedence(50),

            BinaryOperator::Add | BinaryOperator::Subtract => BinaryOperatorPrecedence(45),
            BinaryOperator::LeftShift | BinaryOperator::RightShift => BinaryOperatorPrecedence(42),

            BinaryOperator::LessThan
            | BinaryOperator::LessThanOrEqual
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterThanOrEqual => BinaryOperatorPrecedence(40),

            BinaryOperator::Equal | BinaryOperator::NotEqual => BinaryOperatorPrecedence(38),

            BinaryOperator::BitwiseAnd => BinaryOperatorPrecedence(36),
            BinaryOperator::BitwiseXor => BinaryOperatorPrecedence(34),
            BinaryOperator::BitwiseOr => BinaryOperatorPrecedence(32),

            BinaryOperator::And => BinaryOperatorPrecedence(30),
            BinaryOperator::Or => BinaryOperatorPrecedence(28),
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpressionKind<'a> {
    IntConstant(&'a str, Radix),
    Unary(UnaryOperator, Box<Expression<'a>>),
    Binary(BinaryOperator, Box<Expression<'a>>, Box<Expression<'a>>),
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Expression<'a> {
    pub(crate) location: Location,
    pub kind: ExpressionKind<'a>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimitiveKind {
    Integer,
    UnsignedInteger,
    LongInteger,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeExpressionKind {
    Primitive(PrimitiveKind),
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TypeExpression {
    pub location: Location,
    pub kind: TypeExpressionKind,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StatementKind<'a> {
    Return(Expression<'a>),
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Statement<'a> {
    pub location: Location,
    pub kind: StatementKind<'a>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct FunctionDefinition<'a> {
    pub location: Location,
    pub name: Symbol<'a>,
    pub body: Vec<Statement<'a>>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ProgramDefinition<'a> {
    pub functions: Vec<FunctionDefinition<'a>>,
}