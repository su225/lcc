use std::fmt::Debug;
use std::num::ParseIntError;

use derive_more::with_trait::Display;
use thiserror::Error;

use crate::parser::{BinaryOperator, Expression, ExpressionKind, FunctionDefinition, ProgramDefinition, Statement, StatementKind, UnaryOperator};
use crate::tacky::Instruction::{Binary, Return, Unary};
use crate::tacky::IRValue::Constant;

#[derive(Debug, PartialEq)]
pub(crate) struct IRProgram {
    pub functions: Vec<IRFunction>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct IRFunction {
    pub identifier: IRSymbol,
    pub body: Vec<Instruction>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum IRUnaryOperator {
    Complement,
    Negate,
}

impl From<&UnaryOperator> for IRUnaryOperator {
    fn from(value: &UnaryOperator) -> Self {
        match value {
            UnaryOperator::Complement => IRUnaryOperator::Complement,
            UnaryOperator::Negate => IRUnaryOperator::Negate,
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum IRBinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
}

impl From<&BinaryOperator> for IRBinaryOperator {
    fn from(value: &BinaryOperator) -> Self {
        match value {
            BinaryOperator::Add => IRBinaryOperator::Add,
            BinaryOperator::Subtract => IRBinaryOperator::Subtract,
            BinaryOperator::Multiply => IRBinaryOperator::Multiply,
            BinaryOperator::Divide => IRBinaryOperator::Divide,
            BinaryOperator::Modulo => IRBinaryOperator::Modulo,
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum IRValue {
    Constant(i64),
    Variable(IRSymbol),
}

#[derive(Debug, PartialEq)]
pub(crate) enum Instruction {
    Unary {
        operator: IRUnaryOperator,
        src: IRValue,
        dst: IRValue,
    },
    Binary {
        operator: IRBinaryOperator,
        src1: IRValue,
        src2: IRValue,
        dst: IRValue,
    },
    Return(IRValue),
}

#[derive(Error, PartialEq, Debug)]
pub(crate) enum TackyError {
    #[error(transparent)]
    IntImmediateParseError(#[from] ParseIntError),
}

const COMPILER_GEN_PREFIX: &'static str = "<t>";

#[derive(Debug, Eq, PartialEq, Clone, Hash, Display)]
pub(crate) struct IRSymbol(pub String);

impl From<&str> for IRSymbol {
    fn from(value: &str) -> Self {
        IRSymbol(value.to_string())
    }
}

impl IRSymbol {
    pub fn is_generated(&self) -> bool {
        self.0.starts_with(COMPILER_GEN_PREFIX)
    }
}

struct TackyContext {
    next_int: i64,
}

impl TackyContext {
    fn new() -> TackyContext {
        TackyContext { next_int: 0 }
    }

    fn next_temporary_identifier(&mut self) -> IRSymbol {
        let identifier = format!("{}.{}", COMPILER_GEN_PREFIX, self.next_int);
        self.next_int += 1;
        IRSymbol(identifier)
    }
}

pub fn emit(prog: &ProgramDefinition) -> Result<IRProgram, TackyError> {
    let mut ctx = TackyContext::new();
    let mut f = vec![];
    for fd in prog.functions.iter() {
        let tf = emit_tacky_for_function(&mut ctx, fd)?;
        f.push(tf);
    }
    Ok(IRProgram { functions: f })
}

fn emit_tacky_for_function(ctx: &mut TackyContext, f: &FunctionDefinition) -> Result<IRFunction, TackyError> {
    let mut instructions = vec![];
    for stmt in f.body.iter() {
        let instrs = emit_tacky_for_statement(ctx, stmt)?;
        instructions.extend(instrs);
    }
    Ok(IRFunction {
        identifier: IRSymbol(f.name.name.into()),
        body: instructions,
    })
}

fn emit_tacky_for_statement(ctx: &mut TackyContext, s: &Statement) -> Result<Vec<Instruction>, TackyError> {
    match s.kind {
        StatementKind::Return(ref expr) => {
            let (dst, mut expr_instrs) = emit_tacky_for_expression(ctx, expr)?;
            expr_instrs.push(Return(dst));
            Ok(expr_instrs)
        },
    }
}

fn emit_tacky_for_expression(ctx: &mut TackyContext, e: &Expression) -> Result<(IRValue, Vec<Instruction>), TackyError> {
    match &e.kind {
        ExpressionKind::IntConstant(c, radix) => {
            let n = i64::from_str_radix(c, radix.value())?;
            Ok((Constant(n), vec![]))
        }
        ExpressionKind::Unary(unary_op, src) => {
            let (src_tacky, mut tacky_instrs) = emit_tacky_for_expression(ctx, src)?;
            let dst_tacky_identifier = ctx.next_temporary_identifier();
            let dst_tacky = IRValue::Variable(dst_tacky_identifier.clone());
            let result_val = IRValue::Variable(dst_tacky_identifier);
            tacky_instrs.push(Unary {
                operator: IRUnaryOperator::from(unary_op),
                src: src_tacky,
                dst: dst_tacky,
            });
            Ok((result_val, tacky_instrs))
        }
        ExpressionKind::Binary(binary_op, op1, op2) => {
            let (src1_tacky, mut tacky_instrs) = emit_tacky_for_expression(ctx, op1)?;
            let (src2_tacky, src2_tacky_instrs) = emit_tacky_for_expression(ctx, op2)?;
            let dst_tacky_identifier = ctx.next_temporary_identifier();
            let dst_tacky = IRValue::Variable(dst_tacky_identifier.clone());
            let result = IRValue::Variable(dst_tacky_identifier);
            tacky_instrs.extend(src2_tacky_instrs);
            tacky_instrs.push(Binary {
                operator: IRBinaryOperator::from(binary_op),
                src1: src1_tacky,
                src2: src2_tacky,
                dst: dst_tacky,
            });
            Ok((result, tacky_instrs))
        },
    }
}
