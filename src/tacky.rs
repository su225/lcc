use std::num::ParseIntError;
use thiserror::Error;

use crate::parser::{Expression, ExpressionKind, FunctionDefinition, ProgramDefinition, Statement, StatementKind};
use crate::tacky::Instruction::{Return, Unary};
use crate::tacky::Value::Constant;

#[derive(Debug, PartialEq)]
pub(crate) struct Program {
    functions: Vec<Function>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Function {
    identifier: Symbol,
    body: Vec<Instruction>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum UnaryOperator {
    Complement,
    Negate,
}

impl From<&crate::parser::UnaryOperator> for UnaryOperator {
    fn from(value: &crate::parser::UnaryOperator) -> Self {
        match value {
            crate::parser::UnaryOperator::Complement => UnaryOperator::Complement,
            crate::parser::UnaryOperator::Negate => UnaryOperator::Negate,
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum Value {
    Constant(i64),
    Variable(Symbol),
}

#[derive(Debug, PartialEq)]
pub(crate) enum Instruction {
    Return(Value),
    Unary {
        operator: UnaryOperator,
        src: Value,
        dst: Value,
    }
}

#[derive(Error, PartialEq, Debug)]
pub(crate) enum TackyError {
    #[error(transparent)]
    IntImmediateParseError(#[from] ParseIntError),
}

const COMPILER_GEN_PREFIX: &'static str = "<t>";

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Symbol(String);

impl Symbol {
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

    fn next_temporary_identifier(&mut self) -> Symbol {
        let identifier = format!("{}.{}", COMPILER_GEN_PREFIX, self.next_int);
        self.next_int += 1;
        Symbol(identifier)
    }
}

pub fn emit(prog: &ProgramDefinition) -> Result<Program, TackyError> {
    let mut ctx = TackyContext::new();
    let mut f = vec![];
    for fd in prog.functions.iter() {
        let tf = emit_tacky_for_function(&mut ctx, fd)?;
        f.push(tf);
    }
    Ok(Program { functions: f })
}

fn emit_tacky_for_function(ctx: &mut TackyContext, f: &FunctionDefinition) -> Result<Function, TackyError> {
    let mut instructions = vec![];
    for stmt in f.body.iter() {
        let instrs = emit_tacky_for_statement(ctx, stmt)?;
        instructions.extend(instrs);
    }
    Ok(Function {
        identifier: Symbol(f.name.name.into()),
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

fn emit_tacky_for_expression(ctx: &mut TackyContext, e: &Expression) -> Result<(Value, Vec<Instruction>), TackyError> {
    match &e.kind {
        ExpressionKind::IntConstant(c, radix) => {
            let n = i64::from_str_radix(c, radix.value())?;
            Ok((Constant(n), vec![]))
        }
        ExpressionKind::Unary(unary_op, src) => {
            let (src_tacky, mut tacky_instrs) = emit_tacky_for_expression(ctx, src)?;
            let dst_tacky_identifier = ctx.next_temporary_identifier();
            let dst_tacky = Value::Variable(dst_tacky_identifier.clone());
            let result_val = Value::Variable(dst_tacky_identifier);
            tacky_instrs.push(Unary {
                operator: UnaryOperator::from(unary_op),
                src: src_tacky,
                dst: dst_tacky,
            });
            Ok((result_val, tacky_instrs))
        }
    }
}
