use std::num::ParseIntError;
use thiserror::Error;

use crate::parser::{Expression, ExpressionKind, FunctionDefinition, ProgramDefinition, Statement, StatementKind};
use crate::tacky_emit::Instruction::{Return, Unary};
use crate::tacky_emit::Value::Constant;

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

pub(crate) struct TackyEmitter<'a> {
    next_int: u64,
    program: ProgramDefinition<'a>,
}

const COMPILER_GEN_PREFIX: &'static str = "<t>";

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Symbol(String);

impl Symbol {
    pub fn is_generated(&self) -> bool {
        self.0.starts_with(COMPILER_GEN_PREFIX)
    }
}

impl<'a> TackyEmitter<'a> {
    pub fn new(ast: ProgramDefinition<'a>) -> TackyEmitter<'a> {
        TackyEmitter {
            next_int: 0,
            program: ast,
        }
    }

    pub fn emit_tacky(&mut self) -> Result<Program, TackyError> {
        let mut f = vec![];
        for fd in self.program.functions.iter() {
            let tf = self.emit_tacky_for_function(fd)?;
            f.push(tf);
        }
        Ok(Program { functions: f })
    }

    fn emit_tacky_for_function(&mut self, f: &FunctionDefinition) -> Result<Function, TackyError> {
        let mut instructions = vec![];
        for stmt in f.body.iter() {
            let instrs = self.emit_tacky_for_statement(stmt)?;
            instructions.extend(instrs);
        }
        Ok(Function {
            identifier: Symbol(f.name.name.into()),
            body: instructions,
        })
    }

    fn emit_tacky_for_statement(&mut self, s: &Statement) -> Result<Vec<Instruction>, TackyError> {
        match s.kind {
            StatementKind::Return(ref expr) => {
                let (dst, mut expr_instrs) = self.emit_tacky_for_expression(expr)?;
                expr_instrs.push(Return(dst));
                Ok(expr_instrs)
            },
        }
    }

    fn emit_tacky_for_expression(&mut self, e: &Expression) -> Result<(Value, Vec<Instruction>), TackyError> {
        match &e.kind {
            ExpressionKind::IntConstant(c, radix) => {
                let n = i64::from_str_radix(c, radix.value())?;
                Ok((Constant(n), vec![]))
            }
            ExpressionKind::Unary(unary_op, src) => {
                let (src_tacky, mut tacky_instrs) = self.emit_tacky_for_expression(src)?;
                let dst_tacky_identifier = self.next_temporary_identifier();
                let dst_tacky = Value::Variable(dst_tacky_identifier);
                tacky_instrs.push(Unary {
                    operator: UnaryOperator::from(unary_op),
                    src: src_tacky,
                    dst: dst_tacky,
                });
                Ok((Value::Variable(dst_tacky_identifier.clone()), tacky_instrs))
            }
        }
    }

    fn next_temporary_identifier(&mut self) -> Symbol {
        let identifier = format!("{}.{}", COMPILER_GEN_PREFIX, self.next_int);
        self.next_int += 1;
        Symbol(identifier)
    }
}