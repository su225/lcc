use std::fmt::Debug;

use derive_more::with_trait::Display;
use crate::parser::types::*;
use crate::tacky::errors::TackyError;
use crate::tacky::types::{TackyInstruction, TackyBinaryOperator, TackyProgram, TackyUnaryOperator, TackyValue, TackyFunction};
use crate::tacky::types::TackyInstruction::{Binary, Return, Unary};
use crate::tacky::types::TackyValue::Constant32;

const COMPILER_GEN_PREFIX: &'static str = "<t>";

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

struct TackyContext {
    next_int: i64,
}

impl TackyContext {
    fn new() -> TackyContext {
        TackyContext { next_int: 0 }
    }

    fn next_temporary_identifier(&mut self) -> TackySymbol {
        let identifier = format!("{}.{}", COMPILER_GEN_PREFIX, self.next_int);
        self.next_int += 1;
        TackySymbol(identifier)
    }
}

pub fn emit(prog: &ProgramDefinition) -> Result<TackyProgram, TackyError> {
    let mut ctx = TackyContext::new();
    let mut f = vec![];
    for fd in prog.functions.iter() {
        let tf = emit_tacky_for_function(&mut ctx, fd)?;
        f.push(tf);
    }
    Ok(TackyProgram { functions: f })
}

fn emit_tacky_for_function(ctx: &mut TackyContext, f: &FunctionDefinition) -> Result<TackyFunction, TackyError> {
    let mut instructions = vec![];
    for stmt in f.body.iter() {
        let instrs = emit_tacky_for_statement(ctx, stmt)?;
        instructions.extend(instrs);
    }
    Ok(TackyFunction {
        identifier: TackySymbol(f.name.name.into()),
        body: instructions,
    })
}

fn emit_tacky_for_statement(ctx: &mut TackyContext, s: &Statement) -> Result<Vec<TackyInstruction>, TackyError> {
    match s.kind {
        StatementKind::Return(ref expr) => {
            let (dst, mut expr_instrs) = emit_tacky_for_expression(ctx, expr)?;
            expr_instrs.push(Return(dst));
            Ok(expr_instrs)
        }
    }
}

fn emit_tacky_for_expression(ctx: &mut TackyContext, e: &Expression) -> Result<(TackyValue, Vec<TackyInstruction>), TackyError> {
    match &e.kind {
        ExpressionKind::IntConstant(c, radix) => {
            let n = i32::from_str_radix(c, radix.value())?;
            Ok((Constant32(n), vec![]))
        }
        ExpressionKind::Unary(unary_op, src) => {
            let (src_tacky, mut tacky_instrs) = emit_tacky_for_expression(ctx, src)?;
            let dst_tacky_identifier = ctx.next_temporary_identifier();
            let dst_tacky = TackyValue::Variable(dst_tacky_identifier.clone());
            let result_val = TackyValue::Variable(dst_tacky_identifier);
            tacky_instrs.push(Unary {
                operator: TackyUnaryOperator::from(unary_op),
                src: src_tacky,
                dst: dst_tacky,
            });
            Ok((result_val, tacky_instrs))
        }
        ExpressionKind::Binary(binary_op, op1, op2) => {
            let (src1_tacky, mut tacky_instrs) = emit_tacky_for_expression(ctx, op1)?;
            let (src2_tacky, src2_tacky_instrs) = emit_tacky_for_expression(ctx, op2)?;
            let dst_tacky_identifier = ctx.next_temporary_identifier();
            let dst_tacky = TackyValue::Variable(dst_tacky_identifier.clone());
            let result = TackyValue::Variable(dst_tacky_identifier);
            tacky_instrs.extend(src2_tacky_instrs);
            tacky_instrs.push(Binary {
                operator: TackyBinaryOperator::from(binary_op),
                src1: src1_tacky,
                src2: src2_tacky,
                dst: dst_tacky,
            });
            Ok((result, tacky_instrs))
        }
    }
}
