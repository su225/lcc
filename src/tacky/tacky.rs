use std::fmt::{Display, Formatter};
use std::num::ParseIntError;
use std::vec::Vec;

use derive_more::Display;
use thiserror::Error;
use itertools::Itertools;

use crate::parser::{BinaryOperator, BlockItem, Declaration, DeclarationKind, Expression, ExpressionKind, ForInit, Function, Program, Statement, StatementKind, Symbol, UnaryOperator, VariableDeclaration};
use crate::tacky::TackyInstruction::*;
use crate::tacky::TackyValue::{Int32, Variable};

pub(crate) const COMPILER_GEN_PREFIX: &'static str = "<t>";
pub(crate) const COMPILER_GEN_LABEL_PREFIX: &'static str = "_L";

#[derive(Debug, Eq, PartialEq, Clone, Hash, Display)]
pub struct TackySymbol(pub String);

impl From<&str> for TackySymbol {
    fn from(value: &str) -> Self {
        TackySymbol(value.to_string())
    }
}

impl From<&String> for TackySymbol {
    fn from(value: &String) -> Self {
        TackySymbol(value.to_string())
    }
}

impl From<&Symbol> for TackySymbol {
    fn from(value: &Symbol) -> Self {
        TackySymbol(value.name.to_string())
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

impl Display for TackyProgram {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for tacky_func in self.functions.iter() {
            tacky_func.fmt(f)?;
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct TackyFunctionParameter {
    pub name: TackySymbol,
}

#[derive(Debug, PartialEq)]
pub struct TackyFunction {
    pub identifier: TackySymbol,
    pub body: Vec<TackyInstruction>,
    pub params: Vec<TackyFunctionParameter>,
}

impl Display for TackyFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("int {}(void) {{\n", self.identifier.0))?;
        for instr in self.body.iter() {
            f.write_fmt(format_args!("{}\n", instr))?;
        }
        f.write_str("}\n\n")?;
        Ok(())
    }
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
            UnaryOperator::Not => TackyUnaryOperator::Not,
            UnaryOperator::Increment | UnaryOperator::Decrement =>
                panic!("++ and -- must be desugared before IR generation")
        }
    }
}

impl Display for TackyUnaryOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TackyUnaryOperator::Complement => f.write_str("~"),
            TackyUnaryOperator::Negate => f.write_str("-"),
            TackyUnaryOperator::Not => f.write_str("!"),
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

impl Display for TackyBinaryOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TackyBinaryOperator::Add => f.write_str("+"),
            TackyBinaryOperator::Subtract => f.write_str("-"),
            TackyBinaryOperator::Multiply => f.write_str("*"),
            TackyBinaryOperator::Divide => f.write_str("/"),
            TackyBinaryOperator::Modulo => f.write_str("%"),
            TackyBinaryOperator::BitwiseAnd => f.write_str("&"),
            TackyBinaryOperator::BitwiseOr => f.write_str("|"),
            TackyBinaryOperator::BitwiseXor => f.write_str("^"),
            TackyBinaryOperator::LeftShift => f.write_str("<<"),
            TackyBinaryOperator::RightShift => f.write_str(">>"),
            TackyBinaryOperator::Equal => f.write_str("=="),
            TackyBinaryOperator::NotEqual => f.write_str("!="),
            TackyBinaryOperator::LessThan => f.write_str("<"),
            TackyBinaryOperator::LessOrEqual => f.write_str("<="),
            TackyBinaryOperator::GreaterThan => f.write_str(">"),
            TackyBinaryOperator::GreaterOrEqual => f.write_str(">="),
        }
    }
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

            BinaryOperator::And => panic!("logical AND cannot be translated as binary operator"),
            BinaryOperator::Or => panic!("logical OR cannot be translated as binary operator"),
            BinaryOperator::Equal => TackyBinaryOperator::Equal,
            BinaryOperator::NotEqual => TackyBinaryOperator::NotEqual,
            BinaryOperator::LessThan => TackyBinaryOperator::LessThan,
            BinaryOperator::LessThanOrEqual => TackyBinaryOperator::LessOrEqual,
            BinaryOperator::GreaterThan => TackyBinaryOperator::GreaterThan,
            BinaryOperator::GreaterThanOrEqual => TackyBinaryOperator::GreaterOrEqual,

            BinaryOperator::Assignment => panic!("Assignment cannot be translated as binary operator"),
            BinaryOperator::CompoundAssignment(_) => panic!("Compound assignment operator cannot be translated as binary operator"),
            BinaryOperator::TernaryThen => panic!("Ternary then cannot be translated as binary operator")
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TackyValue {
    Int32(i32),
    Variable(TackySymbol),
}

impl Display for TackyValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Int32(c) => f.write_fmt(format_args!("(int32 {})", c)),
            Variable(var) => f.write_str(&(var.0.clone())),
        }
    }
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
    Copy { src: TackyValue, dst: TackySymbol },
    Jump { target: TackySymbol },
    JumpIfZero { condition: TackyValue, target: TackySymbol },
    JumpIfNotZero { condition: TackyValue, target: TackySymbol },
    Label(TackySymbol),
    FunctionCall { func_name: TackySymbol, args: Vec<TackyValue>, dst: TackyValue },
}

impl TackyInstruction {
    fn is_return(&self) -> bool {
        match self {
            Return(_) => true,
            _ => false,
        }
    }
}

impl Display for TackyInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Unary { operator, src, dst } => {
                f.write_fmt(format_args!("    {dst} = {operator} {src};"))
            },
            Binary { operator, src1, src2, dst } => {
                f.write_fmt(format_args!("    {dst} = {src1} {operator} {src2};"))
            },
            Return(ret_value) => {
                f.write_fmt(format_args!("    return {ret_value};"))
            },
            Copy { src, dst } => {
                f.write_fmt(format_args!("    {dst} = {src};"))
            },
            Jump { target } => {
                f.write_fmt(format_args!("    jump {target};"))
            },
            JumpIfZero { condition, target } => {
                f.write_fmt(format_args!("    jump_if_zero ({condition}) {target};"))
            }
            JumpIfNotZero { condition, target } => {
                f.write_fmt(format_args!("    jump_if_not_zero ({condition}) {target};"))
            },
            Label(lbl) => f.write_fmt(format_args!("{}:", lbl.0)),
            FunctionCall { func_name, args, dst } => f.write_fmt(format_args!("    {dst} = func_call ({func_name}) (args:{a})",
                                                                              a = args.iter().map(|v| v.to_string()).join(","))),
        }
    }
}

#[derive(Error, PartialEq, Debug)]
pub enum TackyError {
    #[error(transparent)]
    IntImmediateParseError(#[from] ParseIntError),
}

struct TackyContext {
    next_int: i64,
    next_label_int: i64,
}

impl TackyContext {
    fn new() -> TackyContext {
        TackyContext { next_int: 0, next_label_int: 0 }
    }

    fn next_temporary_identifier(&mut self) -> TackySymbol {
        let identifier = format!("{}.{}", COMPILER_GEN_PREFIX, self.next_int);
        self.next_int += 1;
        TackySymbol(identifier)
    }

    fn next_temporary_label(&mut self, prefix: &str) -> TackySymbol {
        if prefix.is_empty() {
            panic!("empty prefix is not allowed for temp label")
        }
        let identifier = format!("{}.{}.{}", COMPILER_GEN_LABEL_PREFIX, prefix, self.next_label_int);
        self.next_label_int += 1;
        TackySymbol(identifier)
    }
}

pub fn emit(prog: &Program) -> Result<TackyProgram, TackyError> {
    let mut ctx = TackyContext::new();
    let mut tacky_funcs = vec![];
    for decl in prog.declarations.iter() {
        match &decl.kind {
            DeclarationKind::FunctionDeclaration(ref f) => {
                if f.body.is_none() {
                    continue;
                }
                let tf = emit_tacky_for_function(&mut ctx, f)?;
                tacky_funcs.push(tf);
            }
            DeclarationKind::VarDeclaration { .. } => unimplemented!("global vars not yet supported")
        }
    }
    Ok(TackyProgram { functions: tacky_funcs })
}

fn emit_tacky_for_function(ctx: &mut TackyContext, f: &Function) -> Result<TackyFunction, TackyError> {
    let mut instructions = vec![];
    for blk_item in f.body.as_ref().unwrap().items.iter() {
        let instrs = emit_tacky_for_block_item(ctx, blk_item)?;
        instructions.extend(instrs);
    }
    if instructions.is_empty() || !instructions.last().unwrap().is_return() {
        instructions.push(Return(Int32(0)));
    }
    Ok(TackyFunction {
        identifier: TackySymbol(f.name.name.clone()),
        body: instructions,
        params: f.params.iter().map(|p| TackyFunctionParameter {
            name: TackySymbol::from(&f.name),
        }).collect_vec(),
    })
}

fn emit_tacky_for_block_item(ctx: &mut TackyContext, blk_item: &BlockItem) -> Result<Vec<TackyInstruction>, TackyError> {
    match blk_item {
        BlockItem::Statement(stmt) => emit_tacky_for_statement(ctx, stmt),
        BlockItem::Declaration(decl) => emit_tacky_for_declaration(ctx, decl),
    }
}

fn emit_tacky_for_declaration(ctx: &mut TackyContext, decl: &Declaration) -> Result<Vec<TackyInstruction>, TackyError> {
    match &decl.kind {
        DeclarationKind::VarDeclaration(v) => emit_tacky_for_variable_declaration(ctx, v),
        DeclarationKind::FunctionDeclaration(_) => {
            // We do nothing here because functions can only be defined at the top-level, and
            // it already calls emit_tacky_for_function to get TackyFunction. We reach here only
            // when we are in a function body. If there is a definition inside another function body,
            // then it is already an ERROR caught in the earlier phases. We won't have a body and
            // hence, there is NOTHING to do.
            Ok(vec![])
        },
    }
}

fn emit_tacky_for_variable_declaration(ctx: &mut TackyContext, var_dec: &VariableDeclaration) -> Result<Vec<TackyInstruction>, TackyError> {
    let identifier = &var_dec.identifier;
    match &var_dec.init_expression {
        None => Ok(vec![]),
        Some(expr) => {
            let (tacky_val, mut expr_tacky) = emit_tacky_for_expression(ctx, expr)?;
            expr_tacky.push(Copy { src: tacky_val, dst: TackySymbol(identifier.name.clone()) });
            Ok(expr_tacky)
        },
    }
}

fn emit_tacky_for_statement(ctx: &mut TackyContext, s: &Statement) -> Result<Vec<TackyInstruction>, TackyError> {
    let mut instrs = vec![];
    for lbl in s.labels.iter() {
        instrs.push(Label(TackySymbol::from(lbl)));
    }
    match &s.kind {
        StatementKind::Return(ref expr) => {
            let (dst, expr_instrs) = emit_tacky_for_expression(ctx, expr)?;
            instrs.extend(expr_instrs);
            instrs.push(Return(dst));
        }
        StatementKind::SubBlock(block) => {
            for blk_item in block.items.iter() {
                let blk_instrs = emit_tacky_for_block_item(ctx, blk_item)?;
                instrs.extend(blk_instrs);
            }
        }
        StatementKind::Expression(e) => {
            let (_, expr_instrs) = emit_tacky_for_expression(ctx, e)?;
            instrs.extend(expr_instrs)
        },
        StatementKind::If { condition, then_statement, else_statement } => {
            let (cond_res, cond_instrs) = emit_tacky_for_expression(ctx, condition)?;
            instrs.extend(cond_instrs);

            let then_stmts = emit_tacky_for_statement(ctx, then_statement)?;
            let else_stmts = else_statement.as_ref()
                .map(|else_stmt| emit_tacky_for_statement(ctx, else_stmt))
                .unwrap_or_else(|| Ok(vec![]))?;
            let end_lbl = ctx.next_temporary_label("if_end");
            if else_stmts.is_empty() {
                let tmp_cond_res = ctx.next_temporary_identifier();
                instrs.push(Copy { src: cond_res, dst: tmp_cond_res.clone() });
                instrs.push(JumpIfZero {
                    condition: Variable(tmp_cond_res),
                    target: end_lbl.clone(),
                });
                instrs.extend(then_stmts);
            } else {
                let else_lbl = ctx.next_temporary_label("if_else");
                let tmp_cond_res = ctx.next_temporary_identifier();
                instrs.push(Copy { src: cond_res, dst: tmp_cond_res.clone() });
                instrs.push(JumpIfZero {
                    condition: Variable(tmp_cond_res),
                    target: else_lbl.clone(),
                });
                instrs.extend(then_stmts);
                instrs.push(Jump { target: end_lbl.clone() });

                instrs.push(Label(else_lbl.clone()));
                instrs.extend(else_stmts);
            }
            instrs.push(Label(end_lbl));
        },
        StatementKind::Goto {target} => {
            instrs.push(Jump { target: TackySymbol::from(target) });
        },
        StatementKind::Null => {},
        StatementKind::While { pre_condition, loop_body, loop_label } => {
            debug_assert!(loop_label.is_some());
            let loop_continue_label = loop_continue_symbol(TackySymbol::from(loop_label.as_ref().unwrap()));
            let loop_break_label = loop_break_symbol(TackySymbol::from(loop_label.as_ref().unwrap()));
            instrs.push(Label(loop_continue_label.clone()));

            let (cond_res, cond_instrs) = emit_tacky_for_expression(ctx, &*pre_condition)?;
            instrs.extend(cond_instrs);
            let tmp_cond_res = ctx.next_temporary_identifier();
            instrs.push(Copy { src: cond_res, dst: tmp_cond_res.clone() });
            instrs.push(JumpIfZero { condition: Variable(tmp_cond_res), target: loop_break_label.clone() });

            let body_instrs = emit_tacky_for_statement(ctx, &*loop_body)?;
            instrs.extend(body_instrs);
            instrs.push(Jump { target: loop_continue_label });
            instrs.push(Label(loop_break_label));
        },
        StatementKind::DoWhile { loop_body, post_condition, loop_label } => {
            debug_assert!(loop_label.is_some());
            let loop_start_label = loop_start_symbol(TackySymbol::from(loop_label.as_ref().unwrap()));
            instrs.push(Label(loop_start_label.clone()));

            let body_instrs = emit_tacky_for_statement(ctx, &*loop_body)?;
            instrs.extend(body_instrs);

            instrs.push(Label(loop_continue_symbol(TackySymbol::from(loop_label.as_ref().unwrap()))));
            let (cond_res, cond_instrs) = emit_tacky_for_expression(ctx, &*post_condition)?;
            instrs.extend(cond_instrs);
            let tmp_cond_res = ctx.next_temporary_identifier();
            instrs.push(Copy { src: cond_res, dst: tmp_cond_res.clone() });
            instrs.push(JumpIfNotZero { condition: Variable(tmp_cond_res), target: loop_start_label });
            instrs.push(Label(loop_break_symbol(TackySymbol::from(loop_label.as_ref().unwrap()))));
        },
        StatementKind::For { init, condition, post, loop_body, loop_label } => {
            debug_assert!(loop_label.is_some());
            let lbl = loop_label.as_ref().unwrap();
            let loop_start_label = loop_start_symbol(TackySymbol::from(lbl));
            let loop_continue_label = loop_continue_symbol(TackySymbol::from(lbl));
            let loop_break_label = loop_break_symbol(TackySymbol::from(lbl));

            let init_instrs = emit_tacky_for_forloop_init(ctx, init)?;
            instrs.extend(init_instrs);

            instrs.push(Label(loop_start_label.clone()));
            if let Some(pre_condition) = condition {
                let (pre_cond_res, pre_cond_instrs) = emit_tacky_for_expression(ctx, &*pre_condition)?;
                let tmp_pre_cond_res = ctx.next_temporary_identifier();
                instrs.extend(pre_cond_instrs);
                instrs.push(Copy { src: pre_cond_res, dst: tmp_pre_cond_res.clone() });
                instrs.push(JumpIfZero {condition: Variable(tmp_pre_cond_res), target: loop_break_label.clone() });
            }
            let loop_body_instrs = emit_tacky_for_statement(ctx, &*loop_body)?;
            instrs.extend(loop_body_instrs);
            instrs.push(Label(loop_continue_label.clone()));
            if let Some(post_expr) = post {
                let (_, post_expr_instrs) = emit_tacky_for_expression(ctx, &*post_expr)?;
                instrs.extend(post_expr_instrs);
            }
            instrs.push(Jump {target: loop_start_label});
            instrs.push(Label(loop_break_label));
        },
        StatementKind::Break(loop_label) => {
            let break_label = loop_break_symbol(TackySymbol::from(loop_label.as_ref().unwrap()));
            instrs.push(Jump {target: break_label});
        },
        StatementKind::Continue(loop_label) => {
            let continue_label = loop_continue_symbol(TackySymbol::from(loop_label.as_ref().unwrap()));
            instrs.push(Jump { target: continue_label })
        },
    };
    Ok(instrs)
}

fn emit_tacky_for_forloop_init(ctx: &mut TackyContext, for_init: &ForInit) -> Result<Vec<TackyInstruction>, TackyError> {
    match &for_init {
        ForInit::InitDecl(decl) => emit_tacky_for_variable_declaration(ctx, &*decl),
        ForInit::InitExpr(expr) => {
            let (_, init_expr_instrs) = emit_tacky_for_expression(ctx, &*expr)?;
            Ok(init_expr_instrs)
        }
        ForInit::Null => Ok(vec![])
    }
}

#[inline]
fn loop_start_symbol(s: TackySymbol) -> TackySymbol {
    TackySymbol::from(format!("{}.start", s.0).as_str())
}

#[inline]
fn loop_continue_symbol(s: TackySymbol) -> TackySymbol {
    TackySymbol::from(format!("{}.continue", s.0).as_str())
}

#[inline]
fn loop_break_symbol(s: TackySymbol) -> TackySymbol {
    TackySymbol::from(format!("{}.break", s.0).as_str())
}

fn emit_tacky_for_expression(ctx: &mut TackyContext, e: &Expression) -> Result<(TackyValue, Vec<TackyInstruction>), TackyError> {
    match &e.kind {
        ExpressionKind::IntConstant(c, radix) => {
            let n = i32::from_str_radix(c, radix.value())?;
            Ok((Int32(n), vec![]))
        }
        ExpressionKind::Variable(v) => Ok((Variable(TackySymbol::from(v)), vec![])),
        ExpressionKind::Unary(unary_op, src) => {
            let (src_tacky, mut tacky_instrs) = emit_tacky_for_expression(ctx, src)?;
            let dst_tacky_identifier = ctx.next_temporary_identifier();
            let dst_tacky = Variable(dst_tacky_identifier.clone());
            let result_val = Variable(dst_tacky_identifier);
            tacky_instrs.push(Unary {
                operator: TackyUnaryOperator::from(unary_op),
                src: src_tacky,
                dst: dst_tacky,
            });
            Ok((result_val, tacky_instrs))
        }
        ExpressionKind::Binary(BinaryOperator::And, op1, op2) => {
            let mut tacky_instrs = vec![];
            let result = ctx.next_temporary_identifier();
            let temp_false_label = ctx.next_temporary_label("and_false");
            let temp_end_label = ctx.next_temporary_label("and_end");

            let (e1_tacky_result, e1_tacky_instrs) = emit_tacky_for_expression(ctx, op1)?;
            tacky_instrs.extend(e1_tacky_instrs);
            let v1 = ctx.next_temporary_identifier();
            tacky_instrs.extend(vec![
                Copy { src: e1_tacky_result, dst: v1.clone() },
                JumpIfZero { condition: Variable(v1), target: temp_false_label.clone() },
            ]);

            let (e2_tacky_result, e2_tacky_instrs) = emit_tacky_for_expression(ctx, op2)?;
            tacky_instrs.extend(e2_tacky_instrs);
            let v2 = ctx.next_temporary_identifier();
            tacky_instrs.extend(vec![
                Copy { src: e2_tacky_result, dst: v2.clone() },
                JumpIfZero { condition: Variable(v2), target: temp_false_label.clone() },
            ]);

            // All expressions are true
            tacky_instrs.extend(vec![
                Copy { src: Int32(1), dst: result.clone() },
                Jump { target: temp_end_label.clone() }
            ]);
            tacky_instrs.extend(vec![
                Label(temp_false_label),
                Copy { src: Int32(0), dst: result.clone() },
            ]);
            tacky_instrs.push(Label(temp_end_label));

            Ok((Variable(result), tacky_instrs))
        },
        ExpressionKind::Binary(BinaryOperator::Or, op1, op2) => {
            let mut tacky_instrs = vec![];
            let result = ctx.next_temporary_identifier();
            let temp_true_label = ctx.next_temporary_label("or_true");
            let temp_end_label = ctx.next_temporary_label("or_end");

            let (e1_tacky_result, e1_tacky_instrs) = emit_tacky_for_expression(ctx, op1)?;
            tacky_instrs.extend(e1_tacky_instrs);
            let v1 = ctx.next_temporary_identifier();
            tacky_instrs.extend(vec![
                Copy { src: e1_tacky_result, dst: v1.clone() },
                JumpIfNotZero { condition: Variable(v1), target: temp_true_label.clone() },
            ]);

            let (e2_tacky_result, e2_tacky_instrs) = emit_tacky_for_expression(ctx, op2)?;
            tacky_instrs.extend(e2_tacky_instrs);
            let v2 = ctx.next_temporary_identifier();
            tacky_instrs.extend(vec![
                Copy { src: e2_tacky_result, dst: v2.clone() },
                JumpIfNotZero { condition: Variable(v2), target: temp_true_label.clone() },
            ]);

            // All expressions are false
            tacky_instrs.extend(vec![
                Copy { src: Int32(0), dst: result.clone() },
                Jump { target: temp_end_label.clone() },
            ]);
            tacky_instrs.extend(vec![
                Label(temp_true_label),
                Copy { src: Int32(1), dst: result.clone() },
            ]);
            tacky_instrs.push(Label(temp_end_label));

            Ok((Variable(result), tacky_instrs))
        },
        ExpressionKind::Binary(binary_op, op1, op2) => {
            let (src1_tacky, mut tacky_instrs) = emit_tacky_for_expression(ctx, op1)?;
            let (src2_tacky, src2_tacky_instrs) = emit_tacky_for_expression(ctx, op2)?;
            let dst_tacky_identifier = ctx.next_temporary_identifier();
            let dst_tacky = Variable(dst_tacky_identifier.clone());
            tacky_instrs.extend(src2_tacky_instrs);
            tacky_instrs.push(Binary {
                operator: TackyBinaryOperator::from(binary_op),
                src1: src1_tacky,
                src2: src2_tacky,
                dst: dst_tacky.clone(),
            });
            Ok((dst_tacky, tacky_instrs))
        },
        ExpressionKind::Assignment { lvalue, rvalue, op} => {
            debug_assert!(op.is_none(), "compound assignment not desugared before IR generation");
            debug_assert!(lvalue.kind.is_lvalue_expression());
            let (rhs_tacky, mut instrs) = emit_tacky_for_expression(ctx, rvalue)?;
            let final_result = match &lvalue.kind {
                ExpressionKind::Variable(v) => TackySymbol::from(v),
                _ => panic!("must be lvalue")
            };
            instrs.push(Copy {
                src: rhs_tacky,
                dst: final_result.clone(),
            });
            Ok((Variable(final_result), instrs))
        },
        ExpressionKind::Increment { e, is_post } => {
            debug_assert!(e.kind.is_lvalue_expression());
            let (e_tacky, mut instrs) = emit_tacky_for_expression(ctx, e)?;
            let incr_tacky_instr = Binary {
                operator: TackyBinaryOperator::Add,
                src1: e_tacky.clone(),
                src2: Int32(1),
                dst: e_tacky.clone(),
            };
            // In case of post increment, we have to return the original value
            // and then increment whatever is left.
            if *is_post {
                let tmp_cur_val = ctx.next_temporary_identifier();
                instrs.push(Copy { dst: tmp_cur_val.clone(), src: e_tacky.clone() });
                instrs.push(incr_tacky_instr);
                Ok((Variable(tmp_cur_val), instrs))
            } else {
                instrs.push(incr_tacky_instr);
                Ok((e_tacky, instrs))
            }
        }
        ExpressionKind::Decrement { e, is_post } => {
            debug_assert!(e.kind.is_lvalue_expression());
            let (e_tacky, mut instrs) = emit_tacky_for_expression(ctx, e)?;
            let decr_tacky_instr = Binary {
                operator: TackyBinaryOperator::Subtract,
                src1: e_tacky.clone(),
                src2: Int32(1),
                dst: e_tacky.clone(),
            };
            if *is_post {
                let tmp_cur_val = ctx.next_temporary_identifier();
                instrs.push(Copy { dst: tmp_cur_val.clone(), src: e_tacky.clone() });
                instrs.push(decr_tacky_instr);
                Ok((Variable(tmp_cur_val), instrs))
            } else {
                instrs.push(decr_tacky_instr);
                Ok((e_tacky, instrs))
            }
        }
        ExpressionKind::Conditional { condition, then_expr, else_expr } => {
            let (cond_res, mut instructions) = emit_tacky_for_expression(ctx, condition)?;
            let (then_res, then_instrs) = emit_tacky_for_expression(ctx, then_expr)?;
            let (else_res, else_instrs) = emit_tacky_for_expression(ctx, else_expr)?;

            let tmp_cond_res = ctx.next_temporary_identifier();
            let tmp_final_res = ctx.next_temporary_identifier();

            let cond_else_lbl = ctx.next_temporary_label("cond_else");
            let cond_end_lbl = ctx.next_temporary_label("cond_end");

            instructions.push(Copy { src: cond_res, dst: tmp_cond_res.clone() });
            instructions.push(JumpIfZero {
                condition: Variable(tmp_cond_res.clone()),
                target: cond_else_lbl.clone(),
            });

            instructions.extend(then_instrs);
            instructions.push(Copy { src: then_res, dst: tmp_final_res.clone() });
            instructions.push(Jump { target: cond_end_lbl.clone() });

            instructions.push(Label(cond_else_lbl.clone()));
            instructions.extend(else_instrs);
            instructions.push(Copy { src: else_res, dst: tmp_final_res.clone() });

            instructions.push(Label(cond_end_lbl));
            Ok((Variable(tmp_final_res), instructions))
        }
        ExpressionKind::FunctionCall { func_name, actual_params } => {
            let mut actual_param_tacky = Vec::with_capacity(actual_params.len());
            let mut instructions = vec![];
            for ap in actual_params.iter() {
                let (av, instrs) = emit_tacky_for_expression(ctx, ap)?;
                instructions.extend(instrs);

                let apv = ctx.next_temporary_identifier();
                instructions.push(Copy { src: av, dst: apv.clone() });
                actual_param_tacky.push(Variable(apv));
            }
            let dst = ctx.next_temporary_identifier();
            let dst_val = Variable(dst);
            let func_call_instr = FunctionCall {
                func_name: TackySymbol::from(func_name),
                args: actual_param_tacky,
                dst: dst_val.clone(),
            };
            instructions.push(func_call_instr);
            Ok((dst_val, instructions))
        },
    }
}

#[cfg(test)]
mod test {
    use std::fs;
    use std::path::{Path, PathBuf};

    use insta::assert_snapshot;
    use rstest::rstest;

    use crate::common::Radix::Decimal;
    use crate::lexer::Lexer;
    use crate::parser::{BinaryOperator, Declaration, DeclarationKind, Expression, ExpressionKind, ForInit, Parser, Statement, StatementKind, Symbol, UnaryOperator, VariableDeclaration};
    use crate::parser::ExpressionKind::{Assignment, Binary, Decrement, Increment, IntConstant, Unary};
    use crate::parser::StatementKind::For;
    use crate::semantic_analysis::identifier_resolution::resolve_program;
    use crate::semantic_analysis::loop_labeling::loop_label_program_definition;
    use crate::tacky::*;
    use crate::tacky::tacky::{emit_tacky_for_declaration, emit_tacky_for_expression, emit_tacky_for_statement, TackyContext};
    use crate::tacky::TackyBinaryOperator::LessThan;
    use crate::tacky::TackyInstruction::*;
    use crate::tacky::TackyValue::*;

    #[test]
    fn test_emit_tacky_for_int_constant() {
        let expr = Expression {
            location: (0,0).into(),
            kind: IntConstant("10".to_string(), Decimal),
        };
        let mut ctx = TackyContext::new();
        let (tval, tinstrs) = emit_tacky_for_expression(&mut ctx, &expr)
            .expect("error while generating IR");
        assert_eq!(tval, Int32(10));
        assert_eq!(tinstrs, vec![]);
    }

    #[test]
    fn test_emit_tacky_for_unary_complement() {
        let expr = Expression {
            location: (0,0).into(),
            kind: Unary(UnaryOperator::Complement,
                Box::new(Expression {
                    location: (0,0).into(),
                    kind: IntConstant("10".to_string(), Decimal),
                })),
        };
        let mut ctx = TackyContext::new();
        let (tval, tinstrs) = emit_tacky_for_expression(&mut ctx, &expr).unwrap();
        assert_eq!(tval, Variable(TackySymbol("<t>.0".into())));
        assert_eq!(tinstrs, vec![
            TackyInstruction::Unary {
                operator: TackyUnaryOperator::Complement,
                src: Int32(10),
                dst: Variable(TackySymbol("<t>.0".into())),
            },
        ]);
    }

    #[test]
    fn test_emit_tacky_for_unary_negate() {
        let expr = Expression {
            location: (0,0).into(),
            kind: Unary(UnaryOperator::Negate,
                        Box::new(Expression {
                            location: (0,0).into(),
                            kind: IntConstant("10".to_string(), Decimal),
                        })),
        };
        let mut ctx = TackyContext::new();
        let (tval, tinstrs) = emit_tacky_for_expression(&mut ctx, &expr).unwrap();
        assert_eq!(tval, Variable(TackySymbol("<t>.0".into())));
        assert_eq!(tinstrs, vec![
            TackyInstruction::Unary {
                operator: TackyUnaryOperator::Negate,
                src: Int32(10),
                dst: Variable(TackySymbol("<t>.0".into())),
            },
        ]);
    }

    #[test]
    fn test_emit_tacky_for_unary_not() {
        let expr = Expression {
            location: (0,0).into(),
            kind: Unary(UnaryOperator::Not,
                        Box::new(Expression {
                            location: (0,0).into(),
                            kind: IntConstant("10".to_string(), Decimal),
                        })),
        };
        let mut ctx = TackyContext::new();
        let (tval, tinstrs) = emit_tacky_for_expression(&mut ctx, &expr).unwrap();
        assert_eq!(tval, Variable(TackySymbol("<t>.0".into())));
        assert_eq!(tinstrs, vec![
            TackyInstruction::Unary {
                operator: TackyUnaryOperator::Not,
                src: Int32(10),
                dst: Variable(TackySymbol("<t>.0".into())),
            },
        ]);
    }

    #[test]
    fn test_emit_tacky_post_increment() {
        let expr = Expression {
            location: (0,0).into(),
            kind: Increment {
                is_post: true,
                e: Box::new(Expression {
                    location: (0,0).into(),
                    kind: ExpressionKind::Variable(Symbol { name: "a".to_string(), location: (0,0).into(), original_name: None }),
                }),
            },
        };
        let mut ctx = TackyContext::new();
        let (tval, actual) = emit_tacky_for_expression(&mut ctx, &expr).unwrap();
        assert_eq!(tval, Variable(TackySymbol::from("<t>.0")));
        let expected = vec![
            Copy { src: Variable(TackySymbol::from("a")), dst: TackySymbol::from("<t>.0") },
            TackyInstruction::Binary {
                operator: TackyBinaryOperator::Add,
                src1: Variable(TackySymbol::from("a")),
                src2: Int32(1),
                dst: Variable(TackySymbol::from("a")),
            }
        ];
        assert_eq!(tval, Variable(TackySymbol::from("<t>.0")));
        assert_eq!(actual, expected, "expected:{:#?}\nactual:{:#?}\n", expected, actual);
    }

    #[test]
    fn test_emit_tacky_for_post_decrement() {
        let expr = Expression {
            location: (0,0).into(),
            kind: Decrement {
                is_post: true,
                e: Box::new(Expression {
                    location: (0,0).into(),
                    kind: ExpressionKind::Variable(Symbol { name: "a".to_string(), location: (0,0).into(), original_name: None }),
                }),
            },
        };
        let mut ctx = TackyContext::new();
        let (tval, actual) = emit_tacky_for_expression(&mut ctx, &expr).unwrap();
        assert_eq!(tval, Variable(TackySymbol::from("<t>.0")));
        let expected = vec![
            Copy { src: Variable(TackySymbol::from("a")), dst: TackySymbol::from("<t>.0") },
            TackyInstruction::Binary {
                operator: TackyBinaryOperator::Subtract,
                src1: Variable(TackySymbol::from("a")),
                src2: Int32(1),
                dst: Variable(TackySymbol::from("a")),
            }
        ];
        assert_eq!(tval, Variable(TackySymbol::from("<t>.0")));
        assert_eq!(actual, expected, "expected:{:#?}\nactual:{:#?}\n", expected, actual);
    }

    #[test]
    fn test_emit_tacky_for_pre_increment() {
        let expr = Expression {
            location: (0,0).into(),
            kind: Increment {
                is_post: false,
                e: Box::new(Expression {
                    location: (0,0).into(),
                    kind: ExpressionKind::Variable(Symbol { name: "a".to_string(), location: (0,0).into(), original_name: None }),
                }),
            },
        };
        let mut ctx = TackyContext::new();
        let (tval, actual) = emit_tacky_for_expression(&mut ctx, &expr).unwrap();
        assert_eq!(tval, Variable(TackySymbol::from("a")));
        let expected = vec![
            TackyInstruction::Binary {
                operator: TackyBinaryOperator::Add,
                src1: Variable(TackySymbol::from("a")),
                src2: Int32(1),
                dst: Variable(TackySymbol::from("a")),
            }
        ];
        assert_eq!(tval, Variable(TackySymbol::from("a")));
        assert_eq!(actual, expected, "expected:{:#?}\nactual:{:#?}\n", expected, actual);
    }

    #[test]
    fn test_emit_tacky_for_pre_decrement() {
        let expr = Expression {
            location: (0,0).into(),
            kind: Decrement {
                is_post: false,
                e: Box::new(Expression {
                    location: (0,0).into(),
                    kind: ExpressionKind::Variable(Symbol { name: "a".to_string(), location: (0,0).into(), original_name: None }),
                }),
            },
        };
        let mut ctx = TackyContext::new();
        let (tval, actual) = emit_tacky_for_expression(&mut ctx, &expr).unwrap();
        assert_eq!(tval, Variable(TackySymbol::from("a")));
        let expected = vec![
            TackyInstruction::Binary {
                operator: TackyBinaryOperator::Subtract,
                src1: Variable(TackySymbol::from("a")),
                src2: Int32(1),
                dst: Variable(TackySymbol::from("a")),
            }
        ];
        assert_eq!(tval, Variable(TackySymbol::from("a")));
        assert_eq!(actual, expected, "expected:{:#?}\nactual:{:#?}\n", expected, actual);
    }
    
    #[test]
    fn test_emit_tacky_for_logical_and_with_shortcircuiting() {
        let expr = Expression {
            location: (0, 0).into(),
            kind: Binary(BinaryOperator::And,
                         Box::new(Expression {
                             location: (0, 0).into(),
                             kind: IntConstant("0".to_string(), Decimal),
                         }),
                         Box::new(Expression {
                             location: (0, 0).into(),
                             kind: IntConstant("1".to_string(), Decimal),
                         })),
        };
        let mut ctx = TackyContext::new();
        let (tval, tinstrs) = emit_tacky_for_expression(&mut ctx, &expr).unwrap();
        assert_eq!(tval, Variable(TackySymbol("<t>.0".into())));
        assert_eq!(tinstrs, vec![
            Copy { src: Int32(0), dst: TackySymbol("<t>.1".to_string()) },
            JumpIfZero { condition: Variable(TackySymbol("<t>.1".to_string())), target: TackySymbol("_L.and_false.0".to_string()) },

            Copy { src: Int32(1), dst: TackySymbol("<t>.2".to_string()) },
            JumpIfZero { condition: Variable(TackySymbol("<t>.2".to_string())), target: TackySymbol("_L.and_false.0".to_string()) },

            // condition evaluated to true
            Copy { src: Int32(1), dst: TackySymbol("<t>.0".to_string()) },
            Jump { target: TackySymbol("_L.and_end.1".to_string()) },

            // condition evaluated to false
            Label(TackySymbol("_L.and_false.0".to_string())),
            Copy { src: Int32(0), dst: TackySymbol("<t>.0".to_string()) },

            Label(TackySymbol("_L.and_end.1".to_string())),
        ]);
    }
    
    #[test]
    fn test_emit_tacky_for_logical_or_with_shortcircuiting() {
        let expr = Expression {
            location: (0, 0).into(),
            kind: Binary(BinaryOperator::Or,
                         Box::new(Expression {
                             location: (0, 0).into(),
                             kind: IntConstant("0".to_string(), Decimal),
                         }),
                         Box::new(Expression {
                             location: (0, 0).into(),
                             kind: IntConstant("1".to_string(), Decimal),
                         })),
        };
        let mut ctx = TackyContext::new();
        let (tval, tinstrs) = emit_tacky_for_expression(&mut ctx, &expr).unwrap();
        assert_eq!(tval, Variable(TackySymbol("<t>.0".into())));
        assert_eq!(tinstrs, vec![
            Copy { src: Int32(0), dst: TackySymbol("<t>.1".to_string()) },
            JumpIfNotZero { condition: Variable(TackySymbol("<t>.1".to_string())), target: TackySymbol("_L.or_true.0".to_string()) },

            Copy { src: Int32(1), dst: TackySymbol("<t>.2".to_string()) },
            JumpIfNotZero { condition: Variable(TackySymbol("<t>.2".to_string())), target: TackySymbol("_L.or_true.0".to_string()) },

            // condition evaluated to false
            Copy { src: Int32(0), dst: TackySymbol("<t>.0".to_string()) },
            Jump { target: TackySymbol("_L.or_end.1".to_string()) },

            // condition evaluated to false
            Label(TackySymbol("_L.or_true.0".to_string())),
            Copy { src: Int32(1), dst: TackySymbol("<t>.0".to_string()) },

            Label(TackySymbol("_L.or_end.1".to_string())),
        ]);
    }

    #[test]
    fn test_emit_tacky_for_simple_assignment() {
        let expr = Expression {
            location: (0,0).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (0,0).into(),
                    kind: ExpressionKind::Variable(Symbol { name: "a".to_string(), location: (0,0).into(), original_name: None }),
                }),
                rvalue: Box::new(Expression {
                    location: (0,0).into(),
                    kind: IntConstant("10".to_string(), Decimal),
                }),
                op: None,
            },
        };
        let mut ctx = TackyContext::new();
        let (tval, tinstrs) = emit_tacky_for_expression(&mut ctx, &expr).unwrap();
        assert_eq!(tval, Variable(TackySymbol("a".into())));
        assert_eq!(tinstrs, vec![
            Copy { src: Int32(10), dst: TackySymbol::from("a") },
        ]);
    }

    #[test]
    fn test_emit_tacky_for_declaration_and_assignment() {
        let decl = Declaration {
            location: (0,0).into(),
            kind: DeclarationKind::VarDeclaration(VariableDeclaration {
                identifier: Symbol {
                    location: (0, 0).into(),
                    name: "a".to_string(),
                    original_name: None,
                },
                init_expression: Some(Expression {
                    location: (0,0).into(),
                    kind: IntConstant("10".to_string(), Decimal)
                }),
            }),
        };
        let mut ctx = TackyContext::new();
        let tinstrs = emit_tacky_for_declaration(&mut ctx, &decl).unwrap();
        assert_eq!(tinstrs, vec![
            Copy { src: Int32(10), dst: TackySymbol::from("a") },
        ]);
    }

    #[test]
    fn test_emit_tacky_forloop_init_declaration() {
        let forloop = Statement {
            location: (0,0).into(),
            labels: vec![],
            kind: For {
                init: ForInit::InitDecl(Box::new(VariableDeclaration {
                    identifier: Symbol { name: "i".to_string(), location: (0,0).into(), original_name: None },
                    init_expression: Some(Expression {
                        location: (0,0).into(),
                        kind: IntConstant("0".to_string(), Decimal),
                    }),
                })),
                condition: None,
                post: Some(Box::new(Expression {
                    location: (0,0).into(),
                    kind: Increment {
                        is_post: true,
                        e: Box::new(Expression {
                            location: (0,0).into(),
                            kind: ExpressionKind::Variable(Symbol { name: "i".to_string(), location: (0,0).into(), original_name: None }),
                        }),
                    },
                })),
                loop_body: Box::new(Statement {
                    location: (0,0).into(),
                    labels: vec![],
                    kind: StatementKind::Expression(Expression {
                        location: (0,0).into(),
                        kind: Increment {
                            is_post: false,
                            e: Box::new(Expression {
                                location: (0,0).into(),
                                kind: ExpressionKind::Variable(Symbol { name: "a".to_string(), location: (0,0).into(), original_name: None }),
                            }),
                        },
                    }),
                }),
                loop_label: Some("forloop".to_string()),
            },
        };
        let mut ctx = TackyContext::new();
        let tinstrs = emit_tacky_for_statement(&mut ctx, &forloop).unwrap();
        assert_eq!(tinstrs, vec![
            Copy { src: Int32(0), dst: TackySymbol::from("i") },
            Label(TackySymbol::from("forloop.start")),
            TackyInstruction::Binary {
                operator: TackyBinaryOperator::Add,
                src1: Variable(TackySymbol::from("a")),
                src2: Int32(1),
                dst: Variable(TackySymbol::from("a")),
            },
            Label(TackySymbol::from("forloop.continue")),
            Copy { src: Variable(TackySymbol::from("i")), dst: TackySymbol::from("<t>.0")},
            TackyInstruction::Binary {
                operator: TackyBinaryOperator::Add,
                src1: Variable(TackySymbol::from("i")),
                src2: Int32(1),
                dst: Variable(TackySymbol::from("i")),
            },
            Jump {target: TackySymbol::from("forloop.start")},
            Label(TackySymbol::from("forloop.break")),
        ], "{:#?}", tinstrs);
    }

    #[test]
    fn test_emit_tacky_for_while_loop() {
        let whileloop = Statement {
            location: (0,0).into(),
            labels: vec![],
            kind: StatementKind::While {
                pre_condition: Box::new(Expression {
                    location: (0,0).into(),
                    kind: Binary(
                        BinaryOperator::LessThan,
                        Box::new(Expression {
                            location: (0,0).into(),
                            kind: ExpressionKind::Variable(Symbol { name: "i".to_string(), location: (0,0).into(), original_name: None }),
                        }),
                        Box::new(Expression {
                            location: (0,0).into(),
                            kind: IntConstant("10".to_string(), Decimal),
                        }),
                    ),
                }),
                loop_body: Box::new(Statement {
                    location: (0,0).into(),
                    labels: vec![],
                    kind: StatementKind::Expression(Expression {
                        location: (0,0).into(),
                        kind: Increment {
                            is_post: true,
                            e: Box::new(Expression {
                                location: (0,0).into(),
                                kind: ExpressionKind::Variable(Symbol { name: "i".to_string(), location: (0,0).into(), original_name: None }),
                            }),
                        },
                    }),
                }),
                loop_label: Some("loop".to_string()),
            },
        };
        let mut ctx = TackyContext::new();
        let tinstrs = emit_tacky_for_statement(&mut ctx, &whileloop).unwrap();
        assert_eq!(tinstrs, vec![
            Label(TackySymbol::from("loop.continue")),
            TackyInstruction::Binary {
                operator: TackyBinaryOperator::LessThan,
                src1: Variable(TackySymbol::from("i")),
                src2: Int32(10),
                dst: Variable(TackySymbol::from("<t>.0")),
            },
            Copy { src: Variable(TackySymbol::from("<t>.0")), dst: TackySymbol::from("<t>.1") },
            JumpIfZero {
                condition: Variable(TackySymbol::from("<t>.1")),
                target: TackySymbol::from("loop.break"),
            },
            Copy { src: Variable(TackySymbol::from("i")), dst: TackySymbol::from("<t>.2") },
            TackyInstruction::Binary {
                operator: TackyBinaryOperator::Add,
                src1: Variable(TackySymbol::from("i")),
                src2: Int32(1),
                dst: Variable(TackySymbol::from("i")),
            },
            Jump {target: TackySymbol::from("loop.continue")},
            Label(TackySymbol::from("loop.break")),
        ], "{:#?}", tinstrs);
    }

    #[test]
    fn test_emit_tacky_for_dowhile_loop() {
        let dowhileloop = Statement {
            location: (0,0).into(),
            labels: vec![],
            kind: StatementKind::DoWhile {
                loop_body: Box::new(Statement {
                    location: (0,0).into(),
                    labels: vec![],
                    kind: StatementKind::Expression(Expression {
                        location: (0,0).into(),
                        kind: Increment {
                            is_post: true,
                            e: Box::new(Expression {
                                location: (0,0).into(),
                                kind: ExpressionKind::Variable(Symbol { name: "i".to_string(), location: (0,0).into(), original_name: None }),
                            }),
                        },
                    }),
                }),
                post_condition: Box::new(Expression {
                    location: (0,0).into(),
                    kind: Binary(
                        BinaryOperator::LessThan,
                        Box::new(Expression {
                            location: (0,0).into(),
                            kind: ExpressionKind::Variable(Symbol { name: "i".to_string(), location: (0,0).into(), original_name: None }),
                        }),
                        Box::new(Expression {
                            location: (0,0).into(),
                            kind: IntConstant("10".to_string(), Decimal),
                        }),
                    ),
                }),
                loop_label: Some("loop".to_string()),
            },
        };
        let mut ctx = TackyContext::new();
        let tinstrs = emit_tacky_for_statement(&mut ctx, &dowhileloop).unwrap();
        assert_eq!(tinstrs, vec![
            Label(TackySymbol::from("loop.start")),
            Copy { src: Variable(TackySymbol::from("i")), dst: TackySymbol::from("<t>.0") },
            TackyInstruction::Binary {
                operator: TackyBinaryOperator::Add,
                src1: Variable(TackySymbol::from("i")),
                src2: Int32(1),
                dst: Variable(TackySymbol::from("i")),
            },
            Label(TackySymbol::from("loop.continue")),
            TackyInstruction::Binary {
                operator: LessThan,
                src1: Variable(TackySymbol::from("i")),
                src2: Int32(10),
                dst: Variable(TackySymbol::from("<t>.1")),
            },
            Copy { src: Variable(TackySymbol::from("<t>.1")), dst: TackySymbol::from("<t>.2") },
            JumpIfNotZero {
                condition: Variable(TackySymbol::from("<t>.2")),
                target: TackySymbol::from("loop.start"),
            },
            Label(TackySymbol::from("loop.break")),
        ], "{:#?}", tinstrs);
    }

    #[rstest]
    #[case("multifunc/simple.c")]
    fn test_generation_for_multiple_functions(#[case] input_path: &str) {
        run_ir_generation_snapshot_test("multiple functions", input_path);
    }

    #[rstest]
    #[case("unary/complement.c")]
    #[case("unary/negation.c")]
    #[case("unary/not.c")]
    fn test_generation_for_unary_operators(#[case] input_path: &str) {
        run_ir_generation_snapshot_test("unary operators", input_path)
    }

    #[rstest]
    #[case("unary/increment.c")]
    #[case("unary/decrement.c")]
    #[case("unary/pre_increment.c")]
    #[case("unary/pre_decrement.c")]
    fn test_generation_for_increment_and_decrement_operators(#[case] input_path: &str) {
        run_ir_generation_snapshot_test("increment and decrement operators", input_path)
    }

    #[rstest]
    #[case("binary/arithmetic_add.c")]
    #[case("binary/arithmetic_add_leftassoc.c")]
    #[case("binary/arithmetic_precedence.c")]
    #[case("binary/arithmetic_precedence_override.c")]
    #[case("binary/bitwise_and.c")]
    #[case("binary/relational_eq.c")]
    fn test_generation_for_binary_operators(#[case] input_path: &str) {
        run_ir_generation_snapshot_test("binary operators", input_path)
    }

    #[rstest]
    #[case("logical/logical_and.c")]
    #[case("logical/logical_and_expr.c")]
    #[case("logical/logical_and_multi.c")]
    #[case("logical/logical_not.c")]
    #[case("logical/logical_not_expr.c")]
    #[case("logical/logical_not_double.c")]
    #[case("logical/logical_or.c")]
    #[case("logical/logical_or_expr.c")]
    #[case("logical/logical_or_multi.c")]
    #[case("logical/logical_and_mixed_assoc.c")]
    fn test_generation_for_logical_operators(#[case] input_path: &str) {
        run_ir_generation_snapshot_test("logical operators", input_path)
    }

    #[rstest]
    #[case("localvars/simple.c")]
    #[case("localvars/declaration_and_assign.c")]
    #[case("localvars/nested_scopes.c")]
    #[case("localvars/expression_with_var.c")]
    fn test_generation_with_local_variables(#[case] input_path: &str) {
        run_ir_generation_snapshot_test("local variables", input_path)
    }

    #[rstest]
    #[case("conditional/if.c")]
    #[case("conditional/if_else.c")]
    #[case("conditional/if_else_if.c")]
    #[case("conditional/dangling_if.c")]
    #[case("conditional/ternary.c")]
    fn test_generation_with_conditional(#[case] input_path: &str) {
        run_ir_generation_snapshot_test("if statement", input_path)
    }

    #[rstest]
    #[case("loops/simple_for.c")]
    #[case("loops/for_init_decl.c")]
    #[case("loops/for_init_decl_shadow.c")]
    #[case("loops/for_noinit.c")]
    #[case("loops/for_all_empty.c")]
    #[case("loops/for_with_break.c")]
    #[case("loops/for_with_continue.c")]
    #[case("loops/nested_for.c")]
    #[case("loops/nested_for_with_break.c")]
    #[case("loops/nested_for_with_continue.c")]
    fn test_generation_with_for_loops(#[case] input_path: &str) {
        run_ir_generation_snapshot_test("for loop", input_path)
    }

    #[rstest]
    #[case("loops/simple_while.c")]
    #[case("loops/while_with_break.c")]
    #[case("loops/while_with_continue.c")]
    fn test_generation_with_while_loops(#[case] input_path: &str) {
        run_ir_generation_snapshot_test("while loop", input_path)
    }

    #[rstest]
    #[case("loops/simple_dowhile.c")]
    #[case("loops/dowhile_with_break.c")]
    #[case("loops/dowhile_with_continue.c")]
    fn test_generation_with_do_while_loops(#[case] input_path: &str) {
        run_ir_generation_snapshot_test("do-while loop", input_path)
    }

    fn run_ir_generation_snapshot_test(suite_description: &str, src_file: &str) {
        let base_dir = file!();
        let src_path = Path::new(base_dir).parent().unwrap().join("input").join(src_file);
        let source = fs::read_to_string(src_path.clone());
        assert!(source.is_ok(), "failed to read {:?}", src_path);

        let src = source.unwrap();
        let lexer = Lexer::new(&src);
        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("parsing must be successful");
        let resolved_ast = resolve_program(ast).expect("identity resolution must be successful");
        let loop_labeled_ast = loop_label_program_definition(resolved_ast).expect("loop labeling must be successful");
        let tacky = emit(&loop_labeled_ast).expect("tacky IR generation must be successful");

        let (out_dir, snapshot_file) = output_path_parts(src_file);
        insta::with_settings!({
            sort_maps => true,
            prepend_module_to_snapshot => false,
            description => suite_description,
            snapshot_path => out_dir,
            info => &format!("{}", src_file),
        }, {
            assert_snapshot!(snapshot_file, tacky);
        });
    }

    fn output_path_parts(src_file: &str) -> (PathBuf, String) {
        let input_path = Path::new(src_file);
        let parent = input_path.parent().unwrap_or_else(|| Path::new(""));
        let stem = input_path.file_stem().expect("No file stem").to_string_lossy();
        let output_dir = Path::new("output").join(parent);
        let output_file = format!("{}.tacky", stem);
        (output_dir, output_file)
    }
}