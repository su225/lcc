use std::fmt::{Display, Formatter};
use std::num::ParseIntError;

use derive_more::Display;
use thiserror::Error;

use crate::parser::{BinaryOperator, BlockItem, Declaration, DeclarationKind, Expression, ExpressionKind, FunctionDefinition, ProgramDefinition, Statement, StatementKind, UnaryOperator};
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

#[derive(Debug, PartialEq)]
pub struct TackyFunction {
    pub identifier: TackySymbol,
    pub body: Vec<TackyInstruction>,
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
    for blk_item in f.body.items.iter() {
        let instrs = emit_tacky_for_block_item(ctx, blk_item)?;
        instructions.extend(instrs);
    }
    Ok(TackyFunction {
        identifier: TackySymbol(f.name.name.into()),
        body: instructions,
    })
}

fn emit_tacky_for_block_item(ctx: &mut TackyContext, blk_item: &BlockItem) -> Result<Vec<TackyInstruction>, TackyError> {
    match blk_item {
        BlockItem::Statement(stmt) => emit_tacky_for_statement(ctx, stmt),
        BlockItem::Declaration(decl) => emit_tacky_for_declaration(ctx, decl),
        BlockItem::SubBlock(sub_block) => {
            let mut sub_block_instrs = vec![];
            for sub_block_item in sub_block.items.iter() {
                let instrs = emit_tacky_for_block_item(ctx, sub_block_item)?;
                sub_block_instrs.extend(instrs);
            }
            Ok(sub_block_instrs)
        }
    }
}

fn emit_tacky_for_declaration(ctx: &mut TackyContext, decl: &Declaration) -> Result<Vec<TackyInstruction>, TackyError> {
    match &decl.kind {
        DeclarationKind::Declaration { identifier, init_expression: Some(expr) } => {
            let (tacky_val, mut expr_tacky) = emit_tacky_for_expression(ctx, expr)?;
            expr_tacky.push(Copy { src: tacky_val, dst: identifier.name.into() });
            Ok(expr_tacky)
        },
        DeclarationKind::Declaration { init_expression: None, .. } => Ok(vec![]),
    }
}

fn emit_tacky_for_statement(ctx: &mut TackyContext, s: &Statement) -> Result<Vec<TackyInstruction>, TackyError> {
    match s.kind {
        StatementKind::Return(ref expr) => {
            let (dst, mut expr_instrs) = emit_tacky_for_expression(ctx, expr)?;
            expr_instrs.push(Return(dst));
            Ok(expr_instrs)
        }
        StatementKind::Expression(_) => todo!(),
        StatementKind::Null => Ok(vec![]),
    }
}

fn emit_tacky_for_expression(ctx: &mut TackyContext, e: &Expression) -> Result<(TackyValue, Vec<TackyInstruction>), TackyError> {
    match &e.kind {
        ExpressionKind::IntConstant(c, radix) => {
            let n = i32::from_str_radix(c, radix.value())?;
            Ok((Int32(n), vec![]))
        }
        ExpressionKind::Variable(_) => todo!(),
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
            let result = Variable(dst_tacky_identifier);
            tacky_instrs.extend(src2_tacky_instrs);
            tacky_instrs.push(Binary {
                operator: TackyBinaryOperator::from(binary_op),
                src1: src1_tacky,
                src2: src2_tacky,
                dst: dst_tacky,
            });
            Ok((result, tacky_instrs))
        },
        ExpressionKind::Assignment {..} => todo!(),
    }
}

#[cfg(test)]
mod test {
    use crate::common::Radix;
    use crate::parser::{BinaryOperator, Expression, UnaryOperator};
    use crate::parser::ExpressionKind::{Binary, IntConstant, Unary};
    use crate::tacky::*;
    use crate::tacky::tacky::{emit_tacky_for_expression, TackyContext};
    use crate::tacky::TackyInstruction::*;
    use crate::tacky::TackyValue::*;

    #[test]
    fn test_emit_tacky_for_int_constant() {
        let expr = Expression {
            location: (0,0).into(),
            kind: IntConstant("10", Radix::Decimal),
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
                    kind: IntConstant("10", Radix::Decimal),
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
                            kind: IntConstant("10", Radix::Decimal),
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
                            kind: IntConstant("10", Radix::Decimal),
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
    fn test_emit_tacky_for_logical_and_with_shortcircuiting() {
        let expr = Expression {
            location: (0, 0).into(),
            kind: Binary(BinaryOperator::And,
                         Box::new(Expression {
                             location: (0, 0).into(),
                             kind: IntConstant("0", Radix::Decimal),
                         }),
                         Box::new(Expression {
                             location: (0, 0).into(),
                             kind: IntConstant("1", Radix::Decimal),
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
                             kind: IntConstant("0", Radix::Decimal),
                         }),
                         Box::new(Expression {
                             location: (0, 0).into(),
                             kind: IntConstant("1", Radix::Decimal),
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
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use insta::assert_snapshot;
    use rstest::rstest;

    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::tacky::emit;

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

    fn run_ir_generation_snapshot_test(suite_description: &str, src_file: &str) {
        let base_dir = file!();
        let src_path = Path::new(base_dir).parent().unwrap().join("input").join(src_file);
        let source = fs::read_to_string(src_path.clone());
        assert!(source.is_ok(), "failed to read {:?}", src_path);

        let src = source.unwrap();
        let lexer = Lexer::new(&src);
        let mut parser = Parser::new(lexer);
        let ast = parser.parse().unwrap();
        let tacky = emit(&ast).unwrap();

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