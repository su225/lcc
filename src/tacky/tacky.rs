use std::fmt::{Display, Formatter};
use std::num::ParseIntError;
use derive_more::Display;
use thiserror::Error;
use crate::parser::types::*;
use crate::tacky::TackyInstruction::*;
use crate::tacky::TackyValue::*;

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

#[derive(Debug, PartialEq)]
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
    Copy { src: TackyValue, dst: TackyValue },
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
                f.write_fmt(format_args!("    {src} = {dst};"))
            },
            Jump { target } => {
                f.write_fmt(format_args!("    jump {target};"))
            },
            JumpIfZero { condition, target } => {
                f.write_fmt(format_args!("    jump_if_zero {condition} {target};"))
            }
            JumpIfNotZero { condition, target } => {
                f.write_fmt(format_args!("    jump_if_not_zero {condition} {target};"))
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
            Ok((Int32(n), vec![]))
        }
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
        }
    }
}

#[cfg(test)]
mod test {
    use ExpressionKind::*;
    use crate::common::Radix;
    use crate::parser::types::{Expression, ExpressionKind, UnaryOperator};
    use crate::tacky::tacky::{emit_tacky_for_expression, TackyContext};
    use crate::tacky::*;
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
}

#[cfg(test)]
mod tacky_ir_generation_snapshot_tests {
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