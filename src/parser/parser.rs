//! Parser for the language: This implements parser for a subset of
//! features in C programming language. A simple Recursive Descent
//! Parsing is used. It is handwritten.

use std::iter::Peekable;

use derive_more::with_trait::Add;
use thiserror::Error;
use serde::Serialize;
use crate::common::{Location, Radix};
use crate::lexer::{KeywordIdentifier, Lexer, LexerError, Token, TokenTag, TokenType};
use crate::parser::ParserError::{ExpectedBinaryOperator, UnexpectedEnd};

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Symbol<'a> {
    pub name: &'a str,
    location: Location,
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
pub struct BinaryOperatorPrecedence(u16);

impl BinaryOperator {
    #[inline]
    fn associativity(&self) -> BinaryOperatorAssociativity {
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
    fn precedence(&self) -> BinaryOperatorPrecedence {
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
    location: Location,
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

pub struct Parser<'a> {
    token_provider: Peekable<Lexer<'a>>,
}

#[derive(Error, Debug, PartialEq)]
pub enum ParserError {
    #[error("error from lexer: {0}")]
    TokenizationError(LexerError),

    #[error("keyword {kwd:?} used as identifier")]
    KeywordUsedAsIdentifier { location: Location, kwd: KeywordIdentifier },

    #[error("{location:?}: unexpected token")]
    UnexpectedToken { location: Location, expected_token_tags: Vec<TokenTag> },

    #[error("unexpected end of file. Expected {:?}", .0)]
    UnexpectedEnd(Vec<TokenTag>),

    #[error("{location:?}: expected unary operator, but found {actual_token:?}")]
    ExpectedUnaryOperator {
        location: Location,
        actual_token: TokenTag,
    },

    #[error("{location:?}: expected binary operator, but found {actual_token:?}")]
    ExpectedBinaryOperator {
        location: Location,
        actual_token: TokenTag,
    },

    #[error("{location:?}: expected keyword {keyword_identifier:?}")]
    ExpectedKeyword {
        location: Location,
        keyword_identifier: KeywordIdentifier,
        actual_token: TokenTag,
    },
}

impl<'a> Parser<'a> {
    pub fn new(token_provider: Lexer<'a>) -> Parser<'a> {
        Parser { token_provider: token_provider.peekable() }
    }

    /// parse parses the given source file and returns the
    /// Abstract Syntax Tree (AST).
    pub fn parse(&mut self) -> Result<ProgramDefinition<'a>, ParserError> {
        Ok(self.parse_program()?)
    }

    fn parse_program(&mut self) -> Result<ProgramDefinition<'a>, ParserError> {
        let mut functions: Vec<FunctionDefinition<'a>> = vec![];
        loop {
            if self.token_provider.peek().is_none() {
                break; // No more tokens and hence we are safe to stop here
            }
            // Otherwise parse the next function definition
            let func_definition = self.parse_function_definition()?;
            functions.push(func_definition);
        }

        Ok(ProgramDefinition { functions })
    }

    fn parse_function_definition(&mut self) -> Result<FunctionDefinition<'a>, ParserError> {
        let return_type = self.parse_type_expression()?;
        let name = self.parse_identifier()?;
        self.expect_open_parentheses()?;
        self.parse_function_parameters()?;
        self.expect_close_parentheses()?;
        let body = self.parse_function_body()?;
        Ok(FunctionDefinition { location: return_type.location, name, body })
    }

    fn parse_function_parameters(&mut self) -> Result<(), ParserError> {
        self.expect_keyword(KeywordIdentifier::TypeVoid)?;
        Ok(())
    }

    fn parse_function_body(&mut self) -> Result<Vec<Statement<'a>>, ParserError> {
        self.expect_open_braces()?;
        let stmt = self.parse_statement()?;
        self.expect_close_braces()?;
        Ok(vec![stmt])
    }

    fn expect_open_parentheses(&mut self) -> Result<(), ParserError> {
        self.expect_token_with_tag(TokenTag::OpenParentheses)
    }

    fn expect_close_parentheses(&mut self) -> Result<(), ParserError> {
        self.expect_token_with_tag(TokenTag::CloseParentheses)
    }

    fn expect_open_braces(&mut self) -> Result<(), ParserError> {
        self.expect_token_with_tag(TokenTag::OpenBrace)
    }

    fn expect_close_braces(&mut self) -> Result<(), ParserError> {
        self.expect_token_with_tag(TokenTag::CloseBrace)
    }

    fn expect_semicolon(&mut self) -> Result<(), ParserError> {
        self.expect_token_with_tag(TokenTag::Semicolon)
    }

    fn expect_keyword(&mut self, expected_kwid: KeywordIdentifier) -> Result<Location, ParserError> {
        match self.get_token_with_tag(TokenTag::Keyword)? {
            Token { location, token_type: TokenType::Keyword(kwid) } => {
                if kwid == expected_kwid {
                    Ok(location)
                } else {
                    Err(ParserError::ExpectedKeyword { location, keyword_identifier: expected_kwid, actual_token: TokenTag::Keyword })
                }
            }
            Token { location, token_type } => Err(ParserError::ExpectedKeyword {
                location,
                keyword_identifier: expected_kwid,
                actual_token: token_type.tag(),
            })
        }
    }

    fn parse_identifier(&mut self) -> Result<Symbol<'a>, ParserError> {
        match self.token_provider.next() {
            Some(Ok(Token { location, token_type })) => {
                match token_type {
                    TokenType::Identifier(name) => Ok(Symbol { name, location }),
                    TokenType::Keyword(kwd) => Err(ParserError::KeywordUsedAsIdentifier { location, kwd }),
                    _ => Err(ParserError::UnexpectedToken { location, expected_token_tags: vec![TokenTag::Identifier] })
                }
            }
            Some(Err(e)) => Err(ParserError::TokenizationError(e)),
            None => Err(ParserError::UnexpectedEnd(vec![TokenTag::Identifier])),
        }
    }

    fn parse_type_expression(&mut self) -> Result<TypeExpression, ParserError> {
        let kw_loc = self.expect_keyword(KeywordIdentifier::TypeInt)?;
        Ok(TypeExpression {
            location: kw_loc,
            kind: TypeExpressionKind::Primitive(PrimitiveKind::Integer),
        })
    }

    fn parse_statement(&mut self) -> Result<Statement<'a>, ParserError> {
        self.parse_return_statement()
    }

    fn parse_return_statement(&mut self) -> Result<Statement<'a>, ParserError> {
        let kloc = self.expect_keyword(KeywordIdentifier::Return)?;
        let return_code_expr = self.parse_expression()?;
        self.expect_semicolon()?;
        Ok(Statement { location: kloc, kind: StatementKind::Return(return_code_expr) })
    }

    fn parse_expression(&mut self) -> Result<Expression<'a>, ParserError> {
        let tok = self.token_provider.peek();
        match &tok {
            Some(Ok(_)) => self.parse_expression_with_precedence(BinaryOperatorPrecedence(0)),
            Some(Err(e)) => Err(ParserError::TokenizationError(e.clone())),
            None => Err(UnexpectedEnd(vec![TokenTag::IntConstant, TokenTag::OpenParentheses])),
        }
    }

    fn parse_expression_with_precedence(&mut self, min_precedence: BinaryOperatorPrecedence) -> Result<Expression<'a>, ParserError> {
        let mut result = self.parse_factor()?;
        while let Some(next_token) = self.token_provider.peek() {
            match &next_token {
                Ok(token) if token.token_type.is_binary_operator() => {
                    let binary_op = self.peek_binary_operator_token()?;
                    let binary_op_precedence = binary_op.precedence();
                    let binary_op_associativity = binary_op.associativity();
                    if binary_op_precedence < min_precedence {
                        break;
                    }
                    // Only if we pass the token precedence test, we can advance
                    // the pointer further to parse the next expression
                    self.token_provider.next();

                    let next_min_precedence = match binary_op_associativity {
                        BinaryOperatorAssociativity::Left => binary_op_precedence + BinaryOperatorPrecedence(1),
                        BinaryOperatorAssociativity::Right => binary_op_precedence,
                    };
                    let rhs = self.parse_expression_with_precedence(next_min_precedence)?;
                    result = Expression {
                        location: result.location,
                        kind: ExpressionKind::Binary(binary_op, Box::new(result), Box::new(rhs)),
                    }
                },
                Ok(_) => {
                    // It is not an error to see something else.
                    // Think of something like "10 + 20;" Here semicolon
                    // is a token which is not a binary operator. In this
                    // case, we should not treat it as an error.
                    break;
                }
                Err(e) => {
                    return Err(ParserError::TokenizationError(e.clone()));
                }
            };
        }
        Ok(result)
    }

    fn parse_factor(&mut self) -> Result<Expression<'a>, ParserError> {
        let next_token = self.token_provider.peek();
        match &next_token {
            Some(Ok(Token { token_type, location })) => {
                match token_type {
                    TokenType::IntConstant(_, _) => self.parse_int_constant_expression(),
                    op if op.is_unary_operator() => {
                        let tok_location = location.clone();
                        let unary_op = self.parse_unary_operator_token()?;
                        let factor = self.parse_factor()?;
                        Ok(Expression {
                            location: tok_location,
                            kind: ExpressionKind::Unary(unary_op, Box::new(factor)),
                        })
                    }
                    TokenType::OpenParentheses => {
                        self.expect_token_with_tag(TokenTag::OpenParentheses)?;
                        let expr = self.parse_expression()?;
                        self.expect_token_with_tag(TokenTag::CloseParentheses)?;
                        Ok(expr)
                    }
                    _ => Err(ParserError::UnexpectedToken {
                        location: location.clone(),
                        expected_token_tags: vec![
                            TokenTag::IntConstant,
                            TokenTag::OperatorUnaryComplement,
                            TokenTag::OperatorUnaryComplement,
                            TokenTag::OpenParentheses,
                        ],
                    })
                }
            },
            Some(Err(e)) => Err(ParserError::TokenizationError(e.clone())),
            None => Err(UnexpectedEnd(vec![TokenTag::IntConstant])),
        }
    }

    fn parse_unary_operator_token(&mut self) -> Result<UnaryOperator, ParserError> {
        let op_tok = self.token_provider.next();
        match op_tok {
            None => Err(UnexpectedEnd(vec![TokenTag::OperatorUnaryDecrement, TokenTag::OperatorUnaryComplement])),
            Some(Ok(Token { token_type, location })) => {
                match token_type {
                    TokenType::OperatorUnaryComplement => Ok(UnaryOperator::Complement),
                    TokenType::OperatorMinus => Ok(UnaryOperator::Negate),
                    TokenType::OperatorUnaryLogicalNot => Ok(UnaryOperator::Not),
                    tok_type => Err(ParserError::ExpectedUnaryOperator { location, actual_token: tok_type.tag() })
                }
            },
            Some(Err(e)) => Err(ParserError::TokenizationError(e)),
        }
    }

    fn peek_binary_operator_token(&mut self) -> Result<BinaryOperator, ParserError> {
        let op_tok = self.token_provider.peek();
        match &op_tok {
            None => Err(UnexpectedEnd(vec![TokenTag::OperatorPlus])),
            Some(Err(e)) => Err(ParserError::TokenizationError(e.clone())),
            Some(Ok(Token { token_type, location })) => {
                match token_type {
                    TokenType::OperatorPlus => Ok(BinaryOperator::Add),
                    TokenType::OperatorMinus => Ok(BinaryOperator::Subtract),
                    TokenType::OperatorAsterisk => Ok(BinaryOperator::Multiply),
                    TokenType::OperatorDiv => Ok(BinaryOperator::Divide),
                    TokenType::OperatorModulo => Ok(BinaryOperator::Modulo),
                    TokenType::OperatorLeftShift => Ok(BinaryOperator::LeftShift),
                    TokenType::OperatorRightShift => Ok(BinaryOperator::RightShift),
                    TokenType::OperatorBitwiseOr => Ok(BinaryOperator::BitwiseOr),
                    TokenType::OperatorBitwiseXor => Ok(BinaryOperator::BitwiseXor),
                    TokenType::OperatorBitwiseAnd => Ok(BinaryOperator::BitwiseAnd),
                    TokenType::OperatorLogicalOr => Ok(BinaryOperator::Or),
                    TokenType::OperatorLogicalAnd => Ok(BinaryOperator::And),
                    TokenType::OperatorRelationalEqual => Ok(BinaryOperator::Equal),
                    TokenType::OperatorRelationalNotEqual => Ok(BinaryOperator::NotEqual),
                    TokenType::OperatorRelationalGreaterThan => Ok(BinaryOperator::GreaterThan),
                    TokenType::OperatorRelationalGreaterThanEqualTo => Ok(BinaryOperator::GreaterThanOrEqual),
                    TokenType::OperatorRelationalLessThan => Ok(BinaryOperator::LessThan),
                    TokenType::OperatorRelationalLessThanEqualTo => Ok(BinaryOperator::LessThanOrEqual),
                    tok_type => Err(ExpectedBinaryOperator { location: location.clone(), actual_token: tok_type.tag() })
                }
            }
        }
    }

    fn parse_int_constant_expression(&mut self) -> Result<Expression<'a>, ParserError> {
        let tok = self.get_token_with_tag(TokenTag::IntConstant)?;
        let tok_loc = tok.location;
        match tok.token_type {
            TokenType::IntConstant(c, rad) => Ok(Expression {
                location: tok_loc,
                kind: ExpressionKind::IntConstant(c, rad),
            }),
            _ => panic!("should not reach here"),
        }
    }

    fn expect_token_with_tag(&mut self, expected_token_tag: TokenTag) -> Result<(), ParserError> {
        self.get_token_with_tag(expected_token_tag)?;
        Ok(())
    }

    fn get_token_with_tag(&mut self, expected_token_tag: TokenTag) -> Result<Token<'a>, ParserError> {
        let token = self.token_provider.next();
        match token {
            Some(Ok(token)) => {
                let token_tag = token.token_type.tag();
                if token_tag == expected_token_tag {
                    Ok(token)
                } else {
                    Err(ParserError::UnexpectedToken { location: token.location, expected_token_tags: vec![expected_token_tag] })
                }
            }
            Some(Err(e)) => Err(ParserError::TokenizationError(e)),
            None => Err(UnexpectedEnd(vec![expected_token_tag])),
        }
    }
}

#[cfg(test)]
mod test {
    use indoc::indoc;
    use insta::{assert_yaml_snapshot, with_settings};
    use rstest::rstest;
    use crate::common::{Location, Radix};
    use crate::common::Radix::Decimal;
    use crate::lexer::Lexer;
    use crate::parser::{BinaryOperator, Expression, FunctionDefinition, Parser, ParserError, ProgramDefinition, Statement, Symbol, UnaryOperator};
    use crate::parser::ExpressionKind::{Binary, IntConstant, Unary};
    use crate::parser::StatementKind::Return;

    #[test]
    fn test_parse_program_with_tabs() {
        let src = "int	main	(	void)	{	return	0	;	}";
        let lexer = Lexer::new(src);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert_eq!(Ok(ProgramDefinition {
            functions: vec![
                FunctionDefinition {
                    location: Location {line: 1, column: 1},
                    name: Symbol {
                        name: "main",
                        location: Location {line: 1, column: 8},
                    },
                    body: vec![
                        Statement {
                            location: Location { line: 1, column: 40 },
                            kind: Return(Expression {
                                location: Location { line: 1, column: 48 },
                                kind: IntConstant("0", Decimal),
                            }),
                        },
                    ],
                },
            ],
        }), parsed);
    }

    #[test]
    fn test_parse_multiple_functions() {
        let src = r#"
            int main(void) {
                return 2;
            }

            int foo(void) {
                return 3;
            }
        "#;
        let lexer = Lexer::new(src);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert_eq!(Ok(ProgramDefinition {
            functions: vec![
                FunctionDefinition {
                    location: Location { line: 2, column: 13 },
                    name: Symbol { name: "main", location: Location { line: 2, column: 17 } },
                    body: vec![
                        Statement {
                            location: Location { line: 3, column: 17 },
                            kind: Return(Expression {
                                location: Location { line: 3, column: 24 },
                                kind: IntConstant("2", Decimal),
                            })
                        }
                    ],
                },
                FunctionDefinition {
                    location: Location { line: 6, column: 13 },
                    name: Symbol { name: "foo", location: Location { line: 6, column: 17 } },
                    body: vec![
                        Statement {
                            location: Location { line: 7, column: 17 },
                            kind: Return(Expression {
                                location: Location { line: 7, column: 24 },
                                kind: IntConstant("3", Decimal),
                            }),
                        }
                    ],
                }
            ],
        }), parsed)
    }

    #[test]
    fn test_parse_return_with_binary_operator_expression() {
        let src = indoc! {r#"
        int main(void) {
            return 1 + 2;
        }
        "#};
        let lexer = Lexer::new(src);
        let mut parser = Parser::new(lexer);
        let actual = parser.parse();
        let expected = Ok(ProgramDefinition {
            functions: vec![
                FunctionDefinition {
                    location: (1, 1).into(),
                    name: Symbol { name: "main", location: (1, 5).into() },
                    body: vec![
                        Statement {
                            location: (2, 5).into(),
                            kind: Return(Expression {
                                location: (2, 12).into(),
                                kind: Binary(
                                    BinaryOperator::Add,
                                    Box::new(Expression { location: (2, 12).into(), kind: IntConstant("1", Decimal) }),
                                    Box::new(Expression { location: (2, 16).into(), kind: IntConstant("2", Decimal) }),
                                ),
                            }),
                        },
                    ],
                },
            ],
        });
        assert_eq!(expected, actual, "expected:\n{:#?}\nactual:\n{:#?}\n", expected, actual);
    }

    struct ExprTestCase<'a> {
        src: &'a str,
        expected: Result<Expression<'a>, ParserError>,
    }

    #[test]
    fn test_parse_expression_constant_base_10_integer() {
        let src = "100";
        let expected = Ok(Expression {
            location: Location { line: 1, column: 1 },
            kind: IntConstant("100", Decimal),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected })
    }

    #[test]
    fn test_parse_expression_complement_operator() {
        let src = "~0xdeadbeef";
        let expected = Ok(Expression {
            location: Location { line: 1, column: 1 },
            kind: Unary(UnaryOperator::Complement, Box::new(Expression {
                location: Location { line: 1, column: 2 },
                kind: IntConstant("0xdeadbeef", Radix::Hexadecimal),
            })),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected })
    }

    #[test]
    fn test_parse_expression_negation_operator() {
        let src = "-100";
        let expected = Ok(Expression {
            location: Location { line: 1, column: 1 },
            kind: Unary(UnaryOperator::Negate, Box::new(Expression {
                location: Location { line: 1, column: 2 },
                kind: IntConstant("100", Decimal),
            })),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected })
    }

    #[test]
    fn test_parse_expression_redundant_parentheses_around_int_constant() {
        let src = "(100)";
        let expected = Ok(Expression {
            location: Location { line: 1, column: 2 },
            kind: IntConstant("100", Decimal),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected })
    }

    #[test]
    fn test_parse_expression_double_complement() {
        let src = "~~100";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Unary(UnaryOperator::Complement, Box::new(Expression {
                location: Location { line: 1, column: 2 },
                kind: Unary(UnaryOperator::Complement, Box::new(Expression {
                    location: Location { line: 1, column: 3 },
                    kind: IntConstant("100", Decimal),
                }))
            })),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected })
    }

    #[test]
    fn test_parse_expression_double_negation() {
        let src = "-(-100)";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Unary(
                UnaryOperator::Negate,
                Box::new(Expression {
                    location: (1, 3).into(),
                    kind: Unary(
                        UnaryOperator::Negate,
                        Box::new(Expression {
                            location: (1, 4).into(),
                            kind: IntConstant("100", Decimal),
                        }),
                    ),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_simple_addition() {
        let src = "10 + 20";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Add,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: IntConstant("10", Decimal),
                }),
                Box::new(Expression {
                    location: (1, 6).into(),
                    kind: IntConstant("20", Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_simple_subtraction() {
        let src = "30 - 15";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Subtract,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: IntConstant("30", Decimal),
                }),
                Box::new(Expression {
                    location: (1, 6).into(),
                    kind: IntConstant("15", Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_simple_multiplication() {
        let src = "4 * 5";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Multiply,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: IntConstant("4", Decimal),
                }),
                Box::new(Expression {
                    location: (1, 5).into(),
                    kind: IntConstant("5", Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_simple_division() {
        let src = "100 / 25";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Divide,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: IntConstant("100", Decimal),
                }),
                Box::new(Expression {
                    location: (1, 7).into(),
                    kind: IntConstant("25", Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_addition_left_associative() {
        let src = "1+2+3";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Add,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: Binary(
                        BinaryOperator::Add,
                        Box::new(Expression {
                            location: (1, 1).into(),
                            kind: IntConstant("1", Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 3).into(),
                            kind: IntConstant("2", Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 5).into(),
                    kind: IntConstant("3", Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_subtraction_left_associative() {
        let src = "5-3-1";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Subtract,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: Binary(
                        BinaryOperator::Subtract,
                        Box::new(Expression {
                            location: (1, 1).into(),
                            kind: IntConstant("5", Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 3).into(),
                            kind: IntConstant("3", Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 5).into(),
                    kind: IntConstant("1", Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_multiplication_left_associative() {
        let src = "2*3*4";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Multiply,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: Binary(
                        BinaryOperator::Multiply,
                        Box::new(Expression {
                            location: (1, 1).into(),
                            kind: IntConstant("2", Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 3).into(),
                            kind: IntConstant("3", Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 5).into(),
                    kind: IntConstant("4", Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_division_left_associative() {
        let src = "20/5/2";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Divide,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: Binary(
                        BinaryOperator::Divide,
                        Box::new(Expression {
                            location: (1, 1).into(),
                            kind: IntConstant("20", Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 4).into(),
                            kind: IntConstant("5", Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 6).into(),
                    kind: IntConstant("2", Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_modulo_left_associative() {
        let src = "10%4%2";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Modulo,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: Binary(
                        BinaryOperator::Modulo,
                        Box::new(Expression {
                            location: (1, 1).into(),
                            kind: IntConstant("10", Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 4).into(),
                            kind: IntConstant("4", Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 6).into(),
                    kind: IntConstant("2", Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_multiplication_higher_precedence_than_addition() {
        let src = "2+3*4";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Add,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: IntConstant("2", Decimal),
                }),
                Box::new(Expression {
                    location: (1, 3).into(),
                    kind: Binary(
                        BinaryOperator::Multiply,
                        Box::new(Expression {
                            location: (1, 3).into(),
                            kind: IntConstant("3", Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 5).into(),
                            kind: IntConstant("4", Decimal),
                        }),
                    ),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_division_higher_precedence_than_subtraction() {
        let src = "20-6/2";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Subtract,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: IntConstant("20", Decimal),
                }),
                Box::new(Expression {
                    location: (1, 4).into(),
                    kind: Binary(
                        BinaryOperator::Divide,
                        Box::new(Expression {
                            location: (1, 4).into(),
                            kind: IntConstant("6", Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 6).into(),
                            kind: IntConstant("2", Decimal),
                        }),
                    ),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_modulo_higher_precedence_than_addition() {
        let src = "9+8%5";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Binary(
                BinaryOperator::Add,
                Box::new(Expression {
                    location: (1, 1).into(),
                    kind: IntConstant("9", Decimal),
                }),
                Box::new(Expression {
                    location: (1, 3).into(),
                    kind: Binary(
                        BinaryOperator::Modulo,
                        Box::new(Expression {
                            location: (1, 3).into(),
                            kind: IntConstant("8", Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 5).into(),
                            kind: IntConstant("5", Decimal),
                        }),
                    ),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }


    #[test]
    fn test_parse_expression_parentheses_override_precedence() {
        let src = "(2+3)*4";
        let expected = Ok(Expression {
            location: (1, 2).into(),
            kind: Binary(
                BinaryOperator::Multiply,
                Box::new(Expression {
                    location: (1, 2).into(),
                    kind: Binary(
                        BinaryOperator::Add,
                        Box::new(Expression {
                            location: (1, 2).into(),
                            kind: IntConstant("2", Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 4).into(),
                            kind: IntConstant("3", Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 7).into(),
                    kind: IntConstant("4", Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_nested_parentheses() {
        let src = "(10-(2+3))*2";
        let expected = Ok(Expression {
            location: (1, 2).into(),
            kind: Binary(
                BinaryOperator::Multiply,
                Box::new(Expression {
                    location: (1, 2).into(),
                    kind: Binary(
                        BinaryOperator::Subtract,
                        Box::new(Expression {
                            location: (1, 2).into(),
                            kind: IntConstant("10", Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 6).into(),
                            kind: Binary(
                                BinaryOperator::Add,
                                Box::new(Expression {
                                    location: (1, 6).into(),
                                    kind: IntConstant("2", Decimal),
                                }),
                                Box::new(Expression {
                                    location: (1, 8).into(),
                                    kind: IntConstant("3", Decimal),
                                }),
                            ),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 12).into(),
                    kind: IntConstant("2", Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    fn run_parse_expression_test_case(test_case: ExprTestCase) {
        let lexer = Lexer::new(test_case.src);
        let mut parser = Parser::new(lexer);
        let actual = parser.parse_expression();
        assert_eq!(test_case.expected, actual);
    }
}

#[cfg(test)]
mod expression_test {
    use rstest::rstest;
    use ExpressionKind::{Binary, IntConstant, Unary};
    use crate::lexer::Lexer;
    use crate::parser::{Expression, ExpressionKind, Parser};

    #[rstest]
    #[case("simple_addition", "1+2")]
    #[case("simple_subtraction", "1-20")]
    #[case("simple_multiplication", "10*20")]
    #[case("simple_division", "2/4")]
    #[case("simple_remainder", "3%2")]
    #[case("multiplication_with_unary_operands", "~4*-3")]
    fn test_should_parse_arithmetic_expressions(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("arithmetic expressions", description, "expr/arithmetic", src);
    }

    #[rstest]
    #[case("addition_is_left_associative", "1+2+3")]
    #[case("subtraction_is_left_associative", "1-2-3")]
    #[case("multiplication_is_left_associative", "2*3*4")]
    #[case("division_is_left_associative", "10/2/3")]
    #[case("modulo_is_left_associative", "10 % 2 % 3")]
    fn test_should_parse_arithmetic_expressions_with_correct_associativity(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("arithmetic expressions with correct associativity", description, "expr/arithmetic", src);
    }

    #[rstest]
    #[case("multiplication_has_higher_precedence_than_addition", "4+2*3+8")]
    #[case("division_has_higher_precedence_than_addition", "10+4/2+3")]
    #[case("parentheses_override_precedence", "(2+4)*5")]
    #[case("multiple_nested_parentheses", "(10-(2+3))*2")]
    #[case("unary_negate_binary_operator_expression", "-(4+3)")]
    #[case("operation_with_complement_operator", "4+~3")]
    #[case("addition_with_negated_operand", "4+(-3)")]
    fn test_should_parse_arithmetic_expressions_with_correct_precedence(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("arithmetic expressions with correct precedence", description, "expr/arithmetic", src);
    }

    #[rstest]
    #[case("unary_complement", "~10")]
    #[case("unary_negation", "-10")]
    #[case("double_complement", "~~10")]
    #[case("logical_unary_not", "!20")]
    #[case("double_logical_unary_not", "!!10")]
    fn test_should_parse_unary_expressions(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("unary expressions", description, "expr/unary", src);
    }

    #[rstest]
    #[case("unary_complement", "~10")]
    #[case("unary_negation", "-10")]
    #[case("double_complement", "~~10")]
    #[case("logical_unary_not", "!20")]
    #[case("double_logical_unary_not", "!!10")]
    fn test_should_parse_bitwise_expressions(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("bitwise operator expressions", description, "expr/bitwise", src);
    }

    #[rstest]
    #[case("bitwise_and_is_left_associative", "10 & 20 & 30")]
    #[case("bitwise_or_is_left_associative", "10 | 20 | 30")]
    #[case("bitwise_xor_is_left_associative", "10 ^ 20 ^ 30")]
    #[case("left_shift_is_left_associative", "1<<2<<3")]
    #[case("right_shift_is_left_associative", "200>>1>>1")]
    fn test_should_parse_bitwise_expressions_with_correct_associativity(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("bitwise operator expressions with correct associativity", description, "expr/bitwise", src);
    }

    #[rstest]
    #[case("bitwise_or_and_xor", "1 | 2 ^ 3 & 4")]
    #[case("bitwise_xor_and", "1 ^ 2 & 3")]
    #[case("bitwise_and_shift", "1 & 2 << 3")]
    #[case("bitwise_shift_or", "1 << 2 | 3")]
    #[case("bitwise_or_and", "1 | 2 & 3")]
    #[case("bitwise_shift_left_right", "1 << 2 >> 3")]
    #[case("bitwise_or_with_parens", "1 | (2 ^ 3)")]
    #[case("bitwise_xor_and_parens", "(1 ^ 2) & 3")]
    #[case("bitwise_and_shift_parens", "1 & (2 << 3)")]
    #[case("bitwise_shift_parens", "(1 << 2) >> 3")]
    #[case("bitwise_or_xor_parens", "(1 | 2) ^ 3")]
    #[case("bitwise_xor_and_parens_rhs", "1 ^ (2 & 3)")]
    fn test_should_parse_bitwise_expressions_with_correct_precedence(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("bitwise operator expressions with correct precedence", description, "expr/bitwise", src);
    }

    #[rstest]
    #[case("logical_and", "10 && 20")]
    #[case("logical_or", "1 || 0")]
    #[case("logical_not", "!10")]
    fn test_should_parse_logical_expressions(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("logical expressions", description, "expr/logical", src);
    }

    #[rstest]
    #[case("logical_or_is_left_associative", "1 || 2 || 3")]
    #[case("logical_and_is_left_associative", "1 && 2 && 3")]
    fn test_should_parse_logical_expressions_with_correct_associativity(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("logical expressions with correct associativity", description, "expr/logical", src);
    }

    #[rstest]
    #[case("logical_mixed_or_and", "1 || 2 && 3")]
    #[case("logical_and_with_parens", "(1 && 2) && 3")]
    #[case("logical_or_with_parens", "(1 || 2) || 3")]
    #[case("logical_mixed_and_or_parens", "1 && (2 || 3)")]
    fn test_should_parse_logical_expressions_with_correct_precedence(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("logical expressions with correct precedence", description, "expr/logical", src)
    }

    #[rstest]
    #[case("greater_than", "10 > 5")]
    #[case("less_than", "3 < 4")]
    #[case("greater_equal", "7 >= 7")]
    #[case("less_equal", "2 <= 3")]
    #[case("equal", "5 == 5")]
    #[case("not_equal", "5 != 6")]
    fn test_should_parse_relational_expressions(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("relational expressions", description, "expr/relational", src);
    }

    #[rstest]
    #[case("assoc_less_chain", "1 < 2 < 3")]             // (1 < 2) < 3
    #[case("assoc_greater_chain", "5 > 4 > 3")]          // (5 > 4) > 3
    #[case("assoc_le_ge_chain", "3 <= 3 >= 2")]          // (3 <= 3) >= 2
    #[case("assoc_logical_and", "1 && 1 && 0")]          // (1 && 1) && 0
    #[case("assoc_logical_or", "0 || 1 || 1")]           // (0 || 1) || 1
    fn test_should_parse_relational_expressions_with_correct_associativity(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("relational expressions with correct associativity", description, "expr/relational", src);
    }

    #[rstest]
    #[case("precedence_cmp_and", "1 < 2 && 3 > 2")]      // (<, >) evaluated before &&
    #[case("precedence_cmp_or", "1 == 1 || 0 != 1")]     // (==, !=) before ||
    #[case("precedence_and_or", "1 && 0 || 1")]          // && before ||
    fn test_should_parse_relational_expressions_with_correct_precedence(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test("relational expressions with correct precedence", description, "expr/relational", src);
    }

    fn run_snapshot_test(suite_description: &str, description: &str, snapshot_path: &str, src: &str) {
        let lexer = Lexer::new(src);
        let mut parser = Parser::new(lexer);
        let actual = parser.parse_expression();
        assert!(actual.is_ok(), "src: {}\nactual:{:?}\n", src, actual);
        insta::with_settings!({
            sort_maps => true,
            prepend_module_to_snapshot => false,
            description => suite_description,
            snapshot_path => format!("snapshots/{snapshot_path}"),
            info => &src,
        }, {
            insta::assert_yaml_snapshot!(format!("{description}"), actual.unwrap());
        });
    }

    #[rstest]
    #[case("1+2+3", "(1+2)+3")]
    #[case("(1+2)+3", "((1+2)+3)")]
    #[case("1-2-3", "(1-2)-3")]
    #[case("1*2*3", "(1*2)*3")]
    #[case("10/2/3", "(10/2)/3")]
    #[case("1+2+3+4", "(((1+2)+3)+4)")]
    #[case("10 % 4 % 2", "(10 % 4) % 2")]
    #[case("2*3*4*5", "(((2*3)*4)*5)")]
    #[case("100-20-10-5", "(((100-20)-10)-5)")]
    #[case("1+2*3+4", "1+(2*3)+4")]
    #[case("100/10/2", "(100/10)/2")]
    #[case("10 % (4 % 2)", "10 % (4 % 2)")]
    #[case("1 + 2 * 3", "(1 + (2 * 3))")]
    #[case("2 * 3 + 4", "((2 * 3) + 4)")]
    #[case("10 - 2 * 3", "(10 - (2 * 3))")]
    #[case("2 * 3 - 4", "((2 * 3) - 4)")]
    #[case("12 / 4 + 2", "((12 / 4) + 2)")]
    #[case("2 + 12 / 4", "(2 + (12 / 4))")]
    #[case("10 % 3 + 1", "((10 % 3) + 1)")]
    #[case("10 - 3 % 2", "(10 - (3 % 2))")]
    #[case("1 + 5 % 3", "(1 + (5 % 3))")]
    #[case("10 % 4 * 2", "((10 % 4) * 2)")]
    #[case("1 + 2 + 3 * 4", "((1 + 2) + (3 * 4))")]
    #[case("1 * 2 + 3 * 4", "((1 * 2) + (3 * 4))")]
    #[case("8 / 2 + 3 * 2", "((8 / 2) + (3 * 2))")]
    #[case("6 + 4 / 2 * 3", "(6 + ((4 / 2) * 3))")]
    #[case("6 - 2 * 3 + 1", "((6 - (2 * 3)) + 1)")]
    #[case("20 % 3 * 2", "((20 % 3) * 2)")]
    #[case("4 + 6 / 3 * 2", "(4 + ((6 / 3) * 2))")]
    #[case("1 + 2 + 3 + 4 * 5", "(((1 + 2) + 3) + (4 * 5))")]
    #[case("20 - 5 * 2 + 10 / 2", "(((20 - (5 * 2)) + (10 / 2)))")]
    #[case("2 * 3 % 2", "((2 * 3) % 2)")]
    #[case("3 + 4 * 5 % 2", "(3 + ((4 * 5) % 2))")]
    #[case("100 / 5 * 2 + 1", "(((100 / 5) * 2) + 1)")]
    #[case("5 * 4 / 2 - 1", "(((5 * 4) / 2) - 1)")]
    #[case("1 + 2 * 3 - 4 / 2", "((1 + (2 * 3)) - (4 / 2))")]
    #[case("10 % 4 % 2", "((10 % 4) % 2)")]
    #[case("10 - 4 + 3 * 2", "((10 - 4) + (3 * 2))")]
    #[case("100 / 10 / 2", "((100 / 10) / 2)")]
    #[case("10 % 5 * 3 + 1", "(((10 % 5) * 3) + 1)")]
    #[case("2 + 3 * 4 % 5", "(2 + ((3 * 4) % 5))")]
    #[case("8 - 2 + 1 * 5", "((8 - 2) + (1 * 5))")]
    #[case("9 + 6 / 2 - 3", "((9 + (6 / 2)) - 3)")]
    #[case("10 - 2 * 3 + 4", "((10 - (2 * 3)) + 4)")]
    #[case("5 + 6 * 2 - 3", "((5 + (6 * 2)) - 3)")]
    #[case("6 + 8 / 4 + 1", "((6 + (8 / 4)) + 1)")]
    #[case("7 * 2 + 3 % 2", "((7 * 2) + (3 % 2))")]
    #[case("9 - 4 + 2 * 3", "((9 - 4) + (2 * 3))")]
    #[case("3 * 4 % 5 + 6", "(((3 * 4) % 5) + 6)")]
    #[case("100 / 5 + 2 * 3", "((100 / 5) + (2 * 3))")]
    #[case("1 + 2 * 3 / 4", "(1 + ((2 * 3) / 4))")]
    #[case("1 + (2 + 3) * 4", "(1 + ((2 + 3) * 4))")]
    #[case("(1 + 2) * (3 + 4)", "((1 + 2) * (3 + 4))")]
    #[case("((1 + 2) * 3) + 4", "(((1 + 2) * 3) + 4)")]
    #[case("1 + 2 + 3 + 4 * 5 / 2", "(((1 + 2) + 3) + ((4 * 5) / 2))")]
    #[case("-5 + 3 * 2", "((-5) + (3 * 2))")]
    #[case("~4 + 2 * 3", "((~4) + (2 * 3))")]
    #[case("-(4 + 2) * 3", "((-(4 + 2)) * 3)")]
    #[case("~(3 + 1) * 2", "((~(3 + 1)) * 2)")]
    #[case("4 + ~2 * 3", "(4 + ((~2) * 3))")]
    #[case("-(5 * 2) + 3", "((-(5 * 2)) + 3)")]
    #[case("3 + -4 * 2", "(3 + ((-4) * 2))")]
    #[case("10 / -2 + 1", "((10 / (-2)) + 1)")]
    #[case("10 % ~3 * 2", "(((10 % (~3)) * 2))")]
    #[case("-1 + -2 + -3 * -4", "(((-1) + (-2)) + ((-3) * (-4)))")]
    #[case("~1 + 2 % 3 - 4 * 5", "(((~1) + (2 % 3)) - (4 * 5))")]
    #[case("-(3 * 2 + 1)", "(-( (3 * 2) + 1 ))")]
    #[case("-(3 + 2) * (4 - 1)", "((-(3 + 2)) * (4 - 1))")]
    #[case("~(2 * 3 + 4)", "(~((2 * 3) + 4))")]
    #[case("5 + ~(3 * 2 - 1)", "(5 + (~((3 * 2) - 1)))")]
    #[case("(-1 + 2) * ~(3 + 4)", "((( -1 + 2 )) * (~(3 + 4)))")]
    #[case("10 - ~2 + -3", "((10 - (~2)) + (-3))")]
    #[case("3 + 4 * ~5 - 6", "((3 + (4 * (~5))) - 6)")]
    #[case("1 + ~2 + -3 + 4", "(((1 + (~2)) + (-3)) + 4)")]
    #[case("~10 / -2 + 3", "(((~10) / (-2)) + 3)")]
    #[case("~5 % 3 * -2", "(((~5) % 3) * (-2))")]
    #[case("-(~2 + 3) * 4", "((-(~2 + 3)) * 4)")]
    #[case("~((2 + 3) * 4)", "(~((2 + 3) * 4))")]
    #[case("-((2 + 3) * (4 % 2))", "(-((2 + 3) * (4 % 2)))")]
    #[case("~(2 * 3) + -(4 % 2)", "((~(2 * 3)) + (-(4 % 2)))")]
    #[case("5 + ~(3 + -2) * 4", "(5 + ((~(3 + (-2))) * 4))")]
    #[case("~1 + 2 * ~(3 + 4)", "((~1) + (2 * (~(3 + 4))))")]
    #[case("~(~3 + -2)", "(~((~3) + (-2)))")]
    #[case("-(~2 * 3) + 4", "((-(~2 * 3)) + 4)")]
    #[case("2 + 3 * 4 - 5 / -1 % 2", "(((2 + (3 * 4)) - ((5 / (-1)) % 2)))")]
    #[case("-10 + ~5 * 3 % 2 - 1", "(((-10) + ((~5 * 3) % 2)) - 1)")]
    #[case("~(~1 + 2 * 3) - 4", "((~((~1 + (2 * 3)))) - 4)")]
    fn test_arithmetic_operator_precedence_and_associativity(#[case] src1: &str, #[case] src2: &str) {
        run_expression_equivalence_test(src1, src2);
    }

    #[rstest]
    #[case("1 & 2 & 3", "(1 & 2) & 3")]
    #[case("1 & 2 & 3 & 4", "(((1 & 2) & 3) & 4)")]
    #[case("1 | 2 | 3", "(1 | 2) | 3")]
    #[case("1 | 2 | 3 | 4", "(((1 | 2) | 3) | 4)")]
    #[case("1 ^ 2 ^ 3", "(1 ^ 2) ^ 3")]
    #[case("1 ^ 2 ^ 3 ^ 4", "(((1 ^ 2) ^ 3) ^ 4)")]
    #[case("1 << 2 << 3", "(1 << 2) << 3")]
    #[case("1 << 2 << 3 << 4", "(((1 << 2) << 3) << 4)")]
    #[case("8 >> 2 >> 1", "(8 >> 2) >> 1")]
    #[case("64 >> 2 >> 2 >> 1", "(((64 >> 2) >> 2) >> 1)")]
    #[case("1 | 2 & 3", "1 | (2 & 3)")]
    #[case("1 ^ 2 & 3", "1 ^ (2 & 3)")]
    #[case("1 | 2 ^ 3", "1 | (2 ^ 3)")]
    #[case("1 & 2 ^ 3", "(1 & 2) ^ 3")]
    #[case("1 & 2 << 3", "1 & (2 << 3)")]
    #[case("1 << 2 & 3", "(1 << 2) & 3")]
    #[case("1 ^ 2 | 3", "(1 ^ 2) | 3")]
    #[case("1 << 2 ^ 3", "(1 << 2) ^ 3")]
    #[case("1 << 2 >> 3", "(1 << 2) >> 3")]
    #[case("1 | 2 & 3 ^ 4", "1 | ((2 & 3) ^ 4)")]
    #[case("1 ^ 2 & 3 << 4", "1 ^ (2 & (3 << 4))")]
    #[case("1 & 2 << 3 ^ 4", "(1 & (2 << 3)) ^ 4)")]
    #[case("1 << 2 >> 3 ^ 4", "((1 << 2) >> 3) ^ 4")]
    #[case("1 << 2 | 3", "(1 << 2) | 3")]
    #[case("1 & (2 | 3)", "1 & (2 | 3)")]
    #[case("(1 & 2) | 3", "(1 & 2) | 3")]
    #[case("~1 & 2", "(~1) & 2")]
    #[case("~1 | 2", "(~1) | 2")]
    #[case("-1 << 2", "(-1) << 2")]
    #[case("~1 << 2", "(~1) << 2")]
    #[case("1 << ~2", "1 << (~2)")]
    #[case("1 & 2 | 3 ^ 4", "((1 & 2) | (3 ^ 4))")]
    #[case("1 | 2 ^ 3 & 4", "1 | (2 ^ (3 & 4))")]
    #[case("1 ^ 2 | 3 & 4", "(1 ^ 2) | (3 & 4)")]
    #[case("1 & 2 ^ 3 | 4", "((1 & 2) ^ 3) | 4")]
    #[case("1 ^ 2 & 3 | 4", "((1 ^ (2 & 3)) | 4)")]
    fn test_bitwise_operator_precedence_and_associativity(#[case] src1: &str, #[case] src2: &str) {
        run_expression_equivalence_test(src1, src2);
    }

    #[rstest]
    #[case("1 && 2 && 3", "(1 && 2) && 3")]
    #[case("1 || 2 || 3", "(1 || 2) || 3")]
    #[case("!1 || 0", "(!1) || 0")]
    #[case("1 && !0", "1 && (!0)")]
    #[case("!1 && !0", "(!1) && (!0)")]
    #[case("1 && 2 && 3 && 4", "(((1 && 2) && 3) && 4)")]
    #[case("1 || 2 || 3 || 4", "(((1 || 2) || 3) || 4)")]
    #[case("!1 && 2", "(!1) && 2")]
    #[case("1 && !2", "1 && (!2)")]
    #[case("!1 || 0 && 1", "(!1) || (0 && 1)")]
    #[case("!(1 || 0)", "!(1 || 0)")]
    #[case("1 || 0 && 1", "1 || (0 && 1)")]
    #[case("!1 && !0 || 1", "((!1) && (!0)) || 1")]
    #[case("!(!1)", "!(!1)")]
    #[case("!1 && 0 || 1 && !0", "((!1) && 0) || (1 && (!0))")]
    #[case("1 && 0 || 1 && 0", "((1 && 0) || (1 && 0))")]
    #[case("1 || 0 && 1 || 0", "((1 || (0 && 1)) || 0)")]
    fn test_logical_operator_precedence_and_associativity(#[case] src1: &str, #[case] src2: &str) {
        run_expression_equivalence_test(src1, src2);
    }

    #[rstest]
    #[case("1 < 2 < 3", "(1 < 2) < 3")]  // Not valid logic, but tests associativity in parsing
    #[case("3 > 2 > 1", "(3 > 2) > 1")]
    #[case("1 <= 2 >= 1", "(1 <= 2) >= 1")]
    #[case("1 == 2 == 3", "(1 == 2) == 3")]
    #[case("1 != 2 != 3", "(1 != 2) != 3")]
    #[case("1 < 2 == 1", "(1 < 2) == 1")]
    #[case("3 > 2 != 0", "(3 > 2) != 0")]
    #[case("4 == 4 && 5 > 2", "(4 == 4) && (5 > 2)")]
    #[case("1 < 2 < 3 < 4", "(((1 < 2) < 3) < 4)")]
    #[case("5 > 4 > 3 > 2", "(((5 > 4) > 3) > 2)")]
    #[case("1 <= 2 >= 3 <= 4", "(((1 <= 2) >= 3) <= 4)")]
    #[case("1 == 2 == 3 == 4", "(((1 == 2) == 3) == 4)")]
    #[case("1 != 2 != 3 != 4", "(((1 != 2) != 3) != 4)")]
    #[case("1 < 2 == 3 != 4", "(((1 < 2) == 3) != 4)")]
    #[case("3 > 2 != 1 == 0", "(((3 > 2) != 1) == 0)")]
    #[case("4 == 5 && 6 > 7", "(4 == 5) && (6 > 7)")]
    #[case("1 < 2 && 3 > 4 || 5 == 6", "((1 < 2) && (3 > 4)) || (5 == 6)")]
    #[case("!(1 == 2)", "!(1 == 2)")]
    #[case("!(1 != 2) && 3 < 4", "(!(1 != 2)) && (3 < 4)")]
    fn test_relational_operator_precedence_and_associativity(#[case] src1: &str, #[case] src2: &str) {
        run_expression_equivalence_test(src1, src2);
    }

    fn run_expression_equivalence_test(expr1_src: &str, expr2_src: &str) {
        let lex1 = Lexer::new(expr1_src);
        let mut parser1 = Parser::new(lex1);
        let actual1 = parser1.parse_expression();

        let lex2 = Lexer::new(expr2_src);
        let mut parser2 = Parser::new(lex2);
        let actual2 = parser2.parse_expression();

        assert!(is_equivalent_expression(actual1.unwrap(), actual2.unwrap()),
            "expected {expr1_src} to be equivalent to {expr2_src}");
    }

    fn is_equivalent_expression(e1: Expression, e2: Expression) -> bool {
        match (e1.kind, e2.kind) {
            (IntConstant(c1, r1), IntConstant(c2, r2)) => c1 == c2 && r1 == r2,
            (Unary(uop1, subexp1), Unary(uop2, subexp2)) => uop1 == uop2
                && is_equivalent_expression(*subexp1, *subexp2),
            (Binary(binop1, op11, op12), Binary(binop2, op21, op22)) => binop1 == binop2
                && is_equivalent_expression(*op11, *op21)
                && is_equivalent_expression(*op12, *op22),
            _ => false,
        }
    }
}