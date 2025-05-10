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
pub(crate) enum UnaryOperator {
    Complement,
    Negate,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BinaryOperatorAssociativity {
    Left, Right
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BinaryOperator {
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
}

#[derive(Debug, PartialEq, Ord, PartialOrd, Eq, Add, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct BinaryOperatorPrecedence(u16);

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
        }
    }

    #[inline]
    fn precedence(&self) -> BinaryOperatorPrecedence {
        match self {
            BinaryOperator::BitwiseOr => BinaryOperatorPrecedence(25),
            BinaryOperator::BitwiseXor => BinaryOperatorPrecedence(30),
            BinaryOperator::BitwiseAnd => BinaryOperatorPrecedence(35),
            BinaryOperator::LeftShift => BinaryOperatorPrecedence(40),
            BinaryOperator::RightShift => BinaryOperatorPrecedence(40),
            BinaryOperator::Add => BinaryOperatorPrecedence(45),
            BinaryOperator::Subtract => BinaryOperatorPrecedence(45),
            BinaryOperator::Multiply => BinaryOperatorPrecedence(50),
            BinaryOperator::Divide => BinaryOperatorPrecedence(50),
            BinaryOperator::Modulo => BinaryOperatorPrecedence(50),
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExpressionKind<'a> {
    IntConstant(&'a str, Radix),
    Unary(UnaryOperator, Box<Expression<'a>>),
    Binary(BinaryOperator, Box<Expression<'a>>, Box<Expression<'a>>),
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct Expression<'a> {
    location: Location,
    pub(crate) kind: ExpressionKind<'a>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PrimitiveKind {
    Integer,
    UnsignedInteger,
    LongInteger,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TypeExpressionKind {
    Primitive(PrimitiveKind),
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct TypeExpression {
    pub(crate) location: Location,
    pub(crate) kind: TypeExpressionKind,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StatementKind<'a> {
    Return(Expression<'a>),
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct Statement<'a> {
    pub(crate) location: Location,
    pub(crate) kind: StatementKind<'a>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct FunctionDefinition<'a> {
    pub(crate) location: Location,
    pub(crate) name: Symbol<'a>,
    pub(crate) body: Vec<Statement<'a>>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct ProgramDefinition<'a> {
    pub(crate) functions: Vec<FunctionDefinition<'a>>,
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

    #[rstest]
    #[case("simple_addition", "1+2")]
    #[case("simple_subtraction", "1-20")]
    #[case("simple_multiplication", "10*20")]
    #[case("simple_division", "2/4")]
    #[case("simple_remainder", "3%2")]
    #[case("unary_complement", "~10")]
    #[case("unary_negation", "-10")]
    #[case("double_complement", "~~10")]
    #[case("addition_is_left_associative", "1+2+3")]
    #[case("subtraction_is_left_associative", "1-2-3")]
    #[case("multiplication_is_left_associative", "2*3*4")]
    #[case("division_is_left_associative", "10/2/3")]
    #[case("modulo_is_left_associative", "10 % 2 % 3")]
    #[case("multiplication_has_higher_precedence_than_addition", "4+2*3+8")]
    #[case("division_has_higher_precedence_than_addition", "10+4/2+3")]
    #[case("parentheses_override_precedence", "(2+4)*5")]
    #[case("multiple_nested_parentheses", "(10-(2+3))*2")]
    #[case("unary_negate_binary_operator_expression", "-(4+3)")]
    #[case("operation_with_complement_operator", "4+~3")]
    #[case("addition_with_negated_operand", "4+(-3)")]
    #[case("multiplication_with_unary_operands", "~4*-3")]
    #[case("simple_bitwise_and", "10 & 20")]
    #[case("simple_bitwise_or", "10 | 20")]
    #[case("simple_bitwise_xor", "10 ^ 20")]
    #[case("simple_left_shift", "1 << 2")]
    #[case("simple_right_shift", "100 >> 2")]
    #[case("bitwise_and_is_left_associative", "10 & 20 & 30")]
    #[case("bitwise_or_is_left_associative", "10 | 20 | 30")]
    #[case("bitwise_xor_is_left_associative", "10 ^ 20 ^ 30")]
    #[case("left_shift_is_left_associative", "1<<2<<3")]
    #[case("right_shift_is_left_associative", "200>>1>>1")]
    fn should_parse_expression_correctly(#[case] description: &str, #[case] src: &str) {
        let lexer = Lexer::new(src);
        let mut parser = Parser::new(lexer);
        let actual = parser.parse_expression();
        assert!(actual.is_ok());

        with_settings!({
            sort_maps => true,
            prepend_module_to_snapshot => false,
            description => "parsing expression",
            info => &src,
        }, {
            assert_yaml_snapshot!(format!("parse_expression_{description}"), actual.unwrap());
        });
    }

    fn run_parse_expression_test_case(test_case: ExprTestCase) {
        let lexer = Lexer::new(test_case.src);
        let mut parser = Parser::new(lexer);
        let actual = parser.parse_expression();
        assert_eq!(test_case.expected, actual);
    }
}