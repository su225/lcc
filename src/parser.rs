//! Parser for the language: This implements parser for a subset of
//! features in C programming language. A simple Recursive Descent
//! Parsing is used. It is handwritten.

use std::iter::Peekable;

use thiserror::Error;

use crate::common::{Location, Radix};
use crate::lexer::{KeywordIdentifier, Lexer, LexerError, Token, TokenTag, TokenType};

#[derive(Debug, PartialEq)]
pub struct Symbol<'a> {
    pub name: &'a str,
    location: Location,
}

#[derive(Debug, PartialEq)]
pub(crate) enum UnaryOperator {
    Complement,
    Negate,
}

#[derive(Debug, PartialEq)]
pub(crate) enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
}

#[derive(Debug, PartialEq)]
pub(crate) enum ExpressionKind<'a> {
    IntConstant(&'a str, Radix),
    Unary(UnaryOperator, Box<Expression<'a>>),
}

#[derive(Debug, PartialEq)]
pub(crate) struct Expression<'a> {
    location: Location,
    pub(crate) kind: ExpressionKind<'a>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum PrimitiveKind {
    Integer,
    UnsignedInteger,
    LongInteger,
}

#[derive(Debug, PartialEq)]
pub(crate) enum TypeExpressionKind {
    Primitive(PrimitiveKind),
}

#[derive(Debug, PartialEq)]
pub(crate) struct TypeExpression {
    pub(crate) location: Location,
    pub(crate) kind: TypeExpressionKind,
}

#[derive(Debug, PartialEq)]
pub(crate) enum StatementKind<'a> {
    Return(Expression<'a>),
}

#[derive(Debug, PartialEq)]
pub(crate) struct Statement<'a> {
    pub(crate) location: Location,
    pub(crate) kind: StatementKind<'a>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct FunctionDefinition<'a> {
    pub(crate) location: Location,
    pub(crate) name: Symbol<'a>,
    pub(crate) body: Vec<Statement<'a>>,
}

#[derive(Debug, PartialEq)]
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
            Some(Ok(tok)) => {
                match tok.token_type {
                    TokenType::IntConstant(val, radix) => {
                        let loc = tok.location.clone();
                        self.token_provider.next().unwrap().expect("must be int");
                        Ok(Expression {
                            location: loc,
                            kind: ExpressionKind::IntConstant(val, radix)
                        })
                    },
                    TokenType::OpenParentheses => {
                        self.expect_open_parentheses()?;
                        let expr = self.parse_expression()?;
                        self.expect_close_parentheses()?;
                        Ok(expr)
                    },
                    TokenType::OperatorUnaryComplement => {
                        let op_loc = tok.location;
                        self.expect_token_with_tag(TokenTag::OperatorUnaryComplement)?;
                        let expr = self.parse_expression()?;
                        Ok(Expression {
                            location: op_loc,
                            kind: ExpressionKind::Unary(UnaryOperator::Complement, Box::new(expr)),
                        })
                    },
                    TokenType::OperatorMinus => {
                        let op_loc = tok.location;
                        self.expect_token_with_tag(TokenTag::OperatorMinus)?;
                        let expr = self.parse_expression()?;
                        Ok(Expression {
                            location: op_loc,
                            kind: ExpressionKind::Unary(UnaryOperator::Negate, Box::new(expr)),
                        })
                    },
                    _ => {
                        Err(ParserError::UnexpectedToken {
                            location: tok.location,
                            expected_token_tags: vec![
                                TokenTag::OpenParentheses,
                                TokenTag::OperatorUnaryComplement,
                                TokenTag::OperatorMinus,
                            ],
                        })
                    }
                }
            },
            Some(Err(e)) => Err(ParserError::TokenizationError(e.clone())),
            None => Err(ParserError::UnexpectedEnd(vec![TokenTag::IntConstant, TokenTag::OpenParentheses])),
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
            None => Err(ParserError::UnexpectedEnd(vec![expected_token_tag])),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::common::{Location, Radix};
    use crate::common::Radix::Decimal;
    use crate::lexer::Lexer;
    use crate::parser::{Expression, FunctionDefinition, Parser, ParserError, ProgramDefinition, Statement, Symbol, UnaryOperator};
    use crate::parser::ExpressionKind::{IntConstant, Unary};
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
    fn test_parse_expressions() {
        struct ExprTestCase<'a> {
            name: &'a str,
            src: &'a str,
            expected: Result<Expression<'a>, ParserError>,
        }

        for test_case in [
            ExprTestCase {
                name: "constant base-10 integer",
                src: "100",
                expected: Ok(Expression {
                    location: Location { line: 1, column: 1 },
                    kind:  IntConstant("100", Radix::Decimal),
                }),
            },
            ExprTestCase {
                name: "complement operator",
                src: "~0xdeadbeef",
                expected: Ok(Expression {
                    location: Location { line: 1, column: 1 },
                    kind:  Unary(UnaryOperator::Complement, Box::new(Expression {
                        location: Location { line: 1, column: 2 },
                        kind: IntConstant("0xdeadbeef", Radix::Hexadecimal),
                    })),
                }),
            },
            ExprTestCase {
                name: "negation operator",
                src: "-100",
                expected: Ok(Expression {
                    location: Location { line: 1, column: 1 },
                    kind:  Unary(UnaryOperator::Negate, Box::new(Expression {
                        location: Location { line: 1, column: 2 },
                        kind: IntConstant("100", Radix::Decimal),
                    })),
                }),
            },
            ExprTestCase {
                name: "redundant parentheses around int constant",
                src: "(100)",
                expected: Ok(Expression {
                    location: Location { line: 1, column: 2 },
                    kind: IntConstant("100", Decimal),
                }),
            },
            ExprTestCase {
                name: "double complement",
                src: "~~100",
                expected: Ok(Expression {
                    location: Location { line: 1, column: 1 },
                    kind: Unary(UnaryOperator::Complement, Box::new(Expression {
                        location: Location { line: 1, column: 2 },
                        kind: Unary(UnaryOperator::Complement, Box::new(Expression {
                            location: Location { line: 1, column: 3 },
                            kind: IntConstant("100", Decimal),
                        }))
                    })),
                })
            },
            ExprTestCase {
                name: "double negation",
                src: "-(-100)",
                expected: Ok(Expression {
                    location: Location { line: 1, column: 1 },
                    kind: Unary(UnaryOperator::Negate, Box::new(Expression {
                        location: Location { line: 1, column: 3 },
                        kind: Unary(UnaryOperator::Negate, Box::new(Expression {
                            location: Location { line: 1, column: 4 },
                            kind: IntConstant("100", Decimal),
                        }))
                    })),
                })
            }
        ].into_iter() {
            let lexer = Lexer::new(test_case.src);
            let mut parser = Parser::new(lexer);
            let actual_parsed = parser.parse_expression();
            assert_eq!(test_case.expected, actual_parsed, "{}",
                format!("failed case: {name:?},\n expected: {expected:#?},\n actual: {actual:#?}",
                name = test_case.name, expected = test_case.expected, actual = actual_parsed));
        }
    }
}