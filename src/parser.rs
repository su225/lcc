//! Parser for the language: This implements parser for a subset of
//! features in C programming language. A simple Recursive Descent
//! Parsing is used. It is handwritten.

use std::iter::Peekable;
use thiserror::Error;
use crate::common::{Location, Radix};
use crate::lexer::{KeywordIdentifier, Lexer, LexerError, Token, TokenTag, TokenType};

pub struct Symbol<'a> {
    name: &'a str,
    location: Location,
}

pub(crate) enum ExpressionKind<'a> {
    IntConstant(&'a str, Radix),
}

pub(crate) struct Expression<'a> {
    location: Location,
    kind: ExpressionKind<'a>,
}

pub(crate) enum PrimitiveKind {
    Integer,
    UnsignedInteger,
    LongInteger,
}

pub(crate) enum TypeExpressionKind {
    Primitive(PrimitiveKind),
}

pub(crate) struct TypeExpression {
    location: Location,
    kind: TypeExpressionKind,
}

pub(crate) enum StatementKind<'a> {
    Return(Expression<'a>),
}

pub(crate) struct Statement<'a> {
    location: Location,
    kind: StatementKind<'a>,
}

pub(crate) struct FunctionDefinition<'a> {
    pub(crate) location: Location,
    pub(crate) name: Symbol<'a>,
    pub(crate) body: Vec<Statement<'a>>,
}

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
    UnexpectedToken { location: Location, expected_token_tag: TokenTag },

    #[error("unexpected end of file. Expected {0}")]
    UnexpectedEnd(TokenTag),

    #[error("{location:?}: expected keyword {keyword_identifier:?}")]
    ExpectedKeyword { location: Location, keyword_identifier: KeywordIdentifier },
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
        let func_definition = self.parse_function_definition()?;
        Ok(ProgramDefinition { functions: vec![func_definition] })
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
            Token { location, token_type: TokenType::Keyword(kwid)} => {
                if kwid == expected_kwid {
                    Ok(location)
                } else {
                    Err(ParserError::ExpectedKeyword {location, keyword_identifier: expected_kwid })
                }
            },
            Token { location, .. } => Err(ParserError::ExpectedKeyword { location, keyword_identifier: expected_kwid })
        }
    }

    fn parse_identifier(&mut self) -> Result<Symbol<'a>, ParserError> {
        match self.token_provider.next() {
            Some(Ok(Token { location, token_type })) => {
                match token_type {
                    TokenType::Identifier(name) => Ok(Symbol { name, location }),
                    TokenType::Keyword(kwd) => Err(ParserError::KeywordUsedAsIdentifier { location, kwd }),
                    _ => Err(ParserError::UnexpectedToken { location, expected_token_tag: TokenTag::Identifier })
                }
            },
            Some(Err(e)) => Err(ParserError::TokenizationError(e)),
            None => Err(ParserError::UnexpectedEnd(TokenTag::Identifier)),
        }
    }

    fn parse_type_expression(&mut self) -> Result<TypeExpression, ParserError> {
        let kw_loc = self.expect_keyword(KeywordIdentifier::TypeInt)?;
        Ok(TypeExpression{
            location: kw_loc,
            kind: TypeExpressionKind::Primitive(PrimitiveKind::Integer),
        })
    }

    fn parse_statement(&mut self) -> Result<Statement<'a>, ParserError> {
        self.parse_return_statement()
    }

    fn parse_return_statement(&mut self) -> Result<Statement<'a>, ParserError> {
        let kloc = self.expect_keyword(KeywordIdentifier::TypeVoid)?;
        let return_code_expr = self.parse_expression()?;
        self.expect_semicolon()?;
        Ok(Statement { location: kloc, kind: StatementKind::Return(return_code_expr) })
    }

    fn parse_expression(&mut self) -> Result<Expression<'a>, ParserError> {
        self.parse_int_constant_expression()
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
                    Err(ParserError::UnexpectedToken { location: token.location, expected_token_tag })
                }
            },
            Some(Err(e)) => Err(ParserError::TokenizationError(e)),
            None => Err(ParserError::UnexpectedEnd(expected_token_tag)),
        }
    }
}

#[cfg(test)]
mod test {

}