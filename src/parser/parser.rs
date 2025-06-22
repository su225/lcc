//! Parser for the language: This implements parser for a subset of
//! features in C programming language. A simple Recursive Descent
//! Parsing is used. It is handwritten.

use std::iter::Peekable;

use derive_more::Add;
use serde::Serialize;
use thiserror::Error;

use crate::common::{Location, Radix};
use crate::lexer::{KeywordIdentifier, Lexer, LexerError, Token, TokenTag, TokenType};
use crate::parser::ParserError::*;

#[derive(Debug, Hash, Eq, PartialEq, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct Symbol {
    pub name: String,
    pub(crate) location: Location,
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOperator {
    Complement,
    Negate,
    Not,
    Increment,
    Decrement,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOperatorAssociativity {
    Left,
    Right,
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize)]
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

    Assignment,
}

#[derive(Debug, PartialEq, Ord, PartialOrd, Eq, Add, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct BinaryOperatorPrecedence(pub(crate) u16);

impl BinaryOperator {
    #[inline]
    pub(crate) fn associativity(&self) -> BinaryOperatorAssociativity {
        match self {
            BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo
            | BinaryOperator::BitwiseAnd
            | BinaryOperator::BitwiseOr
            | BinaryOperator::BitwiseXor
            | BinaryOperator::LeftShift
            | BinaryOperator::RightShift
            | BinaryOperator::And
            | BinaryOperator::Or
            | BinaryOperator::Equal
            | BinaryOperator::NotEqual
            | BinaryOperator::LessThan
            | BinaryOperator::LessThanOrEqual
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterThanOrEqual => BinaryOperatorAssociativity::Left,

            BinaryOperator::Assignment => BinaryOperatorAssociativity::Right,
        }
    }

    #[inline]
    pub(crate) fn precedence(&self) -> BinaryOperatorPrecedence {
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

            BinaryOperator::Assignment => BinaryOperatorPrecedence(10),
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpressionKind {
    IntConstant(String, Radix),
    Variable(String),
    Unary(UnaryOperator, Box<Expression>),
    Binary(BinaryOperator, Box<Expression>, Box<Expression>),
    Assignment { lvalue: Box<Expression>, rvalue: Box<Expression> },
    Increment { is_post: bool, e: Box<Expression> },
    Decrement { is_post: bool, e: Box<Expression> },
}

impl ExpressionKind {
    pub fn is_lvalue_expression(&self) -> bool {
        match self {
            ExpressionKind::Variable(_) => true,
            _ => false,
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Expression {
    pub(crate) location: Location,
    pub kind: ExpressionKind,
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
pub enum StatementKind {
    Return(Expression),
    Expression(Expression),
    SubBlock(Block),
    Null,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Statement {
    pub location: Location,
    pub kind: StatementKind,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclarationKind {
    Declaration {
        identifier: Symbol,
        init_expression: Option<Expression>,
    },
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Declaration {
    pub location: Location,
    pub kind: DeclarationKind,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockItem {
    Statement(Statement),
    Declaration(Declaration),
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Block {
    pub start_loc: Location,
    pub end_loc: Location,
    pub items: Vec<BlockItem>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct FunctionDefinition {
    pub location: Location,
    pub name: Symbol,
    pub body: Block,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ProgramDefinition {
    pub functions: Vec<FunctionDefinition>,
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

pub struct Parser<'a> {
    token_provider: Peekable<Lexer<'a>>,
}

impl<'a> Parser<'a> {
    pub fn new(token_provider: Lexer<'a>) -> Parser<'a> {
        Parser { token_provider: token_provider.peekable() }
    }

    /// parse parses the given source file and returns the
    /// Abstract Syntax Tree (AST).
    pub fn parse(&mut self) -> Result<ProgramDefinition, ParserError> {
        Ok(self.parse_program()?)
    }

    fn parse_program(&mut self) -> Result<ProgramDefinition, ParserError> {
        let mut functions: Vec<FunctionDefinition> = vec![];
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

    fn parse_function_definition(&mut self) -> Result<FunctionDefinition, ParserError> {
        let return_type = self.parse_type_expression()?;
        let name = self.parse_identifier()?;
        self.expect_open_parentheses()?;
        self.parse_function_parameters()?;
        self.expect_close_parentheses()?;
        let body = self.parse_block()?;
        Ok(FunctionDefinition { location: return_type.location, name, body })
    }

    fn parse_function_parameters(&mut self) -> Result<(), ParserError> {
        self.expect_keyword(KeywordIdentifier::TypeVoid)?;
        Ok(())
    }

    fn parse_block(&mut self) -> Result<Block, ParserError> {
        let block_open = self.get_token_with_tag(TokenTag::OpenBrace)?;
        let open_loc = block_open.location.clone();
        let mut block_items = Vec::with_capacity(2);
        loop {
            let next_token = self.token_provider.peek();
            match next_token {
                None => { return Err(UnexpectedEnd(vec![TokenTag::CloseBrace])); }
                Some(Ok(tok)) if tok.token_type.tag() == TokenTag::CloseBrace => { break; }
                Some(Ok(_)) => {
                    let block_item = self.parse_block_item()?;
                    block_items.push(block_item);
                }
                Some(Err(e)) => { return Err(TokenizationError(e.clone())) }
            };
        }
        let block_close = self.get_token_with_tag(TokenTag::CloseBrace)?;
        Ok(Block {
            start_loc: open_loc,
            end_loc: block_close.location,
            items: block_items,
        })
    }

    fn parse_block_item(&mut self) -> Result<BlockItem, ParserError> {
        let tok = self.token_provider.peek();
        if tok.is_none() {
            return Err(UnexpectedEnd(vec![TokenTag::Semicolon]));
        }
        match tok.unwrap() {
            Ok(Token { token_type, .. }) => {
                match token_type {
                    TokenType::Keyword(KeywordIdentifier::TypeInt) => {
                        let decl = self.parse_declaration()?;
                        Ok(BlockItem::Declaration(decl))
                    }
                    _ => {
                        let stmt = self.parse_statement()?;
                        Ok(BlockItem::Statement(stmt))
                    }
                }
            }
            Err(e) => Err(TokenizationError(e.clone())),
        }
    }

    fn parse_declaration(&mut self) -> Result<Declaration, ParserError> {
        let ty_decl = self.get_keyword_token(KeywordIdentifier::TypeInt)?;
        let var_name = self.parse_identifier()?;
        let next_tok = self.token_provider.next();
        match next_tok {
            None => Err(UnexpectedEnd(vec![TokenTag::OperatorAssignment, TokenTag::Semicolon])),
            Some(Err(e)) => Err(TokenizationError(e.clone())),
            Some(Ok(tok)) => {
                let tok_loc = tok.location;
                match tok.token_type {
                    TokenType::OperatorAssignment => {
                        let init_expr = self.parse_expression()?;
                        self.expect_semicolon()?;
                        Ok(Declaration {
                            location: ty_decl.location.clone(),
                            kind: DeclarationKind::Declaration {
                                identifier: var_name,
                                init_expression: Some(init_expr),
                            },
                        })
                    }
                    TokenType::Semicolon => {
                        Ok(Declaration {
                            location: ty_decl.location.clone(),
                            kind: DeclarationKind::Declaration {
                                identifier: var_name,
                                init_expression: None,
                            },
                        })
                    }
                    _ => Err(UnexpectedToken {
                        location: tok_loc,
                        expected_token_tags: vec![TokenTag::OperatorAssignment, TokenTag::Semicolon],
                    })
                }
            }
        }
    }

    fn get_keyword_token(&mut self, kw_ident_type: KeywordIdentifier) -> Result<Token<'a>, ParserError> {
        let kwd = self.get_token_with_tag(TokenTag::Keyword)?;
        let kwd_loc = kwd.location;
        match kwd.token_type {
            TokenType::Keyword(kw) if kw_ident_type == kw => Ok(Token {
                location: kwd_loc,
                token_type: TokenType::Keyword(kw_ident_type),
            }),
            _ => Err(UnexpectedToken {
                location: kwd_loc,
                expected_token_tags: vec![TokenTag::Keyword],
            })
        }
    }

    fn expect_open_parentheses(&mut self) -> Result<(), ParserError> {
        self.expect_token_with_tag(TokenTag::OpenParentheses)
    }

    fn expect_close_parentheses(&mut self) -> Result<(), ParserError> {
        self.expect_token_with_tag(TokenTag::CloseParentheses)
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
                    Err(ExpectedKeyword { location, keyword_identifier: expected_kwid, actual_token: TokenTag::Keyword })
                }
            }
            Token { location, token_type } => Err(ExpectedKeyword {
                location,
                keyword_identifier: expected_kwid,
                actual_token: token_type.tag(),
            })
        }
    }

    fn parse_identifier(&mut self) -> Result<Symbol, ParserError> {
        match self.token_provider.next() {
            Some(Ok(Token { location, token_type })) => {
                match token_type {
                    TokenType::Identifier(name) => Ok(Symbol { name: name.to_string(), location }),
                    TokenType::Keyword(kwd) => Err(KeywordUsedAsIdentifier { location, kwd }),
                    _ => Err(UnexpectedToken { location, expected_token_tags: vec![TokenTag::Identifier] })
                }
            }
            Some(Err(e)) => Err(TokenizationError(e)),
            None => Err(UnexpectedEnd(vec![TokenTag::Identifier])),
        }
    }

    fn parse_type_expression(&mut self) -> Result<TypeExpression, ParserError> {
        let kw_loc = self.expect_keyword(KeywordIdentifier::TypeInt)?;
        Ok(TypeExpression {
            location: kw_loc,
            kind: TypeExpressionKind::Primitive(PrimitiveKind::Integer),
        })
    }

    fn parse_statement(&mut self) -> Result<Statement, ParserError> {
        let tok = self.token_provider.peek();
        if tok.is_none() {
            return Err(UnexpectedEnd(vec![TokenTag::Semicolon]));
        }
        match tok.unwrap() {
            Ok(Token { token_type, location }) => {
                let tok_loc = location.clone();
                match token_type {
                    TokenType::OpenBrace => {
                        let sub_block = self.parse_block()?;
                        Ok(Statement { location: tok_loc, kind: StatementKind::SubBlock(sub_block) })
                    }
                    TokenType::Semicolon => {
                        self.token_provider.next();
                        Ok(Statement { location: tok_loc, kind: StatementKind::Null })
                    }
                    TokenType::Keyword(KeywordIdentifier::Return) => self.parse_return_statement(),
                    _ => self.parse_expression_statement(),
                }
            }
            Err(e) => Err(TokenizationError(e.clone())),
        }
    }

    fn parse_return_statement(&mut self) -> Result<Statement, ParserError> {
        let kloc = self.expect_keyword(KeywordIdentifier::Return)?;
        let return_code_expr = self.parse_expression()?;
        self.expect_semicolon()?;
        Ok(Statement { location: kloc, kind: StatementKind::Return(return_code_expr) })
    }

    fn parse_expression_statement(&mut self) -> Result<Statement, ParserError> {
        let expr_stmt = self.parse_expression()?;
        self.expect_semicolon()?;
        Ok(Statement { location: expr_stmt.location.clone(), kind: StatementKind::Expression(expr_stmt) })
    }

    fn parse_expression(&mut self) -> Result<Expression, ParserError> {
        let tok = self.token_provider.peek();
        match &tok {
            Some(Ok(_)) => self.parse_expression_with_precedence(BinaryOperatorPrecedence(0)),
            Some(Err(e)) => Err(TokenizationError(e.clone())),
            None => Err(UnexpectedEnd(vec![TokenTag::IntConstant, TokenTag::OpenParentheses])),
        }
    }

    fn parse_expression_with_precedence(&mut self, min_precedence: BinaryOperatorPrecedence) -> Result<Expression, ParserError> {
        let mut left = self.parse_factor()?;
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
                    let left_loc = left.location.clone();
                    let expr_kind = if binary_op == BinaryOperator::Assignment {
                        ExpressionKind::Assignment { lvalue: Box::new(left), rvalue: Box::new(rhs) }
                    } else {
                        ExpressionKind::Binary(binary_op, Box::new(left), Box::new(rhs))
                    };
                    left = Expression {
                        location: left_loc,
                        kind: expr_kind,
                    }
                }
                Ok(_) => {
                    // It is not an error to see something else.
                    // Think of something like "10 + 20;" Here semicolon
                    // is a token which is not a binary operator. In this
                    // case, we should not treat it as an error.
                    break;
                }
                Err(e) => {
                    return Err(TokenizationError(e.clone()));
                }
            };
        }
        Ok(left)
    }

    fn parse_factor(&mut self) -> Result<Expression, ParserError> {
        let next_token = self.token_provider.peek();
        let f = match &next_token {
            Some(Ok(Token { token_type, location })) => {
                let tok_location = location.clone();
                match token_type {
                    TokenType::IntConstant(_, _) => self.parse_int_constant_expression(),
                    op if op.is_unary_operator() => {
                        let unary_op = self.parse_unary_operator_token()?;
                        let factor = self.parse_factor()?;
                        Ok(Expression {
                            location: tok_location,
                            kind: match unary_op {
                                UnaryOperator::Complement |
                                UnaryOperator::Negate |
                                UnaryOperator::Not => ExpressionKind::Unary(unary_op, Box::new(factor)),
                                UnaryOperator::Increment => ExpressionKind::Increment { is_post: false, e: Box::new(factor) },
                                UnaryOperator::Decrement => ExpressionKind::Decrement { is_post: false, e: Box::new(factor) },
                            },
                        })
                    }
                    TokenType::OpenParentheses => {
                        self.expect_token_with_tag(TokenTag::OpenParentheses)?;
                        let expr = self.parse_expression()?;
                        self.expect_token_with_tag(TokenTag::CloseParentheses)?;
                        Ok(expr)
                    }
                    TokenType::Identifier(identifier) => {
                        let res = Ok(Expression {
                            location: tok_location,
                            kind: ExpressionKind::Variable(identifier.to_string()),
                        });
                        self.token_provider.next();
                        res
                    }
                    _ => Err(UnexpectedToken {
                        location: location.clone(),
                        expected_token_tags: vec![
                            TokenTag::IntConstant,
                            TokenTag::OperatorUnaryComplement,
                            TokenTag::OperatorUnaryComplement,
                            TokenTag::OpenParentheses,
                        ],
                    })
                }
            }
            Some(Err(e)) => Err(TokenizationError(e.clone())),
            None => Err(UnexpectedEnd(vec![TokenTag::IntConstant])),
        }?;
        
        // Check if we have a postfix increment or decrement operators.
        // If yes, then the factor previously parsed has to be bound to it
        let next_token = self.token_provider.peek();
        match &next_token {
            None => Ok(f),
            Some(Ok(Token { token_type, .. })) => {
                if token_type.tag() == TokenTag::OperatorUnaryIncrement {
                    self.token_provider.next();
                    Ok(Expression {
                        location: f.location.clone(),
                        kind: ExpressionKind::Increment { is_post: true, e: Box::new(f) },
                    })    
                } else if token_type.tag() == TokenTag::OperatorUnaryDecrement {
                    self.token_provider.next();
                     Ok(Expression {
                         location: f.location.clone(),
                         kind: ExpressionKind::Decrement { is_post: true, e: Box::new(f) },
                     })
                } else {
                    Ok(f)
                }
            },
            Some(Err(e)) => Err(TokenizationError(e.clone())),
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
                    TokenType::OperatorUnaryIncrement => Ok(UnaryOperator::Increment),
                    TokenType::OperatorUnaryDecrement => Ok(UnaryOperator::Decrement),
                    tok_type => Err(ExpectedUnaryOperator { location, actual_token: tok_type.tag() })
                }
            }
            Some(Err(e)) => Err(TokenizationError(e)),
        }
    }

    fn peek_binary_operator_token(&mut self) -> Result<BinaryOperator, ParserError> {
        let op_tok = self.token_provider.peek();
        match &op_tok {
            None => Err(UnexpectedEnd(vec![TokenTag::OperatorPlus])),
            Some(Err(e)) => Err(TokenizationError(e.clone())),
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
                    TokenType::OperatorAssignment => Ok(BinaryOperator::Assignment),
                    tok_type => Err(ExpectedBinaryOperator { location: location.clone(), actual_token: tok_type.tag() })
                }
            }
        }
    }

    fn parse_int_constant_expression(&mut self) -> Result<Expression, ParserError> {
        let tok = self.get_token_with_tag(TokenTag::IntConstant)?;
        let tok_loc = tok.location;
        match tok.token_type {
            TokenType::IntConstant(c, rad) => Ok(Expression {
                location: tok_loc,
                kind: ExpressionKind::IntConstant(c.to_string(), rad),
            }),
            _ => panic!("should not reach here"),
        }
    }

    fn expect_token_with_tag(&mut self, expected_token_tag: TokenTag) -> Result<(), ParserError> {
        self.get_token_with_tag(expected_token_tag)?;
        Ok(())
    }

    fn get_token_with_tag(&mut self, expected_token_tag: TokenTag) -> Result<Token, ParserError> {
        let token = self.token_provider.next();
        match token {
            Some(Ok(token)) => {
                let token_tag = token.token_type.tag();
                if token_tag == expected_token_tag {
                    Ok(token)
                } else {
                    Err(UnexpectedToken { location: token.location, expected_token_tags: vec![expected_token_tag] })
                }
            }
            Some(Err(e)) => Err(TokenizationError(e)),
            None => Err(UnexpectedEnd(vec![expected_token_tag])),
        }
    }
}

#[cfg(test)]
mod test {
    use std::fs;
    use std::path::{Path, PathBuf};

    use indoc::indoc;
    use rstest::rstest;

    use crate::common::{Location, Radix};
    use crate::common::Radix::Decimal;
    use crate::lexer::Lexer;
    use crate::parser::{BinaryOperator, Block, BlockItem, Declaration, DeclarationKind, Expression, FunctionDefinition, Parser, ParserError, ProgramDefinition, Statement, StatementKind, Symbol, UnaryOperator};
    use crate::parser::ExpressionKind::*;
    use crate::parser::StatementKind::*;

    #[test]
    fn test_parse_program_with_tabs() {
        let src = "int	main	(	void)	{	return	0	;	}";
        let lexer = Lexer::new(src);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert_eq!(Ok(ProgramDefinition {
            functions: vec![
                FunctionDefinition {
                    location: Location { line: 1, column: 1 },
                    name: Symbol {
                        name: "main".to_string(),
                        location: Location { line: 1, column: 8 },
                    },
                    body: Block {
                        start_loc: Location { line: 1, column: 32 }, // ← updated from (0,0)
                        end_loc: Location { line: 1, column: 64 },   // ← updated from (0,0)
                        items: vec![
                            BlockItem::Statement(
                                Statement {
                                    location: Location { line: 1, column: 40 },
                                    kind: Return(Expression {
                                        location: Location { line: 1, column: 48 },
                                        kind: IntConstant("0".to_string(), Decimal),
                                    }),
                                },
                            ),
                        ],
                    },
                },
            ],
        }), parsed);
    }

    #[test]
    fn test_parse_multiple_functions() {
        let src = indoc!(r#"
            int main(void) {
                return 2;
            }

            int foo(void) {
                return 3;
            }
        "#);
        let lexer = Lexer::new(src);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert_eq!(Ok(ProgramDefinition {
            functions: vec![
                FunctionDefinition {
                    location: (1, 1).into(),
                    name: Symbol { name: "main".to_string(), location: (1, 5).into() },
                    body: Block {
                        start_loc: (1, 16).into(),
                        end_loc: (3, 1).into(),
                        items: vec![
                            BlockItem::Statement(Statement {
                                location: (2, 5).into(),
                                kind: Return(Expression {
                                    location: (2, 12).into(),
                                    kind: IntConstant("2".to_string(), Decimal),
                                }),
                            })
                        ],
                    },
                },
                FunctionDefinition {
                    location: (5, 1).into(),
                    name: Symbol { name: "foo".to_string(), location: (5, 5).into() },
                    body: Block {
                        start_loc: (5, 15).into(),
                        end_loc: (7, 1).into(),
                        items: vec![
                            BlockItem::Statement(Statement {
                                location: (6, 5).into(), // ← updated from (7,17)
                                kind: Return(Expression {
                                    location: (6, 12).into(), // ← updated from (7,24)
                                    kind: IntConstant("3".to_string(), Decimal),
                                }),
                            })
                        ],
                    },
                },
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
                    name: Symbol { name: "main".to_string(), location: (1, 5).into() },
                    body: Block {
                        start_loc: (1, 16).into(),
                        end_loc: (3, 1).into(),
                        items: vec![
                            BlockItem::Statement(
                                Statement {
                                    location: (2, 5).into(),
                                    kind: Return(Expression {
                                        location: (2, 12).into(),
                                        kind: Binary(
                                            BinaryOperator::Add,
                                            Box::new(Expression { location: (2, 12).into(), kind: IntConstant("1".to_string(), Decimal) }),
                                            Box::new(Expression { location: (2, 16).into(), kind: IntConstant("2".to_string(), Decimal) }),
                                        ),
                                    }),
                                },
                            ),
                        ],
                    },
                },
            ],
        });
        assert_eq!(expected, actual, "expected:\n{:#?}\nactual:\n{:#?}\n", expected, actual);
    }

    struct StatementTestCase<'a> {
        src: &'a str,
        expected: Result<Statement, ParserError>,
    }

    fn run_parse_statement_test_case(test_case: StatementTestCase) {
        let lexer = Lexer::new(test_case.src);
        let mut parser = Parser::new(lexer);
        let actual = parser.parse_statement();
        assert_eq!(test_case.expected, actual);
    }

    #[test]
    fn test_parse_statement_empty() {
        let src = ";";
        let expected = Ok(Statement { location: (1, 1).into(), kind: Null });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_return() {
        let src = "return 10;";
        let expected = Ok(Statement {
            location: (1, 1).into(),
            kind: Return(Expression {
                location: (1, 8).into(),
                kind: IntConstant("10".to_string(), Decimal),
            }),
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_simple_assignment() {
        let src = "a = 10;";
        let expected = Ok(Statement {
            location: (1, 1).into(),
            kind: StatementKind::Expression(Expression {
                location: (1, 1).into(),
                kind: Assignment {
                    lvalue: Box::new(Expression {
                        location: (1, 1).into(),
                        kind: Variable("a".to_string()),
                    }),
                    rvalue: Box::new(Expression {
                        location: (1, 5).into(),
                        kind: IntConstant("10".to_string(), Decimal),
                    }),
                },
            }),
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    struct ExprTestCase<'a> {
        src: &'a str,
        expected: Result<Expression, ParserError>,
    }

    fn run_parse_expression_test_case(test_case: ExprTestCase) {
        let lexer = Lexer::new(test_case.src);
        let mut parser = Parser::new(lexer);
        let actual = parser.parse_expression();
        assert_eq!(test_case.expected, actual);
    }

    #[test]
    fn test_parse_expression_constant_base_10_integer() {
        let src = "100";
        let expected = Ok(Expression {
            location: Location { line: 1, column: 1 },
            kind: IntConstant("100".to_string(), Decimal),
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
                kind: IntConstant("0xdeadbeef".to_string(), Radix::Hexadecimal),
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
                kind: IntConstant("100".to_string(), Decimal),
            })),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected })
    }

    #[test]
    fn test_parse_expression_redundant_parentheses_around_int_constant() {
        let src = "(100)";
        let expected = Ok(Expression {
            location: Location { line: 1, column: 2 },
            kind: IntConstant("100".to_string(), Decimal),
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
                    kind: IntConstant("100".to_string(), Decimal),
                })),
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
                            kind: IntConstant("100".to_string(), Decimal),
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
                    kind: IntConstant("10".to_string(), Decimal),
                }),
                Box::new(Expression {
                    location: (1, 6).into(),
                    kind: IntConstant("20".to_string(), Decimal),
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
                    kind: IntConstant("30".to_string(), Decimal),
                }),
                Box::new(Expression {
                    location: (1, 6).into(),
                    kind: IntConstant("15".to_string(), Decimal),
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
                    kind: IntConstant("4".to_string(), Decimal),
                }),
                Box::new(Expression {
                    location: (1, 5).into(),
                    kind: IntConstant("5".to_string(), Decimal),
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
                    kind: IntConstant("100".to_string(), Decimal),
                }),
                Box::new(Expression {
                    location: (1, 7).into(),
                    kind: IntConstant("25".to_string(), Decimal),
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
                            kind: IntConstant("1".to_string(), Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 3).into(),
                            kind: IntConstant("2".to_string(), Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 5).into(),
                    kind: IntConstant("3".to_string(), Decimal),
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
                            kind: IntConstant("5".to_string(), Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 3).into(),
                            kind: IntConstant("3".to_string(), Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 5).into(),
                    kind: IntConstant("1".to_string(), Decimal),
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
                            kind: IntConstant("2".to_string(), Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 3).into(),
                            kind: IntConstant("3".to_string(), Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 5).into(),
                    kind: IntConstant("4".to_string(), Decimal),
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
                            kind: IntConstant("20".to_string(), Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 4).into(),
                            kind: IntConstant("5".to_string(), Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 6).into(),
                    kind: IntConstant("2".to_string(), Decimal),
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
                            kind: IntConstant("10".to_string(), Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 4).into(),
                            kind: IntConstant("4".to_string(), Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 6).into(),
                    kind: IntConstant("2".to_string(), Decimal),
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
                    kind: IntConstant("2".to_string(), Decimal),
                }),
                Box::new(Expression {
                    location: (1, 3).into(),
                    kind: Binary(
                        BinaryOperator::Multiply,
                        Box::new(Expression {
                            location: (1, 3).into(),
                            kind: IntConstant("3".to_string(), Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 5).into(),
                            kind: IntConstant("4".to_string(), Decimal),
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
                    kind: IntConstant("20".to_string(), Decimal),
                }),
                Box::new(Expression {
                    location: (1, 4).into(),
                    kind: Binary(
                        BinaryOperator::Divide,
                        Box::new(Expression {
                            location: (1, 4).into(),
                            kind: IntConstant("6".to_string(), Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 6).into(),
                            kind: IntConstant("2".to_string(), Decimal),
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
                    kind: IntConstant("9".to_string(), Decimal),
                }),
                Box::new(Expression {
                    location: (1, 3).into(),
                    kind: Binary(
                        BinaryOperator::Modulo,
                        Box::new(Expression {
                            location: (1, 3).into(),
                            kind: IntConstant("8".to_string(), Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 5).into(),
                            kind: IntConstant("5".to_string(), Decimal),
                        }),
                    ),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_simple_assignment() {
        let src = "a=10";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1, 1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1, 3).into(),
                    kind: IntConstant("10".to_string(), Decimal),
                }),
            },
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_simple_assignment_as_right_associative() {
        let src = "a=b=10";
        let expected = Ok(Expression {
            location: (1, 1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1, 1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1, 3).into(),
                    kind: Assignment {
                        lvalue: Box::new(Expression {
                            location: (1, 3).into(),
                            kind: Variable("b".to_string()),
                        }),
                        rvalue: Box::new(Expression {
                            location: (1, 5).into(),
                            kind: IntConstant("10".to_string(), Decimal),
                        }),
                    },
                }),
            },
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
                            kind: IntConstant("2".to_string(), Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 4).into(),
                            kind: IntConstant("3".to_string(), Decimal),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 7).into(),
                    kind: IntConstant("4".to_string(), Decimal),
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
                            kind: IntConstant("10".to_string(), Decimal),
                        }),
                        Box::new(Expression {
                            location: (1, 6).into(),
                            kind: Binary(
                                BinaryOperator::Add,
                                Box::new(Expression {
                                    location: (1, 6).into(),
                                    kind: IntConstant("2".to_string(), Decimal),
                                }),
                                Box::new(Expression {
                                    location: (1, 8).into(),
                                    kind: IntConstant("3".to_string(), Decimal),
                                }),
                            ),
                        }),
                    ),
                }),
                Box::new(Expression {
                    location: (1, 12).into(),
                    kind: IntConstant("2".to_string(), Decimal),
                }),
            ),
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_post_increment() {
        let src = "a++";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Increment {
                is_post: true,
                e: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string())
                })
            }
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_post_decrement() {
        let src = "a--";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Decrement {
                is_post: true,
                e: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string())
                })
            }
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_pre_decrement() {
        let src = "--a";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Decrement {
                is_post: false,
                e: Box::new(Expression {
                    location: (1,3).into(),
                    kind: Variable("a".to_string())
                })
            }
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_pre_increment() {
        let src = "++a";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Increment {
                is_post: false,
                e: Box::new(Expression {
                    location: (1,3).into(),
                    kind: Variable("a".to_string())
                })
            }
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parenthesized_increment_expressions() {
        run_parse_expression_test_case(ExprTestCase {
            src: "(a)++",
            expected: Ok(Expression {
                location: (1,2).into(),
                kind: Increment {
                    is_post: true,
                    e: Box::new(Expression {
                        location: (1,2).into(),
                        kind: Variable("a".to_string()),
                    }),
                },
            }),
        });
        run_parse_expression_test_case(ExprTestCase {
            src: "++(a)",
            expected: Ok(Expression {
                location: (1,1).into(),
                kind: Increment {
                    is_post: false,
                    e: Box::new(Expression {
                        location: (1,4).into(),
                        kind: Variable("a".to_string()),
                    }),
                },
            }),
        });
    }

    #[test]
    fn test_parenthesized_decrement_expressions() {
        run_parse_expression_test_case(ExprTestCase {
            src: "(a)--",
            expected: Ok(Expression {
                location: (1,2).into(),
                kind: Decrement {
                    is_post: true,
                    e: Box::new(Expression {
                        location: (1,2).into(),
                        kind: Variable("a".to_string()),
                    }),
                },
            }),
        });
        run_parse_expression_test_case(ExprTestCase {
            src: "--(a)",
            expected: Ok(Expression {
                location: (1,1).into(),
                kind: Decrement {
                    is_post: false,
                    e: Box::new(Expression {
                        location: (1,4).into(),
                        kind: Variable("a".to_string()),
                    }),
                },
            }),
        });
    }

    struct DeclarationTestCase<'a> {
        src: &'a str,
        expected: Result<Declaration, ParserError>,
    }

    fn run_parse_declaration_test_case(test_case: DeclarationTestCase) {
        let lexer = Lexer::new(test_case.src);
        let mut parser = Parser::new(lexer);
        let actual = parser.parse_declaration();
        assert_eq!(test_case.expected, actual);
    }

    #[test]
    fn test_parse_declaration_without_initialization() {
        let src = "int a;";
        let expected = Ok(Declaration {
            location: (1, 1).into(),
            kind: DeclarationKind::Declaration {
                identifier: Symbol {
                    name: "a".to_string(),
                    location: (1, 5).into(),
                },
                init_expression: None,
            },
        });
        run_parse_declaration_test_case(DeclarationTestCase { src, expected });
    }

    #[test]
    fn test_parse_declaration_with_initialization() {
        let src = "int a = 10;";
        let expected = Ok(Declaration {
            location: (1, 1).into(),
            kind: DeclarationKind::Declaration {
                identifier: Symbol {
                    name: "a".to_string(),
                    location: (1, 5).into(),
                },
                init_expression: Some(super::Expression {
                    location: (1, 9).into(),
                    kind: IntConstant("10".to_string(), Decimal),
                }),
            },
        });
        run_parse_declaration_test_case(DeclarationTestCase { src, expected });
    }

    struct BlockTestCase<'a> {
        src: &'a str,
        expected: Result<Block, ParserError>,
    }

    fn run_parse_block_test_case(test_case: BlockTestCase) {
        let lexer = Lexer::new(test_case.src);
        let mut parser = Parser::new(lexer);
        let actual = parser.parse_block();
        assert_eq!(test_case.expected, actual);
    }

    #[test]
    fn test_parse_block_empty() {
        let src = "{}";
        let expected = Ok(Block {
            start_loc: (1, 1).into(),
            end_loc: (1, 2).into(),
            items: vec![],
        });
        run_parse_block_test_case(BlockTestCase { src, expected })
    }

    #[test]
    fn test_parse_block_return_0() {
        let src = "{}";
        let expected = Ok(Block {
            start_loc: (1, 1).into(),
            end_loc: (1, 2).into(),
            items: vec![],
        });
        run_parse_block_test_case(BlockTestCase { src, expected })
    }

    #[test]
    fn test_parse_block_with_variable_declaration() {
        let src = indoc! {r#"
        {
            return 0;
        }
        "#};
        let expected = Ok(Block {
            start_loc: (1, 1).into(),
            end_loc: (3, 1).into(),
            items: vec![
                BlockItem::Statement(Statement {
                    location: (2, 5).into(),
                    kind: Return(Expression {
                        location: (2, 12).into(),
                        kind: IntConstant("0".to_string(), Decimal),
                    }),
                }),
            ],
        });
        run_parse_block_test_case(BlockTestCase { src, expected })
    }

    #[test]
    fn test_parse_block_multiple_statements_with_declarations() {
        let src = indoc! {r#"
        {
            int a = 10;
            int b;
            b = 10;
        }
        "#};
        let expected = Ok(Block {
            start_loc: (1, 1).into(),
            end_loc: (5, 1).into(),
            items: vec![
                BlockItem::Declaration(Declaration {
                    location: (2, 5).into(),
                    kind: DeclarationKind::Declaration {
                        identifier: Symbol {
                            location: (2, 9).into(),
                            name: "a".to_string(),
                        },
                        init_expression: Some(Expression {
                            location: (2, 13).into(),
                            kind: IntConstant("10".to_string(), Decimal),
                        }),
                    },
                }),
                BlockItem::Declaration(Declaration {
                    location: (3, 5).into(), // fixed from (2,5)
                    kind: DeclarationKind::Declaration {
                        identifier: Symbol {
                            location: (3, 9).into(),
                            name: "b".to_string(),
                        },
                        init_expression: None,
                    },
                }),
                BlockItem::Statement(Statement {
                    location: (4, 5).into(), // fixed from (3,5)
                    kind: StatementKind::Expression(Expression {
                        location: (4, 5).into(),
                        kind: Assignment {
                            lvalue: Box::new(Expression {
                                location: (4, 5).into(),
                                kind: Variable("b".to_string()),
                            }),
                            rvalue: Box::new(Expression {
                                location: (4, 9).into(),
                                kind: IntConstant("10".to_string(), Decimal),
                            }),
                        },
                    }),
                }),
            ],
        });
        run_parse_block_test_case(BlockTestCase { src, expected })
    }

    #[test]
    fn test_parse_block_subblocks() {
        let src = indoc! {r#"
        {
          int a = 10;
          {
            int b = 20;
            int c = 30;
            a = b + c;
          }
          return 0;
        }
        "#};
        let expected = Ok(Block {
            start_loc: (1, 1).into(),
            end_loc: (9, 1).into(),
            items: vec![
                BlockItem::Declaration(Declaration {
                    location: (2, 3).into(),
                    kind: DeclarationKind::Declaration {
                        identifier: Symbol {
                            name: "a".to_string(),
                            location: (2, 7).into(),
                        },
                        init_expression: Some(Expression {
                            location: (2, 11).into(),
                            kind: IntConstant("10".to_string(), Decimal),
                        }),
                    },
                }),
                BlockItem::Statement(Statement {
                    location: (3, 3).into(),
                    kind: SubBlock(Block {
                        start_loc: (3, 3).into(),
                        end_loc: (7, 3).into(),
                        items: vec![
                            BlockItem::Declaration(Declaration {
                                location: (4, 5).into(),
                                kind: DeclarationKind::Declaration {
                                    identifier: Symbol { location: (4, 9).into(), name: "b".to_string() },
                                    init_expression: Some(Expression {
                                        location: (4, 13).into(),
                                        kind: IntConstant("20".to_string(), Decimal),
                                    }),
                                },
                            }),
                            BlockItem::Declaration(Declaration {
                                location: (5, 5).into(),
                                kind: DeclarationKind::Declaration {
                                    identifier: Symbol { location: (5, 9).into(), name: "c".to_string() },
                                    init_expression: Some(Expression {
                                        location: (5, 13).into(),
                                        kind: IntConstant("30".to_string(), Decimal),
                                    }),
                                },
                            }),
                            BlockItem::Statement(Statement {
                                location: (6, 5).into(),
                                kind: StatementKind::Expression(Expression {
                                    location: (6, 5).into(),
                                    kind: Assignment {
                                        lvalue: Box::new(Expression {
                                            location: (6, 5).into(),
                                            kind: Variable("a".to_string()),
                                        }),
                                        rvalue: Box::new(Expression {
                                            location: (6, 9).into(),
                                            kind: Binary(
                                                BinaryOperator::Add,
                                                Box::new(Expression {
                                                    location: (6, 9).into(),
                                                    kind: Variable("b".to_string()),
                                                }),
                                                Box::new(Expression {
                                                    location: (6, 13).into(),
                                                    kind: Variable("c".to_string()),
                                                }),
                                            ),
                                        }),
                                    },
                                }),
                            }),
                        ],
                    }),
                }),
                BlockItem::Statement(Statement {
                    location: (8, 3).into(),
                    kind: Return(Expression {
                        location: (8, 10).into(),
                        kind: IntConstant("0".to_string(), Decimal),
                    }),
                }),
            ],
        });
        run_parse_block_test_case(BlockTestCase { src, expected })
    }

    #[rstest]
    #[case("simple_addition", "1+2")]
    #[case("simple_subtraction", "1-20")]
    #[case("simple_multiplication", "10*20")]
    #[case("simple_division", "2/4")]
    #[case("simple_remainder", "3%2")]
    #[case("multiplication_with_unary_operands", "~4*-3")]
    fn test_should_parse_arithmetic_expressions(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("arithmetic expressions", description, "expr/arithmetic", src);
    }

    #[rstest]
    #[case("addition_is_left_associative", "1+2+3")]
    #[case("subtraction_is_left_associative", "1-2-3")]
    #[case("multiplication_is_left_associative", "2*3*4")]
    #[case("division_is_left_associative", "10/2/3")]
    #[case("modulo_is_left_associative", "10 % 2 % 3")]
    fn test_should_parse_arithmetic_expressions_with_correct_associativity(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("arithmetic expressions with correct associativity", description, "expr/arithmetic", src);
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
        run_snapshot_test_for_parse_expression("arithmetic expressions with correct precedence", description, "expr/arithmetic", src);
    }

    #[rstest]
    #[case("unary_complement", "~10")]
    #[case("unary_negation", "-10")]
    #[case("double_complement", "~~10")]
    #[case("logical_unary_not", "!20")]
    #[case("double_logical_unary_not", "!!10")]
    fn test_should_parse_unary_expressions(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("unary expressions", description, "expr/unary", src);
    }

    #[rstest]
    #[case("unary_complement", "~10")]
    #[case("unary_negation", "-10")]
    #[case("double_complement", "~~10")]
    #[case("logical_unary_not", "!20")]
    #[case("double_logical_unary_not", "!!10")]
    fn test_should_parse_bitwise_expressions(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("bitwise operator expressions", description, "expr/bitwise", src);
    }

    #[rstest]
    #[case("bitwise_and_is_left_associative", "10 & 20 & 30")]
    #[case("bitwise_or_is_left_associative", "10 | 20 | 30")]
    #[case("bitwise_xor_is_left_associative", "10 ^ 20 ^ 30")]
    #[case("left_shift_is_left_associative", "1<<2<<3")]
    #[case("right_shift_is_left_associative", "200>>1>>1")]
    fn test_should_parse_bitwise_expressions_with_correct_associativity(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("bitwise operator expressions with correct associativity", description, "expr/bitwise", src);
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
        run_snapshot_test_for_parse_expression("bitwise operator expressions with correct precedence", description, "expr/bitwise", src);
    }

    #[rstest]
    #[case("logical_and", "10 && 20")]
    #[case("logical_or", "1 || 0")]
    #[case("logical_not", "!10")]
    #[case("logical_arith_chain", "(10 && 0) + (0 && 4) + (0 && 0)")]
    fn test_should_parse_logical_expressions(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("logical expressions", description, "expr/logical", src);
    }

    #[rstest]
    #[case("logical_or_is_left_associative", "1 || 2 || 3")]
    #[case("logical_and_is_left_associative", "1 && 2 && 3")]
    fn test_should_parse_logical_expressions_with_correct_associativity(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("logical expressions with correct associativity", description, "expr/logical", src);
    }

    #[rstest]
    #[case("logical_mixed_or_and", "1 || 2 && 3")]
    #[case("logical_and_with_parens", "(1 && 2) && 3")]
    #[case("logical_or_with_parens", "(1 || 2) || 3")]
    #[case("logical_mixed_and_or_parens", "1 && (2 || 3)")]
    fn test_should_parse_logical_expressions_with_correct_precedence(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("logical expressions with correct precedence", description, "expr/logical", src)
    }

    #[rstest]
    #[case("greater_than", "10 > 5")]
    #[case("less_than", "3 < 4")]
    #[case("greater_equal", "7 >= 7")]
    #[case("less_equal", "2 <= 3")]
    #[case("equal", "5 == 5")]
    #[case("not_equal", "5 != 6")]
    fn test_should_parse_relational_expressions(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("relational expressions", description, "expr/relational", src);
    }

    #[rstest]
    #[case("assoc_less_chain", "1 < 2 < 3")]             // (1 < 2) < 3
    #[case("assoc_greater_chain", "5 > 4 > 3")]          // (5 > 4) > 3
    #[case("assoc_le_ge_chain", "3 <= 3 >= 2")]          // (3 <= 3) >= 2
    #[case("assoc_logical_and", "1 && 1 && 0")]          // (1 && 1) && 0
    #[case("assoc_logical_or", "0 || 1 || 1")]           // (0 || 1) || 1
    fn test_should_parse_relational_expressions_with_correct_associativity(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("relational expressions with correct associativity", description, "expr/relational", src);
    }

    #[rstest]
    #[case("precedence_cmp_and", "1 < 2 && 3 > 2")]      // (<, >) evaluated before &&
    #[case("precedence_cmp_or", "1 == 1 || 0 != 1")]     // (==, !=) before ||
    #[case("precedence_and_or", "1 && 0 || 1")]          // && before ||
    fn test_should_parse_relational_expressions_with_correct_precedence(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("relational expressions with correct precedence", description, "expr/relational", src);
    }

    #[rstest]
    #[case("increment_pre", "++a")]
    #[case("increment_post", "a++")]
    #[case("decrement_pre", "--a")]
    #[case("decrement_post", "a--")]
    #[case("increment_pre_paren", "++(a)")]
    #[case("increment_post_paren", "(a)++")]
    #[case("decrement_pre_paren", "--(a)")]
    #[case("decrement_post_paren", "(a)--")]
    #[case("increment_pre_with_unary_not", "!++a")]
    #[case("increment_post_with_unary_not", "!a++")]
    #[case("decrement_pre_with_unary_not", "!--a")]
    #[case("decrement_post_with_unary_not", "!a--")]
    #[case("increment_pre_with_unary_neg", "-++a")]
    #[case("increment_post_with_unary_neg", "-a++")]
    #[case("decrement_pre_with_unary_neg", "-(--a)")] // yeah. ---a is illegal
    #[case("decrement_post_with_unary_neg", "-a--")]
    #[case("increment_pre_with_unary_complement", "~++a")]
    #[case("increment_post_with_unary_complement", "~a++")]
    #[case("decrement_pre_with_unary_complement", "~--a")]
    #[case("decrement_post_with_unary_complement", "~a--")]
    #[case("horrible_seq_1", "~a++ + -++a")]
    #[case("horrible_seq_2", "~-a++ - -~a--")]
    #[case("sadistic_seq_1", "~a++*-++b/~c--")] // don't be this person
    fn test_should_parse_increment_and_decrement_with_correct_precendence(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("increment and decrement with correct precedence", description, "expr/incrdecr", src);
    }

    fn run_snapshot_test_for_parse_expression(suite_description: &str, description: &str, snapshot_path: &str, src: &str) {
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
    #[case("(10 && 0) + (0 && 4) + (0 && 0)", "(((10 && 0)+(0 && 4))+(0 && 0))")]
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

    #[rstest]
    #[case("a = b = c", "a = (b = c)")]
    #[case("a = b + 10", "a = (b + 10)")]
    #[case("a = b = c = d + 10", "a = (b = (c = (d + 10)))")]
    fn test_assignment_operator_precedence_and_associativity(#[case] src1: &str, #[case] src2: &str) {
        run_expression_equivalence_test(src1, src2);
    }

    fn run_expression_equivalence_test(expr1_src: &str, expr2_src: &str) {
        let lex1 = Lexer::new(expr1_src);
        let mut parser1 = Parser::new(lex1);
        let actual1 = parser1.parse_expression();

        let lex2 = Lexer::new(expr2_src);
        let mut parser2 = Parser::new(lex2);
        let actual2 = parser2.parse_expression();

        let expr1 = actual1.unwrap();
        let expr2 = actual2.unwrap();
        assert!(is_equivalent_expression(&expr1, &expr2),
                "expected {expr1_src} to be equivalent to {expr2_src}, but parsed as {:#?} and {:#?}", expr1, expr2);
    }

    fn is_equivalent_expression(e1: &Expression, e2: &Expression) -> bool {
        match (&e1.kind, &e2.kind) {
            (Variable(v1), Variable(v2)) => v1 == v2,
            (IntConstant(c1, r1), IntConstant(c2, r2)) => c1 == c2 && r1 == r2,
            (Unary(uop1, subexp1), Unary(uop2, subexp2)) => uop1 == uop2
                && is_equivalent_expression(&*subexp1, &*subexp2),
            (Binary(binop1, op11, op12), Binary(binop2, op21, op22)) => binop1 == binop2
                && is_equivalent_expression(&*op11, &*op21)
                && is_equivalent_expression(&*op12, &*op22),
            (Assignment { lvalue: lv1, rvalue: rv1 },
                Assignment { lvalue: lv2, rvalue: rv2 }) =>
                is_equivalent_expression(&*lv1, &*lv2)
                    && is_equivalent_expression(&*rv1, &*rv2),
            _ => false,
        }
    }

    #[rstest]
    #[case("simple_return.c")]
    #[case("simple_return_with_expression.c")]
    #[case("simple_return_with_declaration.c")]
    #[case("function_body_with_subblocks.c")]
    #[case("multiple_functions.c")]
    fn test_parse_basic(#[case] src_file: &str) {
        run_snapshot_test_for_parsing("basic", src_file);
    }

    fn run_snapshot_test_for_parsing(suite_description: &str, src_file: &str) {
        let base_dir = file!();
        let src_path = Path::new(base_dir).parent().unwrap().join("snapshots").join("input").join(src_file);
        let source = fs::read_to_string(src_path.clone());
        assert!(source.is_ok(), "failed to read {:?}", src_path);

        let src = source.unwrap();
        let lexer = Lexer::new(&src);
        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("parsing failed");

        let (out_dir, snapshot_file) = output_path_parts(src_file);
        insta::with_settings!({
            sort_maps => true,
            prepend_module_to_snapshot => false,
            description => suite_description,
            snapshot_path => out_dir,
            info => &format!("{}", src_file),
        }, {
            insta::assert_yaml_snapshot!(snapshot_file, ast);
        });
    }

    fn output_path_parts(src_file: &str) -> (PathBuf, String) {
        let input_path = Path::new(src_file);
        let parent = input_path.parent().unwrap_or_else(|| Path::new(""));
        let stem = input_path.file_stem().expect("No file stem").to_string_lossy();
        let output_dir = Path::new("snapshots/output").join(parent);
        let output_file = format!("{}.ast", stem);
        (output_dir, output_file)
    }
}