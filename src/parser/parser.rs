//! Parser for the language: This implements parser for a subset of
//! features in C programming language. A simple Recursive Descent
//! Parsing is used. It is handwritten.

use derive_more::Add;
use serde::Serialize;
use thiserror::Error;

use crate::common::{Location, Radix};
use crate::lexer::{KeywordIdentifier, Lexer, LexerError, Token, TokenTag, TokenType};
use crate::lexer::TokenTag::{CloseParentheses, Comma, Keyword};
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
    CompoundAssignment(CompoundAssignmentType),
    TernaryThen, // actually a ternary operator
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompoundAssignmentType {
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

impl Into<BinaryOperator> for CompoundAssignmentType {
    fn into(self) -> BinaryOperator {
        match self {
            CompoundAssignmentType::Add => BinaryOperator::Add,
            CompoundAssignmentType::Subtract => BinaryOperator::Subtract,
            CompoundAssignmentType::Multiply => BinaryOperator::Multiply,
            CompoundAssignmentType::Divide => BinaryOperator::Divide,
            CompoundAssignmentType::Modulo => BinaryOperator::Modulo,
            CompoundAssignmentType::BitwiseAnd => BinaryOperator::BitwiseAnd,
            CompoundAssignmentType::BitwiseOr => BinaryOperator::BitwiseOr,
            CompoundAssignmentType::BitwiseXor => BinaryOperator::BitwiseXor,
            CompoundAssignmentType::LeftShift => BinaryOperator::LeftShift,
            CompoundAssignmentType::RightShift => BinaryOperator::RightShift,
        }
    }
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

            BinaryOperator::Assignment
            | BinaryOperator::TernaryThen
            | BinaryOperator::CompoundAssignment(_) => BinaryOperatorAssociativity::Right,
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

            BinaryOperator::TernaryThen => BinaryOperatorPrecedence(15),

            BinaryOperator::Assignment
            | BinaryOperator::CompoundAssignment(_) => BinaryOperatorPrecedence(10),
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
    Conditional {
        condition: Box<Expression>,
        then_expr: Box<Expression>,
        else_expr: Box<Expression>,
    },
    Assignment { lvalue: Box<Expression>, rvalue: Box<Expression>, op: Option<CompoundAssignmentType> },
    Increment { is_post: bool, e: Box<Expression> },
    Decrement { is_post: bool, e: Box<Expression> },
    FunctionCall { func_name: String, actual_params: Vec<Box<Expression>> },
}

impl ExpressionKind {
    pub fn is_lvalue_expression(&self) -> bool {
        match self {
            ExpressionKind::Variable(_) => true,
            ExpressionKind::Conditional { then_expr, else_expr, .. } =>
                then_expr.kind.is_lvalue_expression()
                    && else_expr.kind.is_lvalue_expression(),
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
pub enum ForInit {
    InitDecl(Box<Declaration>),
    InitExpr(Box<Expression>),
    Null,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StatementKind {
    Return(Expression),
    Expression(Expression),
    SubBlock(Block),
    If {
        condition: Box<Expression>,
        then_statement: Box<Statement>,
        else_statement: Option<Box<Statement>>,
    },
    Break(Option<String>),
    Continue(Option<String>),
    While {
        pre_condition: Box<Expression>,
        loop_body: Box<Statement>,
        loop_label: Option<String>,
    },
    DoWhile {
        loop_body: Box<Statement>,
        post_condition: Box<Expression>,
        loop_label: Option<String>,
    },
    For {
        init: ForInit,
        condition: Option<Box<Expression>>,
        post: Option<Box<Expression>>,
        loop_body: Box<Statement>,
        loop_label: Option<String>,
    },
    Goto { target: String },
    Null,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Statement {
    pub location: Location,
    pub labels: Vec<String>,
    pub kind: StatementKind,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclarationKind {
    VarDeclaration {
        identifier: Symbol,
        init_expression: Option<Expression>,
    },
    FunctionDeclaration (Function),
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Declaration {
    pub location: Location,
    pub kind: DeclarationKind,
}

impl Declaration {
    pub fn identifier(&self) -> Symbol {
        match &self.kind {
            DeclarationKind::FunctionDeclaration(ref f) => f.name.clone(),
            DeclarationKind::VarDeclaration { ref identifier, ..} => identifier.clone(),
        }
    }
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
pub struct FunctionParameter {
    pub loc: Location,
    pub param_type: Box<TypeExpression>,
    pub param_name: String,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Function {
    pub location: Location,
    pub name: Symbol,
    pub params: Vec<FunctionParameter>,
    pub body: Option<Block>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Program {
    /// declarations consist of all the variables
    /// and functions defined in the program unit
    pub declarations: Vec<Declaration>,
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

    #[error("{location:?}: void must be the only parameter and unnamed")]
    InvalidFunctionDeclarationVoidMustBeOnlyParam { location: Location }
}

pub struct Parser<'a> {
    token_provider: Lexer<'a>,
}

fn is_declaration(tok: &Token) -> bool {
    match &tok.token_type {
        TokenType::Keyword(KeywordIdentifier::TypeInt) => true,
        _ => false,
    }
}

impl<'a> Parser<'a> {
    pub fn new(token_provider: Lexer<'a>) -> Parser<'a> {
        Parser { token_provider }
    }

    /// parse parses the given source file and returns the
    /// Abstract Syntax Tree (AST).
    pub fn parse(&mut self) -> Result<Program, ParserError> {
        Ok(self.parse_program()?)
    }

    fn parse_program(&mut self) -> Result<Program, ParserError> {
        let mut decls: Vec<Declaration> = vec![];
        loop {
            if self.token_provider.peek().is_none() {
                break;
            }
            let decl = self.parse_declaration()?;
            decls.push(decl);
        }
        Ok(Program { declarations: decls })
    }

    fn parse_function(&mut self) -> Result<(Vec<FunctionParameter>, Option<Block>), ParserError> {
        let params = self.parse_function_parameters()?;
        self.expect_close_parentheses()?;

        let next_tok = self.token_provider.peek();
        match next_tok {
            None => Err(UnexpectedEnd(vec![TokenTag::OpenBrace, TokenTag::Semicolon])),
            Some(Err(e)) => Err(TokenizationError(e.clone())),
            Some(Ok(tok)) => {
                match &tok.token_type {
                    TokenType::Semicolon => {
                        self.token_provider.next();
                        Ok((params, None))
                    },
                    TokenType::OpenBrace => {
                        let func_body = self.parse_block()?;
                        Ok((params, Some(func_body)))
                    }
                    _ => Err(UnexpectedEnd(vec![TokenTag::OpenBrace, TokenTag::Semicolon])),
                }
            }
        }
    }

    fn parse_function_parameters(&mut self) -> Result<Vec<FunctionParameter>, ParserError> {
        let mut params = vec![];
        loop {
            let cur = self.token_provider.peek();
            match cur {
                None => return Err(UnexpectedEnd(vec![CloseParentheses, Keyword])),
                Some(Err(e)) => return Err(TokenizationError(e.clone())),
                Some(Ok(tok)) => {
                    let tok_loc = tok.location.clone();
                    match &tok.token_type {
                        TokenType::Keyword(KeywordIdentifier::TypeVoid) => {
                            if params.is_empty() {
                                self.token_provider.next();
                                return Ok(vec![]);
                            }
                            return Err(InvalidFunctionDeclarationVoidMustBeOnlyParam { location: tok_loc });
                        }
                        _ => {}
                    }
                }
            }

            let param_type = self.parse_type_expression()?;
            let param_ident = self.parse_identifier()?;
            params.push(FunctionParameter {
                loc: param_type.location.clone(),
                param_type: Box::new(param_type),
                param_name: param_ident.name,
            });

            // decide whether to stop or to continue parsing parameters
            let next_tok = self.token_provider.peek();
            match next_tok {
                Some(Ok(tok)) => {
                    let tok_loc = tok.location.clone();
                    match &tok.token_type {
                        TokenType::CloseParentheses => {
                            break;
                        },
                        TokenType::Comma => {
                            self.token_provider.next();
                            continue;
                        }
                        _ => {
                            return Err(UnexpectedToken {
                                location: tok_loc,
                                expected_token_tags: vec![Comma, CloseParentheses],
                            });
                        },
                    }
                }
                Some(Err(e)) => return Err(TokenizationError(e.clone())),
                None => return Err(UnexpectedEnd(vec![Comma, CloseParentheses])),
            }
        }
        Ok(params)
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
            Ok(tok) => {
                if is_declaration(tok) {
                    let decl = self.parse_declaration()?;
                    Ok(BlockItem::Declaration(decl))
                } else {
                    let stmt = self.parse_statement()?;
                    Ok(BlockItem::Statement(stmt))
                }
            }
            Err(e) => Err(TokenizationError(e.clone())),
        }
    }

    fn parse_declaration(&mut self) -> Result<Declaration, ParserError> {
        let ty_decl = self.parse_type_expression()?;
        let ident = self.parse_identifier()?;
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
                            kind: DeclarationKind::VarDeclaration {
                                identifier: ident,
                                init_expression: Some(init_expr),
                            },
                        })
                    }
                    TokenType::Semicolon => {
                        Ok(Declaration {
                            location: ty_decl.location.clone(),
                            kind: DeclarationKind::VarDeclaration {
                                identifier: ident,
                                init_expression: None,
                            },
                        })
                    }
                    TokenType::OpenParentheses => {
                        let (func_params, func_body) = self.parse_function()?;
                        Ok(Declaration {
                            location: ty_decl.location.clone(),
                            kind: DeclarationKind::FunctionDeclaration(Function {
                                location: ty_decl.location.clone(),
                                name: ident,
                                params: func_params,
                                body: func_body,
                            })
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

    fn parse_statement_labels(&mut self) -> Result<(Vec<String>, Option<Location>), ParserError> {
        let mut labels = vec![];
        let mut first_loc = None;
        loop {
            let tok1 = self.token_provider.peek_n(0);
            if tok1.is_none() {
                break;
            }
            match tok1.unwrap() {
                Ok(lbl_tok) => {
                    let tok_loc = lbl_tok.location.clone();
                    match &lbl_tok.token_type {
                        TokenType::Identifier(lbl) => {
                            let label = lbl.to_string();

                            let tok2 = self.token_provider.peek_n(1);
                            if tok2.is_none() {
                                break;
                            }
                            match tok2.unwrap() {
                                Err(e) => { return Err(TokenizationError(e.clone())) }
                                Ok(Token { token_type: TokenType::OperatorColon, .. }) => {
                                    // we have a valid label after performing all this
                                    // gymnastics. Hence, it is time to celebrate
                                    labels.push(label);
                                    first_loc.get_or_insert(tok_loc);
                                    self.token_provider.next();
                                    self.token_provider.next();
                                }
                                Ok(_) => { break; }
                            }
                        }
                        _ => { break; }
                    }
                }
                Err(e) => { return Err(TokenizationError(e.clone())) }
            }
        }
        Ok((labels, first_loc))
    }

    fn parse_statement(&mut self) -> Result<Statement, ParserError> {
        let (stmt_labels, first_loc) = self.parse_statement_labels()?;
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
                        Ok(Statement {
                            location: first_loc.unwrap_or(tok_loc),
                            labels: stmt_labels,
                            kind: StatementKind::SubBlock(sub_block),
                        })
                    }
                    TokenType::Semicolon => {
                        self.token_provider.next();
                        Ok(Statement {
                            location: first_loc.unwrap_or(tok_loc),
                            labels: stmt_labels,
                            kind: StatementKind::Null,
                        })
                    }
                    TokenType::Keyword(KeywordIdentifier::Return) => {
                        let unlabeled = self.parse_return_statement()?;
                        Ok(Statement {
                            location: first_loc.unwrap_or(unlabeled.location.clone()),
                            labels: stmt_labels,
                            kind: unlabeled.kind,
                        })
                    },
                    TokenType::Keyword(KeywordIdentifier::If) => {
                        let unlabeled = self.parse_if_statement()?;
                        Ok(Statement {
                            location: first_loc.unwrap_or(unlabeled.location.clone()),
                            labels: stmt_labels,
                            kind: unlabeled.kind,
                        })
                    },
                    TokenType::Keyword(KeywordIdentifier::Break) => {
                        self.token_provider.next();
                        self.get_token_with_tag(TokenTag::Semicolon)?;
                        Ok(Statement {
                            location: first_loc.unwrap_or(tok_loc.clone()),
                            labels: stmt_labels,
                            kind: StatementKind::Break(None),
                        })
                    },
                    TokenType::Keyword(KeywordIdentifier::Continue) => {
                        self.token_provider.next();
                        self.get_token_with_tag(TokenTag::Semicolon)?;
                        Ok(Statement {
                            location: first_loc.unwrap_or(tok_loc.clone()),
                            labels: stmt_labels,
                            kind: StatementKind::Continue(None),
                        })
                    },
                    TokenType::Keyword(KeywordIdentifier::For) => {
                        let unlabeled = self.parse_for_loop_statement()?;
                        Ok(Statement {
                            location: first_loc.unwrap_or(unlabeled.location.clone()),
                            labels: stmt_labels,
                            kind: unlabeled.kind,
                        })
                    },
                    TokenType::Keyword(KeywordIdentifier::While) => {
                        self.token_provider.next();
                        self.expect_token_with_tag(TokenTag::OpenParentheses)?;
                        let precondition_expr = self.parse_expression()?;
                        self.expect_token_with_tag(TokenTag::CloseParentheses)?;
                        let loop_body = self.parse_statement()?;
                        Ok(Statement {
                            location: first_loc.unwrap_or(tok_loc.clone()),
                            labels: stmt_labels,
                            kind: StatementKind::While {
                                pre_condition: Box::new(precondition_expr),
                                loop_body: Box::new(loop_body),
                                loop_label: None,
                            },
                        })
                    },
                    TokenType::Keyword(KeywordIdentifier::Do) => {
                        self.token_provider.next();
                        let loop_body = self.parse_statement()?;
                        self.expect_keyword(KeywordIdentifier::While)?;
                        self.expect_token_with_tag(TokenTag::OpenParentheses)?;
                        let post_condition = self.parse_expression()?;
                        self.expect_token_with_tag(TokenTag::CloseParentheses)?;
                        self.expect_semicolon()?;
                        Ok(Statement {
                            location: first_loc.unwrap_or(tok_loc.clone()),
                            labels: stmt_labels,
                            kind: StatementKind::DoWhile {
                                loop_body: Box::new(loop_body),
                                post_condition: Box::new(post_condition),
                                loop_label: None,
                            },
                        })
                    },
                    TokenType::Keyword(KeywordIdentifier::Goto) => {
                        self.token_provider.next();
                        let target_label = self.get_token_with_tag(TokenTag::Identifier)?;
                        let res = match target_label.token_type {
                            TokenType::Identifier(target_lbl) => {
                                Ok(Statement {
                                    location: first_loc.unwrap_or(tok_loc),
                                    labels: stmt_labels,
                                    kind: StatementKind::Goto {
                                        target: target_lbl.to_string(),
                                    },
                                })
                            }
                            _ => unreachable!("non-identifier should have errored")
                        };
                        self.expect_semicolon()?;
                        res
                    }
                    _ => {
                        let expr = self.parse_expression()?;
                        self.expect_semicolon()?;
                        Ok(Statement {
                            location: first_loc.unwrap_or(expr.location.clone()),
                            labels: stmt_labels,
                            kind: StatementKind::Expression(expr),
                        })
                    },
                }
            }
            Err(e) => Err(TokenizationError(e.clone())),
        }
    }

    fn parse_for_loop_statement(&mut self) -> Result<Statement, ParserError> {
        let first_loc = self.expect_keyword(KeywordIdentifier::For)?;
        self.expect_token_with_tag(TokenTag::OpenParentheses)?;
        let loop_init = self.parse_for_loop_init()?;

        let loop_cond = self.parse_for_loop_maybe_parse_expression()?;
        self.expect_token_with_tag(TokenTag::Semicolon)?;

        let loop_post = self.parse_for_loop_maybe_parse_expression()?;
        self.expect_token_with_tag(TokenTag::CloseParentheses)?;

        let loop_body = self.parse_statement()?;
        Ok(Statement {
            location: first_loc,
            labels: vec![],
            kind: StatementKind::For {
                init: loop_init,
                condition: loop_cond.map(|e| Box::new(e)),
                post: loop_post.map(|e| Box::new(e)),
                loop_body: Box::new(loop_body),
                loop_label: None,
            },
        })
    }

    fn parse_for_loop_init(&mut self) -> Result<ForInit, ParserError> {
        let tok = self.token_provider.peek();
        if tok.is_none() {
            return Err(UnexpectedEnd(vec![TokenTag::Semicolon, TokenTag::CloseParentheses]));
        }
        let tok_unwrapped = tok.unwrap();
        let init_expr = match &tok_unwrapped {
            Ok(Token { token_type: TokenType::Semicolon, .. }) => {
                self.expect_semicolon()?;
                Ok(ForInit::Null)
            },
            Ok(tok) => {
                if is_declaration(tok) {
                    let decl = self.parse_declaration()?;
                    Ok(ForInit::InitDecl(Box::new(decl)))
                } else {
                    let expr = self.parse_expression()?;
                    self.expect_semicolon()?;
                    Ok(ForInit::InitExpr(Box::new(expr)))
                }
            },
            Err(e) => Err(TokenizationError(e.clone())),
        };
        if init_expr.is_err() {
            return init_expr;
        }
        init_expr
    }

    fn parse_for_loop_maybe_parse_expression(&mut self) -> Result<Option<Expression>, ParserError> {
        let tok = self.token_provider.peek();
        match &tok {
            None => Err(UnexpectedEnd(vec![TokenTag::Semicolon, TokenTag::CloseParentheses])),
            Some(Ok(Token { token_type: TokenType::Semicolon, .. }))
            | Some(Ok(Token { token_type: TokenType::CloseParentheses, .. })) => Ok(None),
            Some(Ok(_)) => {
                let expr = self.parse_expression()?;
                Ok(Some(expr))
            },
            Some(Err(e)) => Err(TokenizationError(e.clone())),
        }
    }

    fn parse_return_statement(&mut self) -> Result<Statement, ParserError> {
        let kloc = self.expect_keyword(KeywordIdentifier::Return)?;
        let return_code_expr = self.parse_expression()?;
        self.expect_semicolon()?;
        Ok(Statement { location: kloc, labels: vec![], kind: StatementKind::Return(return_code_expr) })
    }

    fn parse_if_statement(&mut self) -> Result<Statement, ParserError> {
        let kloc = self.expect_keyword(KeywordIdentifier::If)?;
        
        self.expect_open_parentheses()?;
        let cond_expr = self.parse_expression()?;
        self.expect_close_parentheses()?;

        let then_stmt = self.parse_statement()?;
        
        // peek and see if it is else.
        let next_tok = self.token_provider.peek();
        match next_tok {
            Some(Err(e)) => Err(TokenizationError(e.clone())),
            Some(Ok(Token { token_type: TokenType::Keyword(KeywordIdentifier::Else), .. })) => {
                self.token_provider.next(); // consume else
                let else_stmt = self.parse_statement()?;
                Ok(Statement {
                    location: kloc.clone(),
                    labels: vec![],
                    kind: StatementKind::If {
                        condition: Box::new(cond_expr),
                        then_statement: Box::new(then_stmt),
                        else_statement: Some(Box::new(else_stmt)),
                    },
                })
            }
            _ => Ok(Statement {
                location: kloc.clone(),
                labels: vec![],
                kind: StatementKind::If {
                    condition: Box::new(cond_expr),
                    then_statement: Box::new(then_stmt),
                    else_statement: None,
                },
            }),
        }
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
                    let left_loc = left.location.clone();
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
                    if binary_op == BinaryOperator::TernaryThen {
                        let then_expr = self.parse_expression()?;
                        self.expect_token_with_tag(TokenTag::OperatorColon)?;
                        let else_expr = self.parse_expression_with_precedence(next_min_precedence)?;
                        left = Expression {
                            location: left_loc,
                            kind: ExpressionKind::Conditional {
                                condition: Box::new(left),
                                then_expr: Box::new(then_expr),
                                else_expr: Box::new(else_expr),
                            },
                        };
                        continue
                    }
                    let rhs = self.parse_expression_with_precedence(next_min_precedence)?;
                    let expr_kind = match binary_op {
                        BinaryOperator::Assignment => ExpressionKind::Assignment {
                            lvalue: Box::new(left),
                            rvalue: Box::new(rhs),
                            op: None,
                        },
                        BinaryOperator::CompoundAssignment(cat) => ExpressionKind::Assignment {
                            lvalue: Box::new(left),
                            rvalue: Box::new(rhs),
                            op: Some(cat),
                        },
                        op => ExpressionKind::Binary(op, Box::new(left), Box::new(rhs))
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
                    TokenType::OperatorTernaryThen => Ok(BinaryOperator::TernaryThen),
                    TokenType::OperatorAssignment => Ok(BinaryOperator::Assignment),
                    TokenType::OperatorCompoundAssignmentAdd => Ok(BinaryOperator::CompoundAssignment(CompoundAssignmentType::Add)),
                    TokenType::OperatorCompoundAssignmentSubtract => Ok(BinaryOperator::CompoundAssignment(CompoundAssignmentType::Subtract)),
                    TokenType::OperatorCompoundAssignmentMultiply => Ok(BinaryOperator::CompoundAssignment(CompoundAssignmentType::Multiply)),
                    TokenType::OperatorCompoundAssignmentDivide => Ok(BinaryOperator::CompoundAssignment(CompoundAssignmentType::Divide)),
                    TokenType::OperatorCompoundAssignmentModulo => Ok(BinaryOperator::CompoundAssignment(CompoundAssignmentType::Modulo)),
                    TokenType::OperatorCompoundAssignmentLeftShift => Ok(BinaryOperator::CompoundAssignment(CompoundAssignmentType::LeftShift)),
                    TokenType::OperatorCompoundAssignmentRightShift => Ok(BinaryOperator::CompoundAssignment(CompoundAssignmentType::RightShift)),
                    TokenType::OperatorCompoundAssignmentBitwiseAnd => Ok(BinaryOperator::CompoundAssignment(CompoundAssignmentType::BitwiseAnd)),
                    TokenType::OperatorCompoundAssignmentBitwiseOr => Ok(BinaryOperator::CompoundAssignment(CompoundAssignmentType::BitwiseOr)),
                    TokenType::OperatorCompoundAssignmentBitwiseXor => Ok(BinaryOperator::CompoundAssignment(CompoundAssignmentType::BitwiseXor)),
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
    use crate::parser::{BinaryOperator, Block, BlockItem, CompoundAssignmentType, Declaration, DeclarationKind, Expression, ForInit, Function, FunctionParameter, Parser, ParserError, PrimitiveKind, Program, Statement, StatementKind, Symbol, TypeExpression, TypeExpressionKind, UnaryOperator};
    use crate::parser::DeclarationKind::FunctionDeclaration;
    use crate::parser::ExpressionKind::*;
    use crate::parser::PrimitiveKind::Integer;
    use crate::parser::StatementKind::*;
    use crate::parser::TypeExpressionKind::Primitive;

    #[test]
    fn test_parse_program_with_tabs() {
        let src = "int	main	(	void)	{	return	0	;	}";
        let lexer = Lexer::new(src);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert_eq!(Ok(Program {
            declarations: vec![
                Declaration {
                    location: (1,1).into(),
                    kind: FunctionDeclaration(Function {
                        location: (1,1).into(),
                        name: Symbol {
                            name: "main".to_string(),
                            location: (1,8).into(),
                        },
                        params: vec![],
                        body: Some(Block {
                            start_loc: (1,32).into(),
                            end_loc: (1,64).into(),
                            items: vec![
                                BlockItem::Statement(
                                    Statement {
                                        location: (1,40).into(),
                                        labels: vec![],
                                        kind: Return(Expression {
                                            location: (1,48).into(),
                                            kind: IntConstant("0".to_string(), Decimal),
                                        }),
                                    },
                                ),
                            ],
                        }),
                    }),
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
        assert_eq!(Ok(Program {
            declarations: vec![
                Declaration {
                    location: (1,1).into(),
                    kind: FunctionDeclaration(Function {
                        location: (1, 1).into(),
                        name: Symbol { name: "main".to_string(), location: (1, 5).into() },
                        params: vec![],
                        body: Some(Block {
                            start_loc: (1, 16).into(),
                            end_loc: (3, 1).into(),
                            items: vec![
                                BlockItem::Statement(Statement {
                                    location: (2, 5).into(),
                                    labels: vec![],
                                    kind: Return(Expression {
                                        location: (2, 12).into(),
                                        kind: IntConstant("2".to_string(), Decimal),
                                    }),
                                })
                            ],
                        }),
                    }),
                },
                Declaration {
                    location: (5,1).into(),
                    kind: FunctionDeclaration(Function {
                        location: (5,1).into(),
                        name: Symbol { name: "foo".to_string(), location: (5,5).into() },
                        params: vec![],
                        body: Some(Block {
                            start_loc: (5,15).into(),
                            end_loc: (7,1).into(),
                            items: vec![
                                BlockItem::Statement(Statement {
                                    location: (6,5).into(),
                                    labels: vec![],
                                    kind: Return(Expression {
                                        location: (6,12).into(),
                                        kind: IntConstant("3".to_string(), Decimal),
                                    }),
                                })
                            ],
                        }),
                    },),
                },
            ],
        }), parsed)
    }
    
    #[test]
    fn test_parse_function_with_multiple_args() {
        let src = indoc! {r#"
        int add(int a, int b) {
            return a + b;
        }
        "#};
        let lexer = Lexer::new(src);
        let mut parser = Parser::new(lexer);
        let actual = parser.parse();
        let expected = Ok(Program {
            declarations: vec![
                Declaration {
                    location: (1,1).into(),
                    kind: FunctionDeclaration(Function {
                        location: (1,1).into(),
                        name: Symbol {
                            name: "add".to_string(),
                            location: (1,5).into(),
                        },
                        params: vec![
                            FunctionParameter {
                                loc: (1,9).into(),
                                param_type: Box::new(TypeExpression {
                                    location: (1,9).into(),
                                    kind: Primitive(Integer),
                                }),
                                param_name: "a".to_string(),
                            },
                            FunctionParameter {
                                loc: (1,16).into(),
                                param_type: Box::new(TypeExpression {
                                    location: (1,16).into(),
                                    kind: Primitive(Integer),
                                }),
                                param_name: "b".to_string(),
                            }
                        ],
                        body: Some(Block {
                            start_loc: (1,23).into(),
                            end_loc: (3,1).into(),
                            items: vec![
                                BlockItem::Statement(Statement {
                                    location: (2,5).into(),
                                    labels: vec![],
                                    kind: Return(Expression {
                                        location: (2,12).into(),
                                        kind: Binary(
                                            BinaryOperator::Add,
                                            Box::new(Expression {
                                                location: (2,12).into(),
                                                kind: Variable("a".to_string()),
                                            }),
                                            Box::new(Expression {
                                                location: (2,16).into(),
                                                kind: Variable("b".to_string()),
                                            }),
                                        ),
                                    }),
                                }),
                            ],
                        }),
                    }),
                },
            ],
        });
        assert_eq!(expected, actual, "expected:\n{:#?}\nactual:\n{:#?}\n", expected, actual);
    }

    #[test]
    fn test_parse_function_declarations() {
        let src = indoc! {r#"
        int add(int a, int b);
        int do_foo(void);
        "#};
        let lexer = Lexer::new(src);
        let mut parser = Parser::new(lexer);
        let actual = parser.parse();
        let expected = Ok(Program {
            declarations: vec![
                Declaration {
                    location: (1,1).into(),
                    kind: FunctionDeclaration(Function {
                        location: (1,1).into(),
                        name: Symbol { location: (1,5).into(), name: "add".to_string() },
                        params: vec![
                            FunctionParameter {
                                loc: (1,9).into(),
                                param_name: "a".to_string(),
                                param_type: Box::new(TypeExpression {
                                    location: (1,9).into(),
                                    kind: Primitive(Integer),
                                }),
                            },
                            FunctionParameter {
                                loc: (1,16).into(),
                                param_name: "b".to_string(),
                                param_type: Box::new(TypeExpression {
                                    location: (1,16).into(),
                                    kind: Primitive(Integer),
                                }),
                            }
                        ],
                        body: None,
                    }),
                },
                Declaration {
                    location: (2,1).into(),
                    kind: FunctionDeclaration(Function {
                        location: (2,1).into(),
                        name: Symbol { location: (2,5).into(), name: "do_foo".to_string() },
                        params: vec![],
                        body: None,
                    }),
                }
            ],
        });
        assert_eq!(expected, actual, "expected:\n{:#?}\nactual:\n{:#?}\n", expected, actual);
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
        let expected = Ok(Program {
            declarations: vec![
                Declaration {
                    location: (1,1).into(),
                    kind: FunctionDeclaration(Function {
                        location: (1,1).into(),
                        name: Symbol { name: "main".to_string(), location: (1,5).into() },
                        params: vec![],
                        body: Some(Block {
                            start_loc: (1,16).into(),
                            end_loc: (3,1).into(),
                            items: vec![
                                BlockItem::Statement(
                                    Statement {
                                        location: (2,5).into(),
                                        labels: vec![],
                                        kind: Return(Expression {
                                            location: (2,12).into(),
                                            kind: Binary(
                                                BinaryOperator::Add,
                                                Box::new(Expression { location: (2,12).into(), kind: IntConstant("1".to_string(), Decimal) }),
                                                Box::new(Expression { location: (2,16).into(), kind: IntConstant("2".to_string(), Decimal) }),
                                            ),
                                        }),
                                    },
                                ),
                            ],
                        }),
                    }),
                }
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
        assert_eq!(test_case.expected, actual,
                   "expected:{:#?}\nactual:{:#?}\n", test_case.expected, actual);
    }

    #[test]
    fn test_parse_statement_empty() {
        let src = ";";
        let expected = Ok(Statement { location: (1, 1).into(), labels: vec![], kind: Null });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_return() {
        let src = "return 10;";
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec![],
            kind: Return(Expression {
                location: (1, 8).into(),
                kind: IntConstant("10".to_string(), Decimal),
            }),
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }
    
    #[test]
    fn test_parse_statement_increment() {
        let src = "a++;";
        let expected = Ok(Statement {
            location: (1,1).into(),
            labels: vec![],
            kind: StatementKind::Expression(Expression {
                location: (1,1).into(),
                kind: Increment {
                    is_post: true,
                    e: Box::new(Expression {
                        location: (1,1).into(),
                        kind: Variable("a".into()),
                    }),
                },
            }),
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_decrement() {
        let src = "a--;";
        let expected = Ok(Statement {
            location: (1,1).into(),
            labels: vec![],
            kind: StatementKind::Expression(Expression {
                location: (1,1).into(),
                kind: Decrement {
                    is_post: true,
                    e: Box::new(Expression {
                        location: (1,1).into(),
                        kind: Variable("a".into()),
                    }),
                },
            }),
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_simple_assignment() {
        let src = "a = 10;";
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec![],
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
                    op: None,
                },
            }),
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }
    
    #[test]
    fn test_parse_statement_simple_if_block() {
        let src = "if (a) b;";
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec![],
            kind: If {
                condition: Box::new(Expression {
                    location: (1,5).into(),
                    kind: Variable("a".to_string()),
                }),
                then_statement: Box::new(Statement {
                    location: (1,8).into(),
                    labels: vec![],
                    kind: StatementKind::Expression(Expression {
                        location: (1,8).into(),
                        kind: Variable("b".to_string()),
                    }),
                }),
                else_statement: None,
            },
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_simple_if_block_with_else() {
        let src = "if (a) b; else c;";
        let expected = Ok(Statement {
            labels: vec![],
            location: (1, 1).into(),
            kind: If {
                condition: Box::new(Expression {
                    location: (1, 5).into(),
                    kind: Variable("a".to_string()),
                }),
                then_statement: Box::new(Statement {
                    location: (1, 8).into(),
                    labels: vec![],
                    kind: Expression(Expression {
                        location: (1, 8).into(),
                        kind: Variable("b".to_string()),
                    }),
                }),
                else_statement: Some(Box::new(Statement {
                    location: (1, 16).into(),
                    labels: vec![],
                    kind: Expression(Expression {
                        location: (1, 16).into(),
                        kind: Variable("c".to_string()),
                    }),
                })),
            },
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }


    #[test]
    fn test_parse_statement_if_with_multiple_statements_in_else() {
        let src = "if (a) { return b; } else { b += 10; return b; }";
        let expected = Ok(Statement {
            labels: vec![],
            location: (1, 1).into(),
            kind: If {
                condition: Box::new(Expression {
                    location: (1, 5).into(),
                    kind: Variable("a".to_string()),
                }),
                then_statement: Box::new(Statement {
                    labels: vec![],
                    location: (1, 8).into(),
                    kind: SubBlock(Block {
                        start_loc: (1, 8).into(),
                        end_loc: (1, 20).into(),
                        items: vec![BlockItem::Statement(Statement {
                            labels: vec![],
                            location: (1, 10).into(),
                            kind: Return(Expression {
                                location: (1, 17).into(),
                                kind: Variable("b".to_string()),
                            }),
                        })],
                    }),
                }),
                else_statement: Some(Box::new(Statement {
                    labels: vec![],
                    location: (1, 27).into(),
                    kind: SubBlock(Block {
                        start_loc: (1, 27).into(),
                        end_loc: (1, 48).into(),
                        items: vec![
                            BlockItem::Statement(Statement {
                                labels: vec![],
                                location: (1, 29).into(),
                                kind: Expression(Expression {
                                    location: (1, 29).into(),
                                    kind: Assignment {
                                        lvalue: Box::new(Expression {
                                            location: (1, 29).into(),
                                            kind: Variable("b".to_string()),
                                        }),
                                        rvalue: Box::new(Expression {
                                            location: (1, 34).into(),
                                            kind: IntConstant("10".to_string(), Decimal),
                                        }),
                                        op: Some(CompoundAssignmentType::Add),
                                    },
                                }),
                            }),
                            BlockItem::Statement(Statement {
                                labels: vec![],
                                location: (1, 38).into(),
                                kind: Return(Expression {
                                    location: (1, 45).into(),
                                    kind: Variable("b".to_string()),
                                }),
                            }),
                        ],
                    }),
                })),
            },
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }


    #[test]
    fn test_parse_statement_if_else_ladder() {
        let src = indoc! {r#"
        if (a < 10)
            return 0;
        else if (a < 20)
            return 1;
        else
            return 2;
        "#};
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec![],
            kind: If {
                condition: Box::new(Expression {
                    location: (1, 5).into(),
                    kind: Binary(
                        BinaryOperator::LessThan,
                        Box::new(Expression {
                            location: (1, 5).into(),
                            kind: Variable("a".to_string()),
                        }),
                        Box::new(Expression {
                            location: (1, 9).into(),
                            kind: IntConstant("10".to_string(), Decimal),
                        }),
                    ),
                }),
                then_statement: Box::new(Statement {
                    location: (2, 5).into(),
                    labels: vec![],
                    kind: Return(Expression {
                        location: (2, 12).into(),
                        kind: IntConstant("0".to_string(), Decimal),
                    }),
                }),
                else_statement: Some(Box::new(Statement {
                    location: (3, 6).into(),
                    labels: vec![],
                    kind: If {
                        condition: Box::new(Expression {
                            location: (3, 10).into(),
                            kind: Binary(
                                BinaryOperator::LessThan,
                                Box::new(Expression {
                                    location: (3, 10).into(),
                                    kind: Variable("a".to_string()),
                                }),
                                Box::new(Expression {
                                    location: (3, 14).into(),
                                    kind: IntConstant("20".to_string(), Decimal),
                                }),
                            ),
                        }),
                        then_statement: Box::new(Statement {
                            location: (4, 5).into(),
                            labels: vec![],
                            kind: Return(Expression {
                                location: (4, 12).into(),
                                kind: IntConstant("1".to_string(), Decimal),
                            }),
                        }),
                        else_statement: Some(Box::new(Statement {
                            location: (6, 5).into(),
                            labels: vec![],
                            kind: Return(Expression {
                                location: (6, 12).into(),
                                kind: IntConstant("2".to_string(), Decimal),
                            }),
                        })),
                    },
                })),
            },
        });

        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_goto() {
        let src = "goto x;";
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec![],
            kind: Goto { target: "x".to_string() },
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_dangling_else_binds_to_nearest_if() {
        let src = indoc! {r#"
        if (a)
            if (b)
                return 1;
            else
                return 2;
        "#};
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec![],
            kind: If {
                condition: Box::new(Expression {
                    location: (1, 5).into(),
                    kind: Variable("a".to_string()),
                }),
                then_statement: Box::new(Statement {
                    labels: vec![],
                    location: (2, 5).into(),
                    kind: If {
                        condition: Box::new(Expression {
                            location: (2, 9).into(),
                            kind: Variable("b".to_string()),
                        }),
                        then_statement: Box::new(Statement {
                            labels: vec![],
                            location: (3, 9).into(),
                            kind: Return(Expression {
                                location: (3, 16).into(),
                                kind: IntConstant("1".to_string(), Decimal),
                            }),
                        }),
                        else_statement: Some(Box::new(Statement {
                            location: (5, 9).into(),
                            labels: vec![],
                            kind: Return(Expression {
                                location: (5, 16).into(),
                                kind: IntConstant("2".to_string(), Decimal),
                            }),
                        })),
                    },
                }),
                else_statement: None,
            },
        });

        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_labeled_simple() {
        let src = "x: a = 10;";
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec!["x".to_string()],
            kind: StatementKind::Expression(Expression {
                location: (1, 4).into(),
                kind: Assignment {
                    lvalue: Box::new(Expression {
                        location: (1, 4).into(),
                        kind: Variable("a".to_string()),
                    }),
                    rvalue: Box::new(Expression {
                        location: (1, 8).into(),
                        kind: IntConstant("10".to_string(), Decimal),
                    }),
                    op: None,
                },
            }),
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_multi_labeled_simple() {
        let src = "x: y: z: a = 10;";
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec![
                "x".to_string(),
                "y".to_string(),
                "z".to_string(),
            ],
            kind: StatementKind::Expression(Expression {
                location: (1, 10).into(),
                kind: Assignment {
                    lvalue: Box::new(Expression {
                        location: (1,10).into(),
                        kind: Variable("a".to_string()),
                    }),
                    rvalue: Box::new(Expression {
                        location: (1, 14).into(),
                        kind: IntConstant("10".to_string(), Decimal),
                    }),
                    op: None,
                },
            }),
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_labeled_if() {
        let src = indoc!{r#"
        lbl1:
            if (a) b; else c;
        "#};
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec!["lbl1".to_string()],
            kind: If {
                condition: Box::new(Expression {
                    location: (2, 9).into(),
                    kind: Variable("a".to_string()),
                }),
                then_statement: Box::new(Statement {
                    location: (2, 12).into(),
                    labels: vec![],
                    kind: StatementKind::Expression(Expression {
                        location: (2, 12).into(),
                        kind: Variable("b".to_string()),
                    }),
                }),
                else_statement: Some(Box::new(Statement {
                    location: (2, 20).into(),
                    labels: vec![],
                    kind: StatementKind::Expression(Expression {
                        location: (2, 20).into(),
                        kind: Variable("c".to_string()),
                    }),
                })),
            },
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_for_loop_with_init_declaration() {
        let src = indoc!{r#"
        for (int i = 0; i < 10; i++)
            x = x + i;
        "#};
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec![],
            kind: For {
                init: ForInit::InitDecl(Box::new(Declaration {
                    location: (1, 6).into(),
                    kind: DeclarationKind::VarDeclaration {
                        identifier: Symbol {
                            name: "i".to_string(),
                            location: (1, 10).into(),
                        },
                        init_expression: Some(Expression {
                            location: (1, 14).into(),
                            kind: IntConstant("0".to_string(), Decimal),
                        }),
                    },
                })),
                condition: Some(Box::new(Expression {
                    location: (1, 17).into(),
                    kind: Binary(
                        BinaryOperator::LessThan,
                        Box::new(Expression {
                            location: (1, 17).into(),
                            kind: Variable("i".to_string()),
                        }),
                        Box::new(Expression {
                            location: (1, 21).into(),
                            kind: IntConstant("10".to_string(), Decimal),
                        }),
                    ),
                })),
                post: Some(Box::new(Expression {
                    location: (1, 25).into(),
                    kind: Increment {
                        is_post: true,
                        e: Box::new(Expression {
                            location: (1, 25).into(),
                            kind: Variable("i".to_string()),
                        }),
                    },
                })),
                loop_body: Box::new(Statement {
                    location: (2, 5).into(),
                    labels: vec![],
                    kind: StatementKind::Expression(Expression {
                        location: (2, 5).into(),
                        kind: Assignment {
                            lvalue: Box::new(Expression {
                                location: (2, 5).into(),
                                kind: Variable("x".to_string()),
                            }),
                            rvalue: Box::new(Expression {
                                location: (2, 9).into(),
                                kind: Binary(
                                    BinaryOperator::Add,
                                    Box::new(Expression {
                                        location: (2, 9).into(),
                                        kind: Variable("x".to_string()),
                                    }),
                                    Box::new(Expression {
                                        location: (2, 13).into(),
                                        kind: Variable("i".to_string()),
                                    }),
                                ),
                            }),
                            op: None,
                        },
                    }),
                }),
                loop_label: None,
            },
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_for_loop_init_statement() {
        let src = indoc! {r#"
        for (i = 0; i < 10; i++)
            x += i;
        "#};
        let expected = Ok(Statement {
            location: (1,1).into(),
            labels: vec![],
            kind: For {
                init: ForInit::InitExpr(Box::new(
                    Expression {
                        location: (1,6).into(),
                        kind: Assignment {
                            lvalue: Box::new(Expression {
                                location: (1,6).into(),
                                kind: Variable("i".to_string()),
                            }),
                            rvalue: Box::new(Expression {
                                location: (1,10).into(),
                                kind: IntConstant("0".to_string(), Decimal),
                            }),
                            op: None,
                        }
                    }
                )),
                condition: Some(Box::new(Expression {
                    location: (1,13).into(),
                    kind: Binary(
                        BinaryOperator::LessThan,
                        Box::new(Expression {
                            location: (1,13).into(),
                            kind: Variable("i".to_string())
                        }),
                        Box::new(Expression {
                            location: (1,17).into(),
                            kind: IntConstant("10".to_string(), Decimal)
                        }),
                    ),
                })),
                post: Some(Box::new(Expression {
                    location: (1,21).into(),
                    kind: Increment {
                        is_post: true,
                        e: Box::new(Expression {
                            location: (1,21).into(),
                            kind: Variable("i".to_string())
                        }),
                    },
                })),
                loop_body: Box::new(Statement {
                    location: (2,5).into(),
                    labels: vec![],
                    kind: StatementKind::Expression(Expression {
                        location: (2,5).into(),
                        kind: Assignment {
                            lvalue: Box::new(Expression {
                                location: (2,5).into(),
                                kind: Variable("x".to_string()),
                            }),
                            rvalue: Box::new(Expression {
                                location: (2,10).into(),
                                kind: Variable("i".to_string()),
                            }),
                            op: Some(CompoundAssignmentType::Add),
                        },
                    }),
                }),
                loop_label: None,
            },
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }


    #[test]
    fn test_parse_statement_for_loop_labeled() {
        let src = "x: for (i = 0; i < 10; i++) x += i;";
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec!["x".into()],
            kind: For {
                init: ForInit::InitExpr(Box::new(Expression {
                    location: (1, 9).into(),
                    kind: Assignment {
                        lvalue: Box::new(Expression {
                            location: (1, 9).into(),
                            kind: Variable("i".into()),
                        }),
                        rvalue: Box::new(Expression {
                            location: (1, 13).into(),
                            kind: IntConstant("0".into(), Decimal),
                        }),
                        op: None,
                    },
                })),
                condition: Some(Box::new(Expression {
                    location: (1, 16).into(),
                    kind: Binary(
                        BinaryOperator::LessThan,
                        Box::new(Expression {
                            location: (1, 16).into(),
                            kind: Variable("i".into()),
                        }),
                        Box::new(Expression {
                            location: (1, 20).into(),
                            kind: IntConstant("10".into(), Decimal),
                        }),
                    ),
                })),
                post: Some(Box::new(Expression {
                    location: (1, 24).into(),
                    kind: Increment {
                        is_post: true,
                        e: Box::new(Expression {
                            location: (1, 24).into(),
                            kind: Variable("i".into()),
                        }),
                    },
                })),
                loop_body: Box::new(Statement {
                    location: (1, 29).into(),
                    labels: vec![],
                    kind: StatementKind::Expression(Expression {
                        location: (1, 29).into(),
                        kind: Assignment {
                            lvalue: Box::new(Expression {
                                location: (1, 29).into(),
                                kind: Variable("x".into()),
                            }),
                            rvalue: Box::new(Expression {
                                location: (1, 34).into(),
                                kind: Variable("i".into()),
                            }),
                            op: Some(CompoundAssignmentType::Add),
                        },
                    }),
                }),
                loop_label: None,
            },
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_for_loop_null_headers() {
        let src = indoc!{r#"
        for(;;);
        "#};
        let expected = Ok(Statement {
            location: (1,1).into(),
            labels: vec![],
            kind: For {
                init: ForInit::Null,
                condition: None,
                post: None,
                loop_body: Box::new(Statement {
                    location: (1,8).into(),
                    labels: vec![],
                    kind: Null,
                }),
                loop_label: None,
            },
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_while_loop() {
        let src = "while (x < 10) x++;";
        let expected = Ok(Statement {
            location: (1, 1).into(),
            labels: vec![],
            kind: While {
                pre_condition: Box::new(Expression {
                    location: (1, 8).into(),
                    kind: Binary(
                        BinaryOperator::LessThan,
                        Box::new(Expression {
                            location: (1, 8).into(),
                            kind: Variable("x".into()),
                        }),
                        Box::new(Expression {
                            location: (1, 12).into(),
                            kind: IntConstant("10".into(), Decimal),
                        }),
                    ),
                }),
                loop_body: Box::new(Statement {
                    location: (1, 16).into(),
                    labels: vec![],
                    kind: StatementKind::Expression(Expression {
                        location: (1, 16).into(),
                        kind: Increment {
                            is_post: true,
                            e: Box::new(Expression {
                                location: (1, 16).into(),
                                kind: Variable("x".into()),
                            }),
                        },
                    }),
                }),
                loop_label: None,
            },
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_statement_do_while_loop() {
        let src = indoc! {r#"
        do {
            x++; --y;
        } while (x < 10);
        "#};
        let expected = Ok(Statement {
            location: (1,1).into(),
            labels: vec![],
            kind: DoWhile {
                loop_body: Box::new(Statement {
                    location: (1,4).into(),
                    labels: vec![],
                    kind: SubBlock(Block {
                        start_loc: (1,4).into(),
                        end_loc: (3,1).into(),
                        items: vec![
                            BlockItem::Statement(Statement {
                                location: (2,5).into(),
                                labels: vec![],
                                kind: Expression(Expression {
                                    location: (2,5).into(),
                                    kind: Increment {
                                        is_post: true,
                                        e: Box::new(Expression {
                                            location: (2,5).into(),
                                            kind: Variable("x".into()),
                                        }),
                                    },
                                }),
                            }),
                            BlockItem::Statement(Statement {
                                location: (2,10).into(),
                                labels: vec![],
                                kind: Expression(Expression {
                                    location: (2,10).into(),
                                    kind: Decrement {
                                        is_post: false,
                                        e: Box::new(Expression {
                                            location: (2,12).into(),
                                            kind: Variable("y".into()),
                                        }),
                                    },
                                }),
                            }),
                        ],
                    }),
                }),
                post_condition: Box::new(Expression {
                    location: (3,10).into(),
                    kind: Binary(
                        BinaryOperator::LessThan,
                        Box::new(Expression {
                            location: (3,10).into(),
                            kind: Variable("x".into()),
                        }),
                        Box::new(Expression {
                            location: (3,14).into(),
                            kind: IntConstant("10".into(), Decimal),
                        }),
                    ),
                }),
                loop_label: None,
            },
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }

    #[test]
    fn test_parse_continue_statement() {
        let src = "continue;";
        let expected = Ok(Statement {
            location: (1,1).into(),
            labels: vec![],
            kind: Continue(None),
        });
        run_parse_statement_test_case(StatementTestCase { src, expected });
    }


    #[test]
    fn test_parse_break_statement() {
        let src = "break;";
        let expected = Ok(Statement {
            location: (1,1).into(),
            labels: vec![],
            kind: Break(None),
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
                op: None,
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
                        op: None,
                    },
                }),
                op: None,
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
    fn test_parse_expression_compound_assignment_add() {
        let src = "a += b";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1,6).into(),
                    kind: Variable("b".to_string()),
                }),
                op: Some(CompoundAssignmentType::Add),
            },
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_compound_assignment_subtract() {
        let src = "a -= b";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1,6).into(),
                    kind: Variable("b".to_string()),
                }),
                op: Some(CompoundAssignmentType::Subtract),
            },
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_compound_assignment_multiply() {
        let src = "a *= b";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1,6).into(),
                    kind: Variable("b".to_string()),
                }),
                op: Some(CompoundAssignmentType::Multiply),
            },
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_compound_assignment_divide() {
        let src = "a /= b";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1,6).into(),
                    kind: Variable("b".to_string()),
                }),
                op: Some(CompoundAssignmentType::Divide),
            },
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_compound_assignment_modulo() {
        let src = "a %= b";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1,6).into(),
                    kind: Variable("b".to_string()),
                }),
                op: Some(CompoundAssignmentType::Modulo),
            },
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_compound_assignment_bitwise_and() {
        let src = "a &= b";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1,6).into(),
                    kind: Variable("b".to_string()),
                }),
                op: Some(CompoundAssignmentType::BitwiseAnd),
            },
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_compound_assignment_bitwise_or() {
        let src = "a |= b";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1,6).into(),
                    kind: Variable("b".to_string()),
                }),
                op: Some(CompoundAssignmentType::BitwiseOr),
            },
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_compound_assignment_bitwise_xor() {
        let src = "a ^= b";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1,6).into(),
                    kind: Variable("b".to_string()),
                }),
                op: Some(CompoundAssignmentType::BitwiseXor),
            },
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_compound_assignment_left_shift() {
        let src = "a <<= b";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1,7).into(),
                    kind: Variable("b".to_string()),
                }),
                op: Some(CompoundAssignmentType::LeftShift),
            },
        });
        run_parse_expression_test_case(ExprTestCase { src, expected });
    }

    #[test]
    fn test_parse_expression_compound_assignment_right_shift() {
        let src = "a >>= b";
        let expected = Ok(Expression {
            location: (1,1).into(),
            kind: Assignment {
                lvalue: Box::new(Expression {
                    location: (1,1).into(),
                    kind: Variable("a".to_string()),
                }),
                rvalue: Box::new(Expression {
                    location: (1,7).into(),
                    kind: Variable("b".to_string()),
                }),
                op: Some(CompoundAssignmentType::RightShift),
            },
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
    
    #[test]
    fn test_simple_ternary_expression_parsing() {
        run_parse_expression_test_case(ExprTestCase {
            src: "is_expected ? 0 : 1",
            expected: Ok(Expression {
                location: (1,1).into(),
                kind: Conditional {
                    condition: Box::new(Expression {
                        location: (1,1).into(),
                        kind: Variable("is_expected".to_string()),
                    }),
                    then_expr: Box::new(Expression {
                        location: (1,15).into(),
                        kind: IntConstant("0".to_string(), Decimal),
                    }),
                    else_expr: Box::new(Expression {
                        location: (1,19).into(),
                        kind: IntConstant("1".to_string(), Decimal),
                    }),
                },
            }),
        })
    }

    #[test]
    fn test_assign_after_eval_ternary_expression() {
        run_parse_expression_test_case(ExprTestCase {
            src: "a = b ? 1 : 2",
            expected: Ok(Expression {
                location: (1, 1).into(),
                kind: Assignment {
                    lvalue: Box::new(Expression {
                        location: (1, 1).into(),
                        kind: Variable("a".to_string()),
                    }),
                    rvalue: Box::new(Expression {
                        location: (1, 5).into(),
                        kind: Conditional {
                            condition: Box::new(Expression {
                                location: (1, 5).into(),
                                kind: Variable("b".to_string()),
                            }),
                            then_expr: Box::new(Expression {
                                location: (1, 9).into(),
                                kind: IntConstant("1".to_string(), Decimal),
                            }),
                            else_expr: Box::new(Expression {
                                location: (1, 13).into(),
                                kind: IntConstant("2".to_string(), Decimal),
                            }),
                        },
                    }),
                    op: None,
                },
            }),
        })
    }


    #[test]
    fn test_logical_operator_in_condition_with_ternary() {
        run_parse_expression_test_case(ExprTestCase {
            src: "a && b ? 1 : 2",
            expected: Ok(Expression {
                location: (1, 1).into(),
                kind: Conditional {
                    condition: Box::new(Expression {
                        location: (1, 1).into(),
                        kind: Binary(
                            BinaryOperator::And,
                            Box::new(Expression {
                                location: (1, 1).into(),
                                kind: Variable("a".to_string()),
                            }),
                            Box::new(Expression {
                                location: (1, 6).into(),
                                kind: Variable("b".to_string()),
                            }),
                        ),
                    }),
                    then_expr: Box::new(Expression {
                        location: (1, 10).into(),
                        kind: IntConstant("1".to_string(), Decimal),
                    }),
                    else_expr: Box::new(Expression {
                        location: (1, 14).into(),
                        kind: IntConstant("2".to_string(), Decimal),
                    }),
                },
            }),
        })
    }


    #[test]
    fn test_ternary_operator_associativity_with_if_else_ladder() {
        run_parse_expression_test_case(ExprTestCase {
            src: "a ? 1 : b ? 2 : 3",
            expected: Ok(Expression {
                location: (1, 1).into(),
                kind: Conditional {
                    condition: Box::new(Expression {
                        location: (1, 1).into(),
                        kind: Variable("a".to_string()),
                    }),
                    then_expr: Box::new(Expression {
                        location: (1, 5).into(),
                        kind: IntConstant("1".to_string(), Decimal),
                    }),
                    else_expr: Box::new(Expression {
                        location: (1, 9).into(),
                        kind: Conditional {
                            condition: Box::new(Expression {
                                location: (1, 9).into(),
                                kind: Variable("b".to_string()),
                            }),
                            then_expr: Box::new(Expression {
                                location: (1, 13).into(),
                                kind: IntConstant("2".to_string(), Decimal),
                            }),
                            else_expr: Box::new(Expression {
                                location: (1, 17).into(),
                                kind: IntConstant("3".to_string(), Decimal),
                            }),
                        },
                    }),
                },
            }),
        })
    }

    #[test]
    fn test_ternary_operator_with_ternary_in_the_middle() {
        run_parse_expression_test_case(ExprTestCase {
            src: "a ? b ? 0 : 1 : 2",
            expected: Ok(Expression {
                location: (1, 1).into(),
                kind: Conditional {
                    condition: Box::new(Expression {
                        location: (1, 1).into(),
                        kind: Variable("a".to_string()),
                    }),
                    then_expr: Box::new(Expression {
                        location: (1, 5).into(),
                        kind: Conditional {
                            condition: Box::new(Expression {
                                location: (1, 5).into(),
                                kind: Variable("b".to_string()),
                            }),
                            then_expr: Box::new(Expression {
                                location: (1, 9).into(),
                                kind: IntConstant("0".to_string(), Decimal),
                            }),
                            else_expr: Box::new(Expression {
                                location: (1, 13).into(),
                                kind: IntConstant("1".to_string(), Decimal),
                            }),
                        },
                    }),
                    else_expr: Box::new(Expression {
                        location: (1, 17).into(),
                        kind: IntConstant("2".to_string(), Decimal),
                    }),
                },
            }),
        })
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
            kind: DeclarationKind::VarDeclaration {
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
            kind: DeclarationKind::VarDeclaration {
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
                    labels: vec![],
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
                    kind: DeclarationKind::VarDeclaration {
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
                    location: (3, 5).into(),
                    kind: DeclarationKind::VarDeclaration {
                        identifier: Symbol {
                            location: (3, 9).into(),
                            name: "b".to_string(),
                        },
                        init_expression: None,
                    },
                }),
                BlockItem::Statement(Statement {
                    location: (4, 5).into(),
                    labels: vec![],
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
                            op: None,
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
                    kind: DeclarationKind::VarDeclaration {
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
                    labels: vec![],
                    kind: SubBlock(Block {
                        start_loc: (3, 3).into(),
                        end_loc: (7, 3).into(),
                        items: vec![
                            BlockItem::Declaration(Declaration {
                                location: (4, 5).into(),
                                kind: DeclarationKind::VarDeclaration {
                                    identifier: Symbol { location: (4, 9).into(), name: "b".to_string() },
                                    init_expression: Some(Expression {
                                        location: (4, 13).into(),
                                        kind: IntConstant("20".to_string(), Decimal),
                                    }),
                                },
                            }),
                            BlockItem::Declaration(Declaration {
                                location: (5, 5).into(),
                                kind: DeclarationKind::VarDeclaration {
                                    identifier: Symbol { location: (5, 9).into(), name: "c".to_string() },
                                    init_expression: Some(Expression {
                                        location: (5, 13).into(),
                                        kind: IntConstant("30".to_string(), Decimal),
                                    }),
                                },
                            }),
                            BlockItem::Statement(Statement {
                                location: (6, 5).into(),
                                labels: vec![],
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
                                        op: None,
                                    },
                                }),
                            }),
                        ],
                    }),
                }),
                BlockItem::Statement(Statement {
                    location: (8, 3).into(),
                    labels: vec![],
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
    #[case("assignment_simple", "a = 1")]
    #[case("assignment_multi", "a = b = 10")]
    #[case("compound_assign_add", "a += 10")]
    #[case("compound_assign_subtract", "a -= 10")]
    #[case("compound_assign_multiply", "a *= 10")]
    #[case("compound_assign_divide", "a /= 10")]
    #[case("compound_assign_modulo", "a %= 10")]
    #[case("compound_assign_left_shift", "a <<= 2")]
    #[case("compound_assign_right_shift", "a >>= 2")]
    #[case("compound_assign_bitwise_and", "a &= 10")]
    #[case("compound_assign_bitwise_or", "a |= 10")]
    #[case("compound_assign_bitwise_xor", "a ^= 10")]
    fn test_should_parse_assignment_operator_with_correct_precedence(#[case] description: &str, #[case] src: &str) {
        run_snapshot_test_for_parse_expression("assignment expression with correct precedence", description, "expr/assignment", src);
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
    #[case("a += b += c", "a += (b += c)")]
    #[case("a -= b -= c", "a -= (b -= c)")]
    #[case("a *= b *= c", "a *= (b *= c)")]
    #[case("a /= b /= c", "a /= (b /= c)")]
    #[case("a %= b %= c", "a %= (b %= c)")]
    #[case("a &= b &= c", "a &= (b &= c)")]
    #[case("a |= b |= c", "a |= (b |= c)")]
    #[case("a ^= b ^= c", "a ^= (b ^= c)")]
    #[case("a >>= b >>= c", "a >>= (b >>= c)")]
    #[case("a <<= b <<= c", "a <<= (b <<= c)")]
    #[case("a += b + 10", "a += (b + 10)")]
    #[case("a -= b + 10", "a -= (b + 10)")]
    #[case("a *= b + 10", "a *= (b + 10)")]
    #[case("a /= b + 10", "a /= (b + 10)")]
    #[case("a %= b + 10", "a %= (b + 10)")]
    #[case("a <<= b + 10", "a <<= (b + 10)")]
    #[case("a >>= b + 10", "a >>= (b + 10)")]
    #[case("a &= b + 10", "a &= (b + 10)")]
    #[case("a |= b + 10", "a |= (b + 10)")]
    #[case("a ^= b + 10", "a ^= (b + 10)")]
    fn test_assignment_operator_precedence_and_associativity(#[case] src1: &str, #[case] src2: &str) {
        run_expression_equivalence_test(src1, src2);
    }

    #[rstest]
    #[case("a ? b : c", "a ? (b) : (c)")]
    #[case("a && b ? c : d", "(a && b) ? (c) : (d)")]
    #[case("a && b ? c + d : c - d", "(a && b)?(c+d):(c-d)")]
    #[case("a = b ? 1 : 2", "a = (b ? 1 : 2)")]
    #[case("a ? x : y = 10", "(a ? x : y) = 10")]
    #[case("a ? b ? 1 : 2 : 3", "a ? (b ? 1: 2): 3")]
    #[case("a ? 1 : b ? 2 : c ? 3 : 4", "a ? 1 : (b ? 2 : (c ? 3 : 4))")]
    #[case("a ? 1 : b + 1", "a ? 1 : (b+1)")]
    #[case("a ? x = y : z", "a ? (x = y) : z")]
    #[case("a ? b : c || d", "a ? b : (c || d)")]
    #[case("a ? x : y += 10", "(a ? x : y) += 10")]
    fn test_ternary_operator_precedence_and_associativity(#[case] src1: &str, #[case] src2: &str) {
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
            (Assignment { lvalue: lv1, rvalue: rv1, op: op1 },
                Assignment { lvalue: lv2, rvalue: rv2, op: op2 }) =>
                is_equivalent_expression(&*lv1, &*lv2)
                    && is_equivalent_expression(&*rv1, &*rv2)
                    && op1 == op2,
            (Conditional { condition: c1, then_expr: t1, else_expr: e1 },
             Conditional { condition: c2, then_expr: t2, else_expr: e2 }) =>
                is_equivalent_expression(&*c1, &*c2)
                    && is_equivalent_expression(&*t1, &*t2)
                    && is_equivalent_expression(&*e1, &*e2),
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