use std::collections::{HashMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::iter::Peekable;
use std::str::Chars;

use once_cell::sync::Lazy;
use thiserror::Error;

use crate::common::{Location, Radix};

#[derive(Debug, Eq, PartialEq, Clone, Hash, Copy)]
pub enum KeywordIdentifier {
    TypeInt,
    TypeVoid,
    Return,

    If,
    Else,
    Goto,

    Do,
    While,
    For,
    Break,
    Continue,
}

impl Radix {
    pub fn value(&self) -> u32 {
        match self {
            Radix::Binary => 2,
            Radix::Octal => 8,
            Radix::Decimal => 10,
            Radix::Hexadecimal => 16,
        }
    }
}

impl Display for Radix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Radix::Binary => f.write_str("binary"),
            Radix::Octal => f.write_str("octal"),
            Radix::Decimal => f.write_str("decimal"),
            Radix::Hexadecimal => f.write_str("hexadecimal"),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum TokenTag {
    Keyword,
    OpenParentheses,
    CloseParentheses,
    OpenBrace,
    CloseBrace,
    Semicolon,
    Identifier,
    IntConstant,
    OperatorDiv,
    OperatorMinus,
    OperatorUnaryComplement,
    OperatorUnaryDecrement,
    OperatorUnaryIncrement,
    OperatorPlus,
    OperatorAsterisk,
    OperatorModulo,
    OperatorLeftShift,
    OperatorRightShift,
    OperatorBitwiseAnd,
    OperatorBitwiseOr,
    OperatorBitwiseXor,
    OperatorUnaryLogicalNot,
    OperatorLogicalAnd,
    OperatorLogicalOr,
    OperatorRelationalEqual,
    OperatorRelationalNotEqual,
    OperatorRelationalLessThan,
    OperatorRelationalLessThanEqualTo,
    OperatorRelationalGreaterThan,
    OperatorRelationalGreaterThanEqualTo,
    OperatorAssignment,
    OperatorCompoundAssignmentAdd,
    OperatorCompoundAssignmentSubtract,
    OperatorCompoundAssignmentMultiply,
    OperatorCompoundAssignmentDivide,
    OperatorCompoundAssignmentModulo,
    OperatorCompoundAssignmentLeftShift,
    OperatorCompoundAssignmentRightShift,
    OperatorCompoundAssignmentBitwiseAnd,
    OperatorCompoundAssignmentBitwiseOr,
    OperatorCompoundAssignmentBitwiseXor,
    OperatorTernaryThen,
    OperatorColon,
    Comma,
}

impl Display for TokenTag {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum TokenType<'a> {
    Keyword(KeywordIdentifier),
    OpenParentheses,
    CloseParentheses,
    OpenBrace,
    CloseBrace,
    Semicolon,
    Identifier(&'a str),
    IntConstant(&'a str, Radix),

    OperatorUnaryComplement,
    OperatorUnaryIncrement,
    OperatorUnaryDecrement,
    OperatorPlus,
    OperatorMinus,
    OperatorAsterisk,
    OperatorDiv,
    OperatorModulo,
    OperatorLeftShift,
    OperatorRightShift,
    OperatorBitwiseAnd,
    OperatorBitwiseOr,
    OperatorBitwiseXor,

    OperatorUnaryLogicalNot,
    OperatorLogicalAnd,
    OperatorLogicalOr,
    OperatorRelationalEqual,
    OperatorRelationalNotEqual,
    OperatorRelationalLessThan,
    OperatorRelationalLessThanEqualTo,
    OperatorRelationalGreaterThan,
    OperatorRelationalGreaterThanEqualTo,

    OperatorAssignment,
    OperatorCompoundAssignmentAdd,
    OperatorCompoundAssignmentSubtract,
    OperatorCompoundAssignmentMultiply,
    OperatorCompoundAssignmentDivide,
    OperatorCompoundAssignmentModulo,
    OperatorCompoundAssignmentLeftShift,
    OperatorCompoundAssignmentRightShift,
    OperatorCompoundAssignmentBitwiseAnd,
    OperatorCompoundAssignmentBitwiseOr,
    OperatorCompoundAssignmentBitwiseXor,

    OperatorTernaryThen,
    OperatorColon,
    Comma,
}

impl<'a> TokenType<'a> {
    pub fn tag(&self) -> TokenTag {
        match self {
            TokenType::Keyword(_) => TokenTag::Keyword,
            TokenType::OpenParentheses => TokenTag::OpenParentheses,
            TokenType::CloseParentheses => TokenTag::CloseParentheses,
            TokenType::OpenBrace => TokenTag::OpenBrace,
            TokenType::CloseBrace => TokenTag::CloseBrace,
            TokenType::Semicolon => TokenTag::Semicolon,
            TokenType::Identifier(_) => TokenTag::Identifier,
            TokenType::IntConstant(_, _) => TokenTag::IntConstant,

            TokenType::OperatorUnaryComplement => TokenTag::OperatorUnaryComplement,
            TokenType::OperatorUnaryDecrement => TokenTag::OperatorUnaryDecrement,
            TokenType::OperatorUnaryIncrement => TokenTag::OperatorUnaryIncrement,

            TokenType::OperatorDiv => TokenTag::OperatorDiv,

            TokenType::OperatorMinus => TokenTag::OperatorMinus,
            TokenType::OperatorPlus => TokenTag::OperatorPlus,
            TokenType::OperatorAsterisk => TokenTag::OperatorAsterisk,
            TokenType::OperatorModulo => TokenTag::OperatorModulo,

            TokenType::OperatorLeftShift => TokenTag::OperatorLeftShift,
            TokenType::OperatorRightShift => TokenTag::OperatorRightShift,
            TokenType::OperatorBitwiseAnd => TokenTag::OperatorBitwiseAnd,
            TokenType::OperatorBitwiseOr => TokenTag::OperatorBitwiseOr,
            TokenType::OperatorBitwiseXor => TokenTag::OperatorBitwiseXor,

            TokenType::OperatorUnaryLogicalNot => TokenTag::OperatorUnaryLogicalNot,
            TokenType::OperatorLogicalAnd => TokenTag::OperatorLogicalAnd,
            TokenType::OperatorLogicalOr => TokenTag::OperatorLogicalOr,
            TokenType::OperatorRelationalEqual => TokenTag::OperatorRelationalEqual,
            TokenType::OperatorRelationalNotEqual => TokenTag::OperatorRelationalNotEqual,
            TokenType::OperatorRelationalLessThan => TokenTag::OperatorRelationalLessThan,
            TokenType::OperatorRelationalLessThanEqualTo => TokenTag::OperatorRelationalLessThanEqualTo,
            TokenType::OperatorRelationalGreaterThan => TokenTag::OperatorRelationalGreaterThan,
            TokenType::OperatorRelationalGreaterThanEqualTo => TokenTag::OperatorRelationalGreaterThanEqualTo,

            TokenType::OperatorAssignment => TokenTag::OperatorAssignment,
            TokenType::OperatorCompoundAssignmentAdd => TokenTag::OperatorCompoundAssignmentAdd,
            TokenType::OperatorCompoundAssignmentSubtract => TokenTag::OperatorCompoundAssignmentSubtract,
            TokenType::OperatorCompoundAssignmentMultiply => TokenTag::OperatorCompoundAssignmentMultiply,
            TokenType::OperatorCompoundAssignmentDivide => TokenTag::OperatorCompoundAssignmentDivide,
            TokenType::OperatorCompoundAssignmentModulo => TokenTag::OperatorCompoundAssignmentModulo,
            TokenType::OperatorCompoundAssignmentLeftShift => TokenTag::OperatorCompoundAssignmentLeftShift,
            TokenType::OperatorCompoundAssignmentRightShift => TokenTag::OperatorCompoundAssignmentRightShift,
            TokenType::OperatorCompoundAssignmentBitwiseAnd => TokenTag::OperatorCompoundAssignmentBitwiseAnd,
            TokenType::OperatorCompoundAssignmentBitwiseOr => TokenTag::OperatorCompoundAssignmentBitwiseOr,
            TokenType::OperatorCompoundAssignmentBitwiseXor => TokenTag::OperatorCompoundAssignmentBitwiseXor,

            TokenType::OperatorTernaryThen => TokenTag::OperatorTernaryThen,
            TokenType::OperatorColon => TokenTag::OperatorColon,
            TokenType::Comma => TokenTag::Comma,
        }
    }

    pub fn is_unary_operator(&self) -> bool {
        match self {
            TokenType::OperatorUnaryComplement
            | TokenType::OperatorUnaryDecrement
            | TokenType::OperatorUnaryIncrement
            | TokenType::OperatorMinus
            | TokenType::OperatorUnaryLogicalNot => true,
            _ => false,
        }
    }

    pub fn is_binary_operator(&self) -> bool {
        match self {
            TokenType::OperatorPlus
            | TokenType::OperatorMinus
            | TokenType::OperatorAsterisk
            | TokenType::OperatorDiv
            | TokenType::OperatorModulo
            | TokenType::OperatorLeftShift
            | TokenType::OperatorRightShift
            | TokenType::OperatorBitwiseAnd
            | TokenType::OperatorBitwiseOr
            | TokenType::OperatorBitwiseXor
            | TokenType::OperatorRelationalEqual
            | TokenType::OperatorRelationalNotEqual
            | TokenType::OperatorRelationalGreaterThan
            | TokenType::OperatorRelationalGreaterThanEqualTo
            | TokenType::OperatorRelationalLessThan
            | TokenType::OperatorRelationalLessThanEqualTo
            | TokenType::OperatorLogicalAnd
            | TokenType::OperatorLogicalOr
            | TokenType::OperatorTernaryThen
            | TokenType::OperatorAssignment
            | TokenType::OperatorCompoundAssignmentAdd
            | TokenType::OperatorCompoundAssignmentSubtract
            | TokenType::OperatorCompoundAssignmentMultiply
            | TokenType::OperatorCompoundAssignmentDivide
            | TokenType::OperatorCompoundAssignmentModulo
            | TokenType::OperatorCompoundAssignmentLeftShift
            | TokenType::OperatorCompoundAssignmentRightShift
            | TokenType::OperatorCompoundAssignmentBitwiseXor
            | TokenType::OperatorCompoundAssignmentBitwiseOr
            | TokenType::OperatorCompoundAssignmentBitwiseAnd => true,
            _ => false,
        }
    }
}

#[derive(Error, Debug, PartialEq, Clone)]
pub enum LexerError {
    #[error("{location:?}: invalid character {ch:?} for identifier")]
    InvalidIdentifierCharacter { location: Location, ch: char },

    #[error("{location:?}: unexpected character {ch:?}")]
    UnexpectedCharacter { location: Location, ch: char },

    #[error("unexpected end of stream")]
    UnexpectedEndOfStream
}

static KEYWORDS: Lazy<HashMap<&'static str, KeywordIdentifier>> = Lazy::new(|| {
    HashMap::from([
        ("int", KeywordIdentifier::TypeInt),
        ("void", KeywordIdentifier::TypeVoid),
        ("return", KeywordIdentifier::Return),

        ("if", KeywordIdentifier::If),
        ("else", KeywordIdentifier::Else),

        ("goto", KeywordIdentifier::Goto),

        ("do", KeywordIdentifier::Do),
        ("while", KeywordIdentifier::While),
        ("for", KeywordIdentifier::For),
        ("break", KeywordIdentifier::Break),
        ("continue", KeywordIdentifier::Continue),
    ])
});

static KEYWORD_STRINGS: Lazy<HashMap<KeywordIdentifier, &'static str>> = Lazy::new(|| {
    HashMap::from([
        (KeywordIdentifier::TypeInt, "int"),
        (KeywordIdentifier::TypeVoid, "void"),
        (KeywordIdentifier::Return, "return"),

        (KeywordIdentifier::If, "if"),
        (KeywordIdentifier::Else, "else"),

        (KeywordIdentifier::Goto, "goto"),

        (KeywordIdentifier::Do, "do"),
        (KeywordIdentifier::While, "while"),
        (KeywordIdentifier::For, "for"),
        (KeywordIdentifier::Break, "break"),
        (KeywordIdentifier::Continue, "continue"),
    ])
});

impl<'a> Display for TokenType<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            TokenType::Keyword(k) => f.write_str(KEYWORD_STRINGS.get(k).unwrap()),
            TokenType::OpenParentheses => f.write_str("("),
            TokenType::CloseParentheses => f.write_str(")"),
            TokenType::OpenBrace => f.write_str("{"),
            TokenType::CloseBrace => f.write_str("}"),
            TokenType::Semicolon => f.write_str(";"),
            TokenType::Identifier(x) => f.write_fmt(format_args!("identifier:{}", x)),
            TokenType::IntConstant(x, radix) => f.write_fmt(format_args!("int:[{}, radix:{}]", x, radix)),
            TokenType::OperatorDiv => f.write_str("/"),
            TokenType::OperatorUnaryComplement => f.write_str("~"),
            TokenType::OperatorUnaryDecrement => f.write_str("--"),
            TokenType::OperatorUnaryIncrement => f.write_str("--"),
            TokenType::OperatorMinus => f.write_str("-"),
            TokenType::OperatorPlus => f.write_str("+"),
            TokenType::OperatorAsterisk => f.write_str("*"),
            TokenType::OperatorModulo => f.write_str("%"),
            TokenType::OperatorLeftShift => f.write_str("<<"),
            TokenType::OperatorRightShift => f.write_str(">>"),
            TokenType::OperatorBitwiseAnd => f.write_str("&"),
            TokenType::OperatorBitwiseOr => f.write_str("|"),
            TokenType::OperatorBitwiseXor => f.write_str("^"),
            TokenType::OperatorUnaryLogicalNot => f.write_str("!"),
            TokenType::OperatorLogicalAnd => f.write_str("&&"),
            TokenType::OperatorLogicalOr => f.write_str("||"),
            TokenType::OperatorRelationalEqual => f.write_str("=="),
            TokenType::OperatorRelationalNotEqual => f.write_str("!="),
            TokenType::OperatorRelationalLessThan => f.write_str("<"),
            TokenType::OperatorRelationalLessThanEqualTo => f.write_str("<="),
            TokenType::OperatorRelationalGreaterThan => f.write_str(">"),
            TokenType::OperatorRelationalGreaterThanEqualTo => f.write_str(">="),
            TokenType::OperatorAssignment => f.write_str("="),
            TokenType::OperatorCompoundAssignmentAdd => f.write_str("+="),
            TokenType::OperatorCompoundAssignmentSubtract => f.write_str("-="),
            TokenType::OperatorCompoundAssignmentMultiply => f.write_str("*="),
            TokenType::OperatorCompoundAssignmentDivide => f.write_str("/="),
            TokenType::OperatorCompoundAssignmentModulo => f.write_str("%="),
            TokenType::OperatorCompoundAssignmentLeftShift => f.write_str("<<="),
            TokenType::OperatorCompoundAssignmentRightShift => f.write_str(">>="),
            TokenType::OperatorCompoundAssignmentBitwiseAnd => f.write_str("&="),
            TokenType::OperatorCompoundAssignmentBitwiseOr => f.write_str("|="),
            TokenType::OperatorCompoundAssignmentBitwiseXor => f.write_str("^="),
            TokenType::OperatorTernaryThen => f.write_str("?"),
            TokenType::OperatorColon => f.write_str(":"),
            TokenType::Comma => f.write_str(","),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Token<'a> {
    pub token_type: TokenType<'a>,
    pub location: Location,
}

#[derive(Debug, Eq, PartialEq)]
enum LexerMode {
    Default,
    LineComment,
    BlockComment,
}

pub struct Lexer<'a> {
    input: &'a str,
    lexer_mode: LexerMode,
    char_stream: Peekable<Chars<'a>>,
    cur_stream_pos: usize,
    cur_location: Location,

    token_peek_buffer: VecDeque<Result<Token<'a>, LexerError>>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            lexer_mode: LexerMode::Default,
            char_stream: input.chars().peekable(),
            cur_stream_pos: 0,
            cur_location: Location { line: 1, column: 1 },
            token_peek_buffer: VecDeque::new(),
        }
    }

    fn next_char(&mut self) -> Option<char> {
        let ch = self.char_stream.next();
        if ch.is_none() {
            return None;
        }
        let c = ch.unwrap();
        self.cur_stream_pos += c.len_utf8();
        match c {
            '\n' => self.cur_location.advance_line(),
            '\t' => self.cur_location.advance_tab(),
            _ => self.cur_location.advance(),
        }
        return ch;
    }

    fn tokenize_single_char(&mut self, token_type: TokenType<'a>) -> Token<'a> {
        let token = Token {
            token_type,
            location: self.cur_location.clone(),
        };
        self.next_char();
        token
    }

    fn tokenize_integer_constant(&mut self) -> Result<Token<'a>, LexerError> {
        let start_loc = self.cur_location.clone();
        let start_pos = self.cur_stream_pos;
        let first_digit = self.next_char().unwrap();
        let mut expected_radix = Radix::Octal;
        if first_digit != '0' {
            expected_radix = Radix::Decimal;
        } else {
            let second_digit = self.char_stream.peek();
            if second_digit.is_none() {
                return Ok(Token { location: start_loc, token_type: TokenType::IntConstant("0", Radix::Decimal) });
            }
            let digit2 = second_digit.unwrap().clone();
            if digit2 == 'x' || digit2 == 'X' {
                expected_radix = Radix::Hexadecimal;
                self.next_char();
            } else if digit2 == 'b' || digit2 == 'B' {
                expected_radix = Radix::Binary;
                self.next_char();
            } else if !digit2.is_digit(Radix::Octal.value()) {
                // If it is neither 0x, 0X, 0b, 0B or 0[Octal digit]
                // then it is an invalid number. Hence, it is an error.
                return Ok(Token { location: start_loc, token_type: TokenType::IntConstant("0", Radix::Decimal) });
            }
        }
        // Once we have determined the radix, we iterate through the digits as long
        // as we find a valid stop point: semicolon, whitespace (space, new-line, tab), end.
        // If we stop at characters that are not valid digits, then we throw an error. We
        // don't support digit grouping yet to keep the lexer simple.
        loop {
            let next = self.char_stream.peek();
            if next.is_none() {
                break;
            }
            let n = next.unwrap().clone();
            if !n.is_digit(expected_radix.value()) {
                if n.is_alphanumeric() {
                    // The current character is an alphabet. This means this is a case of an
                    // invalid identifier. Identifiers are supposed to start with _ or alphabet.
                    return Err(LexerError::InvalidIdentifierCharacter {
                        location: start_loc,
                        ch: first_digit,
                    })
                }
                break;
            }
            self.next_char();
        }
        return Ok(Token {
            location: start_loc,
            token_type: TokenType::IntConstant(&self.input[start_pos..self.cur_stream_pos], expected_radix),
        });
    }

    fn tokenize_identifier_or_keyword(&mut self) -> Result<Token<'a>, LexerError> {
        let start_loc = self.cur_location;
        let start_pos = self.cur_stream_pos;
        let first_char = self.next_char().unwrap();
        if first_char != '_' && !first_char.is_alphabetic() {
            return Err(LexerError::InvalidIdentifierCharacter { location: start_loc, ch: first_char });
        }
        loop {
            let ch = self.char_stream.peek();
            if ch.is_none() {
                break;
            }
            let c = ch.unwrap().clone();
            if !c.is_alphanumeric() && c != '_' {
                break;
            }
            self.next_char();
        }
        let word = &self.input[start_pos..self.cur_stream_pos];
        match KEYWORDS.get(word) {
            None => Ok(Token {
                location: start_loc,
                token_type: TokenType::Identifier(word),
            }),
            Some(k) => Ok(Token {
                location: start_loc,
                token_type: TokenType::Keyword(k.clone()),
            })
        }
    }

    fn next_token(&mut self) -> Result<Option<Token<'a>>, LexerError> {
        loop {
            let cur = self.char_stream.peek();
            if cur.is_none() {
                // We have reached the end of the stream. Hence, we cannot
                // tokenize anymore. So we just return None
                return Ok(None);
            }
            let cur_char = cur.unwrap();
            match self.lexer_mode {
                LexerMode::Default => {
                    match *cur_char {
                        ';' => {
                            let token = self.tokenize_single_char(TokenType::Semicolon);
                            return Ok(Some(token));
                        }
                        '(' => {
                            let token = self.tokenize_single_char(TokenType::OpenParentheses);
                            return Ok(Some(token));
                        }
                        ')' => {
                            let token = self.tokenize_single_char(TokenType::CloseParentheses);
                            return Ok(Some(token));
                        }
                        '{' => {
                            let token = self.tokenize_single_char(TokenType::OpenBrace);
                            return Ok(Some(token));
                        }
                        '}' => {
                            let token = self.tokenize_single_char(TokenType::CloseBrace);
                            return Ok(Some(token));
                        }
                        '+' => {
                            let start_loc = self.cur_location;
                            self.next_char();

                            if let Some(&nxt) = self.char_stream.peek() {
                                if nxt == '+' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorUnaryIncrement,
                                        location: start_loc,
                                    }));
                                } else if nxt == '=' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorCompoundAssignmentAdd,
                                        location: start_loc,
                                    }));
                                }
                            }
                            return Ok(Some(Token {
                                token_type: TokenType::OperatorPlus,
                                location: start_loc,
                            }));
                        }
                        '-' => {
                            let start_loc = self.cur_location;
                            self.next_char();

                            if let Some(&nxt) = self.char_stream.peek() {
                                if nxt == '-' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorUnaryDecrement,
                                        location: start_loc,
                                    }));
                                } else if nxt == '=' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorCompoundAssignmentSubtract,
                                        location: start_loc,
                                    }));
                                }
                            }
                            return Ok(Some(Token {
                                token_type: TokenType::OperatorMinus,
                                location: start_loc,
                            }));
                        }
                        '*' => {
                            let start_loc = self.cur_location;
                            self.next_char();

                            if let Some(&nxt) = self.char_stream.peek() {
                                if nxt == '=' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorCompoundAssignmentMultiply,
                                        location: start_loc,
                                    }));
                                }
                            }
                            return Ok(Some(Token {
                                token_type: TokenType::OperatorAsterisk,
                                location: start_loc,
                            }))
                        }
                        '%' => {
                            let start_loc = self.cur_location;
                            self.next_char();
                            if let Some(&nxt) = self.char_stream.peek() {
                                if nxt == '=' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorCompoundAssignmentModulo,
                                        location: start_loc,
                                    }));
                                }
                            }
                            return Ok(Some(Token {
                                token_type: TokenType::OperatorModulo,
                                location: start_loc,
                            }));
                        }
                        '!' => {
                            let loc = self.cur_location;
                            self.next_char();
                            return match &self.char_stream.peek() {
                                Some('=') => {
                                    self.next_char();
                                    Ok(Some(Token {
                                        token_type: TokenType::OperatorRelationalNotEqual,
                                        location: loc,
                                    }))
                                },
                                None | Some(_) => Ok(Some(Token {
                                    token_type: TokenType::OperatorUnaryLogicalNot,
                                    location: loc,
                                })),
                            }
                        }
                        '~' => {
                            let token = self.tokenize_single_char(TokenType::OperatorUnaryComplement);
                            return Ok(Some(token));
                        }
                        '/' => {
                            let div_loc = self.cur_location;
                            self.next_char();

                            if let Some(&nxt) = self.char_stream.peek() {
                                if nxt == '/' {
                                    self.lexer_mode = LexerMode::LineComment;
                                    self.next_char();
                                    continue;
                                }
                                if nxt == '*' {
                                    self.lexer_mode = LexerMode::BlockComment;
                                    self.next_char();
                                    continue;
                                }
                                if nxt == '=' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorCompoundAssignmentDivide,
                                        location: div_loc,
                                    }));
                                }
                            }
                            return Ok(Some(Token {
                                token_type: TokenType::OperatorDiv,
                                location: div_loc,
                            }))
                        }
                        '&' => {
                            let op_loc = self.cur_location;
                            self.next_char();
                            if let Some(&nxt) = self.char_stream.peek() {
                                if nxt == '&' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorLogicalAnd,
                                        location: op_loc,
                                    }));
                                } else if nxt == '=' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorCompoundAssignmentBitwiseAnd,
                                        location: op_loc,
                                    }));
                                }
                            }
                            return Ok(Some(Token {
                                token_type: TokenType::OperatorBitwiseAnd,
                                location: op_loc,
                            }))
                        }
                        '|' => {
                            let op_loc = self.cur_location;
                            self.next_char();
                            if let Some(&nxt) = self.char_stream.peek() {
                                if nxt == '|' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorLogicalOr,
                                        location: op_loc,
                                    }));
                                } else if nxt == '=' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorCompoundAssignmentBitwiseOr,
                                        location: op_loc,
                                    }));
                                }
                            }
                            return Ok(Some(Token {
                                token_type: TokenType::OperatorBitwiseOr,
                                location: op_loc,
                            }))
                        }
                        '^' => {
                            let xor_loc = self.cur_location;
                            self.next_char();
                            if let Some(&nxt) = self.char_stream.peek() {
                                if nxt == '=' {
                                    self.next_char();
                                    return Ok(Some(Token {
                                        token_type: TokenType::OperatorCompoundAssignmentBitwiseXor,
                                        location: xor_loc,
                                    }));
                                }
                            }
                            return Ok(Some(Token {
                                token_type: TokenType::OperatorBitwiseXor,
                                location: xor_loc,
                            }))
                        }
                        '<' => {
                            let loc = self.cur_location;
                            self.next_char();
                            if let Some(&nxt) = self.char_stream.peek() {
                                match nxt {
                                    '<' => {
                                        self.next_char();
                                        if let Some(&nxt2) = self.char_stream.peek() {
                                            if nxt2 == '=' {
                                                self.next_char();
                                                return Ok(Some(Token {
                                                    token_type: TokenType::OperatorCompoundAssignmentLeftShift,
                                                    location: loc,
                                                }));
                                            }
                                        }
                                        return Ok(Some(Token {
                                            token_type: TokenType::OperatorLeftShift,
                                            location: loc,
                                        }));
                                    }
                                    '=' => {
                                        self.next_char();
                                        return Ok(Some(Token {
                                            token_type: TokenType::OperatorRelationalLessThanEqualTo,
                                            location: loc,
                                        }))
                                    }
                                    _ => {},
                                }
                            }
                            return Ok(Some(Token {
                                token_type: TokenType::OperatorRelationalLessThan,
                                location: loc,
                            }));
                        }
                        '>' => {
                            let loc = self.cur_location;
                            self.next_char();
                            return match &self.char_stream.peek() {
                                Some('>') => {
                                    self.next_char();
                                    if let Some(&nxt2) = self.char_stream.peek() {
                                        if nxt2 == '=' {
                                            self.next_char();
                                            return Ok(Some(Token {
                                                token_type: TokenType::OperatorCompoundAssignmentRightShift,
                                                location: loc,
                                            }));
                                        }
                                    }
                                    Ok(Some(Token {
                                        token_type: TokenType::OperatorRightShift,
                                        location: loc,
                                    }))
                                },
                                Some('=') => {
                                    self.next_char();
                                    Ok(Some(Token {
                                        token_type: TokenType::OperatorRelationalGreaterThanEqualTo,
                                        location: loc,
                                    }))
                                },
                                None | Some(_) => Ok(Some(Token {
                                    token_type: TokenType::OperatorRelationalGreaterThan,
                                    location: loc,
                                })),
                            }
                        }
                        '=' => {
                            let loc = self.cur_location;
                            self.next_char();
                            return match &self.char_stream.peek() {
                                Some('=') => {
                                    self.next_char();
                                    Ok(Some(Token {
                                        token_type: TokenType::OperatorRelationalEqual,
                                        location: loc,
                                    }))
                                }
                                None | Some(_) => Ok(Some(Token {
                                    token_type: TokenType::OperatorAssignment,
                                    location: loc,
                                }))
                            }
                        }
                        '?' => {
                            let token = self.tokenize_single_char(TokenType::OperatorTernaryThen);
                            return Ok(Some(token));
                        }
                        ':' => {
                            let token = self.tokenize_single_char(TokenType::OperatorColon);
                            return Ok(Some(token));
                        }
                        '0'..='9' => {
                            let token = self.tokenize_integer_constant()?;
                            return Ok(Some(token));
                        }
                        '\n' | '\t' | ' ' => {
                            self.next_char();
                        }
                        ',' => {
                            let token = self.tokenize_single_char(TokenType::Comma);
                            return Ok(Some(token));
                        }
                        _ => {
                            let token = self.tokenize_identifier_or_keyword()?;
                            return Ok(Some(token));
                        }
                    }
                },

                LexerMode::LineComment => {
                    match *cur_char {
                        '\n' => {
                            self.lexer_mode = LexerMode::Default;
                            self.next_char();
                        }
                        _ => {
                            self.next_char();
                        }
                    }
                },

                LexerMode::BlockComment => {
                    match *cur_char {
                        '*' => {
                            self.next_char();
                            if let Some(&nxt) = self.char_stream.peek() {
                                if nxt == '/' {
                                    self.lexer_mode = LexerMode::Default;
                                    self.next_char(); // consume /
                                    continue;
                                }
                            }
                        },
                        _ => {
                            self.next_char();
                        }
                    }
                }
            }
        }
    }

    pub fn peek_n(&mut self, n: usize) -> Option<&Result<Token<'a>, LexerError>> {
        while self.token_peek_buffer.len() < n+1 {
            let next_token = self.next_token();
            match next_token {
                Ok(None) => { break; }
                Ok(Some(tok)) => { self.token_peek_buffer.push_back(Ok(tok)); }
                Err(e) => { self.token_peek_buffer.push_back(Err(e)); }
            }
        }
        self.token_peek_buffer.get(n)
    }

    #[inline]
    pub fn peek(&mut self) -> Option<&Result<Token<'a>, LexerError>> {
        self.peek_n(0)
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token<'a>, LexerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(tok) = self.token_peek_buffer.pop_front() {
            Some(tok)
        } else {
            let next_token = self.next_token();
            match next_token {
                Ok(None) => None,
                Ok(Some(tok)) => Some(Ok(tok)),
                Err(e) => Some(Err(e)),
            }
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::common::{Location, Radix};
    use crate::common::Radix::Decimal;
    use crate::lexer::{Lexer, LexerError, Token, TokenType};
    use crate::lexer::lexer::KEYWORDS;
    use crate::lexer::TokenType::*;

    type LexerResult<T> = Result<T, LexerError>;

    #[test]
    fn test_tokenizing_open_and_close_parentheses() {
        let source = "()";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OpenParentheses, location: Location { line: 1, column: 1 } },
            Token { token_type: CloseParentheses, location: Location { line: 1, column: 2 } },
        ]));
    }

    #[test]
    fn test_tokenizing_open_and_close_braces() {
        let source = "{}";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OpenBrace, location: Location { line: 1, column: 1 } },
            Token { token_type: CloseBrace, location: Location { line: 1, column: 2 } },
        ]));
    }

    #[test]
    fn test_tokenizing_with_newlines() {
        let source = "(\n\n)";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OpenParentheses, location: Location { line: 1, column: 1 } },
            Token { token_type: CloseParentheses, location: Location { line: 3, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_with_tabs() {
        let source = "(\t)";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OpenParentheses, location: Location { line: 1, column: 1 } },
            Token { token_type: CloseParentheses, location: Location { line: 1, column: 8 } },
        ]));
    }

    #[test]
    fn test_tokenizing_with_whitespaces() {
        let source = "(   )";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OpenParentheses, location: Location { line: 1, column: 1 } },
            Token { token_type: CloseParentheses, location: Location { line: 1, column: 5 } },
        ]));
    }

    #[test]
    fn test_tokenizing_semicolon() {
        let source = ";";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: Semicolon, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_complement() {
        let source = "~";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorUnaryComplement, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_increment() {
        let source = "++";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorUnaryIncrement, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_decrement() {
        let source = "--";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorUnaryDecrement, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_minus_or_negation() {
        let source = "-";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorMinus, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_left_shift() {
        let source = "<<";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorLeftShift, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_right_shift() {
        let source = ">>";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorRightShift, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_bitwise_and() {
        let source = "&";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorBitwiseAnd, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_bitwise_or() {
        let source = "|";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorBitwiseOr, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_bitwise_xor() {
        let source = "^";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorBitwiseXor, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_logical_not() {
        let source = "!";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorUnaryLogicalNot, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_logical_or() {
        let source = "||";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorLogicalOr, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_logical_and() {
        let source = "&&";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorLogicalAnd, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_relational_equal() {
        let source = "==";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorRelationalEqual, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_relational_not_equal() {
        let source = "!=";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorRelationalNotEqual, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_relational_less_than() {
        let source = "<";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorRelationalLessThan, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_relational_less_than_or_equal_to() {
        let source = "<=";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorRelationalLessThanEqualTo, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_relational_greater_than() {
        let source = ">";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorRelationalGreaterThan, location: Location { line: 1, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_relational_greater_than_or_equal_to() {
        let source = ">=";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorRelationalGreaterThanEqualTo, location: (1, 1).into() },
        ]));
    }

    #[test]
    fn test_tokenizing_ternary_then() {
        let source = "?";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorTernaryThen, location: (1, 1).into() },
        ]));
    }

    #[test]
    fn test_tokenizing_ternary_else() {
        let source = ":";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorColon, location: (1, 1).into() },
        ]));
    }

    #[rstest]
    #[case("+=", TokenType::OperatorCompoundAssignmentAdd)]
    #[case("-=", TokenType::OperatorCompoundAssignmentSubtract)]
    #[case("*=", TokenType::OperatorCompoundAssignmentMultiply)]
    #[case("/=", TokenType::OperatorCompoundAssignmentDivide)]
    #[case("%=", TokenType::OperatorCompoundAssignmentModulo)]
    #[case("<<=", TokenType::OperatorCompoundAssignmentLeftShift)]
    #[case(">>=", TokenType::OperatorCompoundAssignmentRightShift)]
    #[case("&=", TokenType::OperatorCompoundAssignmentBitwiseAnd)]
    #[case("|=", TokenType::OperatorCompoundAssignmentBitwiseOr)]
    #[case("^=", TokenType::OperatorCompoundAssignmentBitwiseXor)]
    fn test_tokenizing_compound_assignment(#[case] source: &str, #[case] token_type: TokenType) {
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type, location: (1, 1).into() },
        ]));
    }

    #[test]
    fn test_tokenizing_assignment_operator() {
        let source = "a = 10";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: Identifier("a"), location: (1,1).into() },
            Token { token_type: OperatorAssignment, location: (1,3).into() },
            Token { token_type: IntConstant("10", Decimal), location: (1,5).into() },
        ]));
    }

    #[test]
    fn test_tokenizing_post_increment() {
        let source = "a++";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: Identifier("a"), location: (1,1).into() },
            Token { token_type: OperatorUnaryIncrement, location: (1,2).into() }
        ]));
    }

    #[test]
    fn test_tokenizing_pre_increment() {
        let source = "++a";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorUnaryIncrement, location: (1,1).into() },
            Token { token_type: Identifier("a"), location: (1,3).into() },
        ]));
    }

    #[test]
    fn test_tokenizing_post_decrement() {
        let source = "a--";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: Identifier("a"), location: (1,1).into() },
            Token { token_type: OperatorUnaryDecrement, location: (1,2).into() }
        ]));
    }

    #[test]
    fn test_tokenizing_pre_decrement() {
        let source = "--a";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: OperatorUnaryDecrement, location: (1,1).into() },
            Token { token_type: Identifier("a"), location: (1,3).into() },
        ]));
    }

    #[test]
    fn test_tokenizing_comma() {
        let source = ",";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![Token { token_type: Comma, location: (1,1).into() }]));
    }

    #[rstest]
    #[case("abcde")]
    #[case("abcde123")]
    #[case("hello_world_123")]
    #[case("helloWorld123")]
    #[case("_abcde")]
    #[case("_123")]
    #[case("_123_456")]
    #[case("café")]
    #[case("αριθμός")]
    #[case("число")]
    #[case("数字")]
    fn test_tokenizing_valid_identifiers(#[case] src: &str) {
        let lexer = Lexer::new(src);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        let expected_tokens: LexerResult<Vec<Token>> = Ok(vec![
            Token { token_type: Identifier(src), location: Location { line: 1, column: 1 } },
        ]);
        assert_eq!(tokens, expected_tokens,
                   "lexing identifier {}: expected: {:?}, actual:{:?}",
                   src, expected_tokens, tokens);
    }

    #[test]
    fn test_tokenizing_keywords() {
        for (kw, kwid) in KEYWORDS.iter() {
            let lexer = Lexer::new(kw);
            let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
            let expected_tokens: LexerResult<Vec<Token>> = Ok(vec![
                Token { token_type: Keyword(kwid.clone()), location: Location { line: 1, column: 1 } },
            ]);
            assert_eq!(tokens, expected_tokens,
                       "lexing keyword identifier {}: expected: {:?}, actual:{:?}",
                       kw, expected_tokens, tokens);
        }
    }

    #[test]
    fn test_tokenizing_arithmetic_operators() {
        let start_loc = Location { line: 1, column: 1 };
        let arith_operator_tests = vec![
            ("+", vec![Token { token_type: OperatorPlus, location: start_loc }]),
            ("-", vec![Token { token_type: OperatorMinus, location: start_loc }]),
            ("*", vec![Token { token_type: OperatorAsterisk, location: start_loc }]),
            ("/", vec![Token { token_type: OperatorDiv, location: start_loc }]),
            ("%", vec![Token { token_type: OperatorModulo, location: start_loc }]),
        ];
        for at in arith_operator_tests {
            let src = at.0;
            let lexer = Lexer::new(src);
            let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
            let expected_tokens = Ok(at.1);
            assert_eq!(tokens, expected_tokens,
                       "lexing arith operator {}: expected: {:?}, actual:{:?}",
                       src, expected_tokens, tokens);
        }
    }

    #[test]
    fn test_tokenizing_arithmetic_expression() {
        let src = "a + b - (c / d) * 2 % 10";
        let lexer = Lexer::new(src);
        let expected = Ok(vec![
            Token { token_type: Identifier("a"), location:  (1, 1).into() },
            Token { token_type: OperatorPlus, location: (1, 3).into() },
            Token { token_type: Identifier("b"), location: (1, 5).into() },
            Token { token_type: OperatorMinus, location: (1, 7).into() },
            Token { token_type: OpenParentheses, location: (1, 9).into() },
            Token { token_type: Identifier("c"), location: (1, 10).into() },
            Token { token_type: OperatorDiv, location: (1, 12).into() },
            Token { token_type: Identifier("d"), location: (1, 14).into() },
            Token { token_type: CloseParentheses, location: (1, 15).into() },
            Token { token_type: OperatorAsterisk, location: (1, 17).into() },
            Token { token_type: IntConstant("2", Radix::Decimal), location: (1, 19).into() },
            Token { token_type: OperatorModulo, location: (1, 21).into() },
            Token { token_type: IntConstant("10", Radix::Decimal), location: (1, 23).into() },
        ]);
        let actual: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(actual, expected,
            "lexing arith expression {}: expected: {:#?}, actual:{:#?}",
            src, expected, actual);
    }

    #[rstest]
    #[case::decimal("1", Radix::Decimal)]
    #[case::decimal("0", Radix::Decimal)]
    #[case::decimal("2", Radix::Decimal)]
    #[case::decimal("3", Radix::Decimal)]
    #[case::decimal("44", Radix::Decimal)]
    #[case::decimal("55", Radix::Decimal)]
    #[case::decimal("66", Radix::Decimal)]
    #[case::decimal("777", Radix::Decimal)]
    #[case::decimal("888", Radix::Decimal)]
    #[case::decimal("9189", Radix::Decimal)]
    #[case::decimal("189087931798698368761873", Radix::Decimal)]
    #[case::decimal("0\t\t", Radix::Decimal)]
    #[case::decimal("0  ", Radix::Decimal)]
    #[case::decimal("0\n", Radix::Decimal)]

    #[case::hex("0xdeadbeef", Radix::Hexadecimal)]
    #[case::hex("0Xcafebabe", Radix::Hexadecimal)]
    #[case::hex("0XDEADBEEF", Radix::Hexadecimal)]
    #[case::hex("0xCAFEbabe", Radix::Hexadecimal)]

    #[case::octal("000", Radix::Octal)]
    #[case::octal("01234567", Radix::Octal)]
    #[case::octal("012300", Radix::Octal)]

    #[case::binary("0b01010010010", Radix::Binary)]
    #[case::binary("0b001001001", Radix::Binary)]
    fn test_tokenizing_valid_integers(#[case] src: &str, #[case] base: Radix) {
        let lexer = Lexer::new(src);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        let expected_tokens = Ok(vec![
            Token { token_type: IntConstant(src.trim(), base), location: Location { line: 1, column: 1 } },
        ]);
        assert_eq!(tokens, expected_tokens,
                   "lexing identifier {}: expected: {:?}, actual:{:?}",
                   src, expected_tokens, tokens);
    }

    #[test]
    fn test_bad_identifier_error() {
        let src = "abcd@";
        let lexer = Lexer::new(src);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        let expected = Err(LexerError::InvalidIdentifierCharacter {
            location: Location { line: 1, column: 5 },
            ch: '@',
        });
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_bad_lexer() {
        let src = "return 0@1;";
        let lexer = Lexer::new(src);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert!(tokens.is_err());
    }

    #[test]
    fn test_bad_number() {
        let src = "1foo";
        let lexer = Lexer::new(src);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert!(tokens.is_err());
    }

    #[test]
    fn test_comment_with_anything() {
        let src = "()//comment@bad\nabcde";
        let lexer = Lexer::new(src);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        let expected = Ok(vec![
            Token {
                token_type: OpenParentheses,
                location: Location { line: 1, column: 1 },
            },
            Token {
                token_type: CloseParentheses,
                location: Location { line: 1, column: 2 },
            },
            Token {
                token_type: Identifier("abcde"),
                location: Location { line: 2, column: 1 },
            }
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_block_comment() {
        let src = "abcde/*hello*/xyz";
        let lexer = Lexer::new(src);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        let expected = Ok(vec![
            Token {
                token_type: Identifier("abcde"),
                location: Location { line: 1, column: 1 },
            },
            Token {
                token_type: Identifier("xyz"),
                location: Location { line: 1, column: 15 },
            }
        ]);
        assert_eq!(tokens, expected);
    }
}