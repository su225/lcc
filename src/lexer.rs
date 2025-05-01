use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::iter::Peekable;
use std::str::Chars;

use once_cell::sync::Lazy;
use thiserror::Error;

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub enum KeywordIdentifier {
    TypeInt,
    TypeVoid,
    Return,
}

type Radix = u8;

const DECIMAL: Radix = 10;
const OCTAL: Radix = 8;
const HEXADECIMAL: Radix = 16;
const BINARY: Radix = 2;

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
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Location {
    line: usize,
    column: usize,
}

impl Location {
    fn advance_line(&mut self) {
        self.line += 1;
        self.column = 1;
    }

    fn advance_tab(&mut self) {
        self.column = ((self.column + 7) / 8) * 8;
    }

    fn advance(&mut self) {
        self.column += 1;
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum LexerError {
    #[error("{location:?}: unexpected character")]
    UnexpectedCharacter { location: Location },

    #[error("{location:?}: unknown radix representation of integer")]
    UnknownRadixRepresentation { location: Location },

    #[error("{location:?}: invalid digit {digit:?} for radix {radix:?}")]
    InvalidDigitForRadix { location: Location, digit: char, radix: Radix },

    #[error("{location:?}: invalid character {ch:?} for identifier")]
    InvalidIdentifierCharacter { location: Location, ch: char },
}

type LexerResult<T> = Result<T, LexerError>;

static KEYWORDS: Lazy<HashMap<&'static str, KeywordIdentifier>> = Lazy::new(|| {
    HashMap::from([
        ("int", KeywordIdentifier::TypeInt),
        ("void", KeywordIdentifier::TypeVoid),

        ("return", KeywordIdentifier::Return),
    ])
});

static KEYWORD_STRINGS: Lazy<HashMap<KeywordIdentifier, &'static str>> = Lazy::new(|| {
    HashMap::from([
        (KeywordIdentifier::TypeInt, "int"),
        (KeywordIdentifier::TypeVoid, "void"),

        (KeywordIdentifier::Return, "return"),
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
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Token<'a> {
    token_type: TokenType<'a>,
    location: Location,
}

pub struct Lexer<'a> {
    input: &'a str,
    char_stream: Peekable<Chars<'a>>,
    cur_stream_pos: usize,
    cur_location: Location,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            char_stream: input.chars().peekable(),
            cur_stream_pos: 0,
            cur_location: Location { line: 1, column: 1 },
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

    fn tokenize_single_char(&mut self, token_type: TokenType<'a>) -> Token<'a>  {
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
        let mut expected_radix = OCTAL;
        if first_digit != '0' {
            expected_radix = DECIMAL;
        } else {
            let second_digit = self.next_char();
            if second_digit.is_none() {
                return Ok(Token { location: start_loc, token_type: TokenType::IntConstant("0", DECIMAL) });
            }
            let digit2 = second_digit.unwrap();
            if digit2 == 'x' || digit2 == 'X' {
                expected_radix = HEXADECIMAL;
            } else if digit2 == 'b' || digit2 == 'B' {
                expected_radix = BINARY;
            } else if !digit2.is_digit(OCTAL as u32) {
                // If it is neither 0x, 0X, 0b, 0B or 0[Octal digit]
                // then it is an invalid number. Hence, it is an error.
                return Ok(Token { location: start_loc, token_type: TokenType::IntConstant("0", DECIMAL) });
            }
        }
        // Once we have determined the radix, we iterate through the digits as long
        // as we find a valid stop point: semicolon, whitespace (space, new-line, tab), end.
        // If we stop at characters that are not valid digits, then we throw an error. We
        // don't support digit grouping yet to keep the lexer simple.
        loop {
            let loc = self.cur_location.clone();
            let next = self.char_stream.peek();
            if next.is_none() {
                break;
            }
            let n = next.unwrap().clone();
            if !n.is_digit(expected_radix as u32) {
                break;
            }
            self.next_char();
        }
        return Ok(Token {
            location: start_loc,
            token_type: TokenType::IntConstant(&self.input[start_pos..self.cur_stream_pos], expected_radix),
        })
    }

    fn tokenize_identifier_or_keyword(&mut self) -> Result<Token<'a>, LexerError> {
        let start_loc = self.cur_location;
        let start_pos = self.cur_stream_pos;
        let first_char = self.next_char().unwrap();
        if first_char != '_' && !first_char.is_alphabetic() {
            return Err(LexerError::InvalidIdentifierCharacter{ location: start_loc, ch: first_char });
        }
        loop {
            let cur_loc = self.cur_location;
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
            None => Ok(Token{
                location: start_loc,
                token_type: TokenType::Identifier(word),
            }),
            Some(k) => Ok(Token{
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
            match *cur_char {
                ';' => {
                    let token = self.tokenize_single_char(TokenType::Semicolon);
                    return Ok(Some(token));
                },
                '(' => {
                    let token = self.tokenize_single_char(TokenType::OpenParentheses);
                    return Ok(Some(token));
                },
                ')' => {
                    let token = self.tokenize_single_char(TokenType::CloseParentheses);
                    return Ok(Some(token));
                },
                '{' => {
                    let token = self.tokenize_single_char(TokenType::OpenBrace);
                    return Ok(Some(token));
                },
                '}' => {
                    let token = self.tokenize_single_char(TokenType::CloseBrace);
                    return Ok(Some(token));
                },
                '0'..='9' => {
                    let token = self.tokenize_integer_constant()?;
                    return Ok(Some(token));
                },
                '\n' | '\t' | ' ' => {
                    self.next_char();
                    continue;
                },
                _ => {
                    let token = self.tokenize_identifier_or_keyword()?;
                    return Ok(Some(token));
                },
            }
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token<'a>, LexerError>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_tok = self.next_token();
        match next_tok {
            Ok(Some(tok)) => Some(Ok(tok)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use crate::lexer::{BINARY, DECIMAL, HEXADECIMAL, KEYWORDS, Lexer, LexerError, LexerResult, Location, OCTAL, Token, TokenType};
    use crate::lexer::TokenType::Keyword;

    #[test]
    fn test_tokenizing_open_and_close_parentheses() {
        let source = "()";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::OpenParentheses, location: Location { line: 1, column: 1 }},
            Token { token_type: TokenType::CloseParentheses, location: Location { line: 1, column: 2 }},
        ]));
    }

    #[test]
    fn test_tokenizing_open_and_close_braces() {
        let source = "{}";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::OpenBrace, location: Location { line: 1, column: 1 }},
            Token { token_type: TokenType::CloseBrace, location: Location { line: 1, column: 2 }},
        ]));
    }

    #[test]
    fn test_tokenizing_with_newlines() {
        let source = "(\n\n)";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::OpenParentheses, location: Location { line: 1, column: 1 }},
            Token { token_type: TokenType::CloseParentheses, location: Location { line: 3, column: 1 }},
        ]));
    }

    #[test]
    fn test_tokenizing_with_tabs() {
        let source = "(\t)";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::OpenParentheses, location: Location { line: 1, column: 1 }},
            Token { token_type: TokenType::CloseParentheses, location: Location { line: 1, column: 8 }},
        ]));
    }

    #[test]
    fn test_tokenizing_with_whitespaces() {
        let source = "(   )";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::OpenParentheses, location: Location { line: 1, column: 1 }},
            Token { token_type: TokenType::CloseParentheses, location: Location { line: 1, column: 5 }},
        ]));
    }

    #[test]
    fn test_tokenizing_semicolon() {
        let source = ";";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::Semicolon, location: Location { line: 1, column: 1 }},
        ]));
    }

    #[test]
    fn test_tokenizing_valid_identifiers() {
        let identifiers = vec![
            "abcde",
            "abcde123",
            "hello_world_123",
            "helloWorld123",
            "_abcde",
            "_123",
            "_123_456",
            "café",
            "αριθμός",
            "число",
            "数字",
        ];
        for src in identifiers.into_iter() {
            let lexer = Lexer::new(src);
            let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
            let expected_tokens: LexerResult<Vec<Token>> = Ok(vec![
                Token { token_type: TokenType::Identifier(src), location: Location { line: 1, column: 1 }},
            ]);
            assert_eq!(tokens, expected_tokens,
                       "lexing identifier {}: expected: {:?}, actual:{:?}",
                        src, expected_tokens, tokens);
        }
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
    fn test_tokenizing_valid_integers() {
        let integers_base10 = vec![
            "1",
            "0",
            "2",
            "3",
            "44",
            "55",
            "66",
            "777",
            "888",
            "9189",
            "189087931798698368761873",
            "0\t\t",
            "0  ",
            "0\n",
        ];

        let integers_hex = vec!["0xdeadbeef",
            "0Xcafebabe",
            "0XDEADBEEF",
            "0xCAFEbabe",
        ];

        let integers_octal = vec![
            "000",
            "01234567",
            "012300",
        ];

        let integers_binary = vec![
            "0b01010010010",
            "0b001001001",
        ];

        let int_tests = HashMap::from([
            (10, integers_base10),
            (16, integers_hex),
            (8, integers_octal),
            (2, integers_binary),
        ]);


        for (base, srcs) in int_tests.into_iter() {
            for src in srcs.into_iter() {
                let lexer = Lexer::new(src);
                let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
                let expected_tokens = Ok(vec![
                    Token { token_type: TokenType::IntConstant(src.trim(), base), location: Location { line: 1, column: 1 }},
                ]);
                assert_eq!(tokens, expected_tokens,
                           "lexing identifier {}: expected: {:?}, actual:{:?}",
                           src, expected_tokens, tokens);
            }
        }
    }

    #[test]
    fn test_bad_identifier_error() {
        let src = "abcd@";
        let lexer = Lexer::new(src);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        let expected = Err(LexerError::InvalidIdentifierCharacter {
            location: Location {line: 1, column: 5},
            ch: '@',
        });
        assert_eq!(tokens, expected);
    }
}