use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::iter::Peekable;
use std::str::Chars;

use once_cell::sync::Lazy;
use thiserror::Error;

use crate::common::{Location, Radix};

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub enum KeywordIdentifier {
    TypeInt,
    TypeVoid,
    Return,
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

    OperatorDiv,
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
            TokenType::OperatorDiv => TokenTag::OperatorDiv,
        }
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum LexerError {
    #[error("{location:?}: invalid character {ch:?} for identifier")]
    InvalidIdentifierCharacter { location: Location, ch: char },
}

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
            TokenType::OperatorDiv => f.write_str("/"),
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
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            lexer_mode: LexerMode::Default,
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
                            }
                            return Ok(Some(Token {
                                token_type: TokenType::OperatorDiv,
                                location: div_loc,
                            }))
                        }
                        '0'..='9' => {
                            let token = self.tokenize_integer_constant()?;
                            return Ok(Some(token));
                        }
                        '\n' | '\t' | ' ' => {
                            self.next_char();
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

    use crate::lexer::{KEYWORDS, Lexer, LexerError, Location, Radix, Token, TokenType};
    use crate::lexer::TokenType::Keyword;

    type LexerResult<T> = Result<T, LexerError>;

    #[test]
    fn test_tokenizing_open_and_close_parentheses() {
        let source = "()";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::OpenParentheses, location: Location { line: 1, column: 1 } },
            Token { token_type: TokenType::CloseParentheses, location: Location { line: 1, column: 2 } },
        ]));
    }

    #[test]
    fn test_tokenizing_open_and_close_braces() {
        let source = "{}";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::OpenBrace, location: Location { line: 1, column: 1 } },
            Token { token_type: TokenType::CloseBrace, location: Location { line: 1, column: 2 } },
        ]));
    }

    #[test]
    fn test_tokenizing_with_newlines() {
        let source = "(\n\n)";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::OpenParentheses, location: Location { line: 1, column: 1 } },
            Token { token_type: TokenType::CloseParentheses, location: Location { line: 3, column: 1 } },
        ]));
    }

    #[test]
    fn test_tokenizing_with_tabs() {
        let source = "(\t)";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::OpenParentheses, location: Location { line: 1, column: 1 } },
            Token { token_type: TokenType::CloseParentheses, location: Location { line: 1, column: 8 } },
        ]));
    }

    #[test]
    fn test_tokenizing_with_whitespaces() {
        let source = "(   )";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::OpenParentheses, location: Location { line: 1, column: 1 } },
            Token { token_type: TokenType::CloseParentheses, location: Location { line: 1, column: 5 } },
        ]));
    }

    #[test]
    fn test_tokenizing_semicolon() {
        let source = ";";
        let lexer = Lexer::new(source);
        let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
        assert_eq!(tokens, Ok(vec![
            Token { token_type: TokenType::Semicolon, location: Location { line: 1, column: 1 } },
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
                Token { token_type: TokenType::Identifier(src), location: Location { line: 1, column: 1 } },
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
            (Radix::Decimal, integers_base10),
            (Radix::Hexadecimal, integers_hex),
            (Radix::Octal, integers_octal),
            (Radix::Binary, integers_binary),
        ]);


        for (base, srcs) in int_tests.into_iter() {
            for src in srcs.into_iter() {
                let lexer = Lexer::new(src);
                let tokens: LexerResult<Vec<Token>> = lexer.into_iter().collect();
                let expected_tokens = Ok(vec![
                    Token { token_type: TokenType::IntConstant(src.trim(), base), location: Location { line: 1, column: 1 } },
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
                token_type: TokenType::OpenParentheses,
                location: Location { line: 1, column: 1 },
            },
            Token {
                token_type: TokenType::CloseParentheses,
                location: Location { line: 1, column: 2 },
            },
            Token {
                token_type: TokenType::Identifier("abcde"),
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
                token_type: TokenType::Identifier("abcde"),
                location: Location { line: 1, column: 1 },
            },
            Token {
                token_type: TokenType::Identifier("xyz"),
                location: Location { line: 1, column: 15 },
            }
        ]);
        assert_eq!(tokens, expected);
    }
}