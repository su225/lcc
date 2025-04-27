use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::iter::Peekable;
use std::str::Chars;
use once_cell::sync::Lazy;

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub enum KeywordIdentifier {
    TypeInt,
    TypeVoid,
    Return,
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

type Radix = u8;

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
    Unknown,
}

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
            TokenType::Unknown => f.write_str("unknown"),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Token<'a> {
    token_type: TokenType<'a>,
    line: usize,
    column: usize,
}

impl<'a> Display for Token<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("<line:{}, col:{}, token:{}>", self.line, self.column, self.token_type))
    }
}

struct Lexer<'a> {
    input: &'a str,
    char_stream: Peekable<Chars<'a>>,
    cur_line: usize,
    cur_col: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            char_stream: input.chars().peekable(),
            cur_line: 1,
            cur_col: 1,
        }
    }

    fn tokenize_single_char(&mut self, token_type: TokenType<'a>) -> Token<'a>  {
        let token = Token {
            token_type,
            line: self.cur_line,
            column: self.cur_col,
        };
        self.char_stream.next();
        self.cur_col += 1;
        token
    }

    fn tokenize_integer_constant(&mut self) -> Token<'a> {
        todo!("tokenize integer constant with radix")
    }

    fn next_token(&mut self) -> Option<Token<'a>> {
        loop {
            let cur = self.char_stream.peek();
            if cur.is_none() {
                // We have reached the end of the stream. Hence, we cannot
                // tokenize anymore. So we just return None
                return None;
            }
            let cur_char = cur.unwrap();
            match *cur_char {
                ';' => {
                    let token = self.tokenize_single_char(TokenType::Semicolon);
                    return Some(token);
                },
                '(' => {
                    let token = self.tokenize_single_char(TokenType::OpenParentheses);
                    return Some(token);
                },
                ')' => {
                    let token = self.tokenize_single_char(TokenType::CloseParentheses);
                    return Some(token);
                },
                '{' => {
                    let token = self.tokenize_single_char(TokenType::OpenBrace);
                    return Some(token);
                },
                '}' => {
                    let token = self.tokenize_single_char(TokenType::CloseBrace);
                    return Some(token);
                },
                '0'..'9' => {
                    let token = self.tokenize_integer_constant();
                    return Some(token);
                },
                '\n' => {
                    self.cur_line += 1;
                    self.cur_col = 1;
                    self.char_stream.next();
                    continue;
                },
                '\t' => {
                    self.cur_col = ((self.cur_col + 7) / 8) * 8;
                    self.char_stream.next();
                    continue;
                },
                ' ' => {
                    self.cur_col += 1;
                    self.char_stream.next();
                    continue;
                }
                _ => {
                    return Some(self.tokenize_single_char(TokenType::Unknown));
                },
            }
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use crate::lexer::{Lexer, Token, TokenType};

    #[test]
    fn test_tokenizing_open_and_close_parentheses() {
        let source = "()";
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenParentheses, line: 1, column: 1},
            Token { token_type: TokenType::CloseParentheses, line: 1, column: 2 },
        ]);
    }

    #[test]
    fn test_tokenizing_open_and_close_braces() {
        let source = "{}";
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenBrace, line: 1, column: 1 },
            Token { token_type: TokenType::CloseBrace, line: 1, column: 2 },
        ]);
    }

    #[test]
    fn test_tokenizing_with_newlines() {
        let source = "(\n\n)";
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenParentheses, line: 1, column: 1 },
            Token { token_type: TokenType::CloseParentheses, line: 3, column: 1 },
        ]);
    }

    #[test]
    fn test_tokenizing_with_tabs() {
        let source = "(\t)";
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenParentheses, line: 1, column: 1 },
            Token { token_type: TokenType::CloseParentheses, line: 1, column: 8 },
        ]);
    }

    #[test]
    fn test_tokenizing_with_whitespaces() {
        let source = "(   )";
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenParentheses, line: 1, column: 1 },
            Token { token_type: TokenType::CloseParentheses, line: 1, column: 5 },
        ]);
    }

    #[test]
    fn test_tokenizing_semicolon() {
        let source = ";";
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::Semicolon, line: 1, column: 1 },
        ]);
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
        ];
        for src in identifiers.into_iter() {
            let lexer = Lexer::new(src);
            let tokens = lexer.into_iter().collect::<Vec<Token>>();
            let expected_tokens = vec![
                Token { token_type: TokenType::Identifier(src), line: 1, column: 1 }
            ];
            assert_eq!(tokens, expected_tokens,
                       "lexing identifier {}: expected: {:?}, actual:{:?}",
                        src, expected_tokens, tokens);
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
                let tokens = lexer.into_iter().collect::<Vec<Token>>();
                let expected_tokens = vec![
                    Token { token_type: TokenType::IntConstant(src, base), line: 1, column: 1 }
                ];
                assert_eq!(tokens, expected_tokens,
                           "lexing identifier {}: expected: {:?}, actual:{:?}",
                           src, expected_tokens, tokens);
            }
        }
    }
}