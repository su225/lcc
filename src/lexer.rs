use std::collections::HashMap;
use std::fmt::{Display, Formatter};
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


#[derive(Debug, Eq, PartialEq)]
pub enum TokenType {
    Keyword(KeywordIdentifier),
    OpenParentheses,
    CloseParentheses,
    OpenBrace,
    CloseBrace,
    Semicolon,
    Identifier(String),
    IntConstant(String),
    Unknown,
}

impl Display for TokenType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            TokenType::Keyword(k) => f.write_str(KEYWORD_STRINGS.get(k).unwrap()),
            TokenType::OpenParentheses => f.write_str("("),
            TokenType::CloseParentheses => f.write_str(")"),
            TokenType::OpenBrace => f.write_str("{"),
            TokenType::CloseBrace => f.write_str("}"),
            TokenType::Semicolon => f.write_str(";"),
            TokenType::Identifier(x) => f.write_fmt(format_args!("identifier:{}", x)),
            TokenType::IntConstant(x) => f.write_fmt(format_args!("int:{}", x)),
            TokenType::Unknown => f.write_str("unknown"),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Token {
    token_type: TokenType,
    line: usize,
    column: usize,
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("<line:{}, col:{}, token:{}>", self.line, self.column, self.token_type))
    }
}

struct Lexer {
    input: String,
    cur_pos: usize,
    cur_line: usize,
    cur_col: usize,
}

impl Lexer {
    fn new(input: String) -> Self {
        Self {
            input,
            cur_pos: 0,
            cur_line: 1,
            cur_col: 1,
        }
    }
    
    fn make_token_at_current_position(&self, token_type: TokenType) -> Token {
        Token {
            token_type,
            line: self.cur_line,
            column: self.cur_col,
        }
    }

    fn next_token(&mut self) -> Option<Token> {
        let input = &self.input[self.cur_pos..];
        for (i, ch) in input.char_indices() {
            self.cur_pos += ch.len_utf8();
            match ch {
                '(' => {
                    let result = self.make_token_at_current_position(TokenType::OpenParentheses);
                    self.cur_col += 1;
                    return Some(result);
                },
                ')' => {
                    let result = self.make_token_at_current_position(TokenType::CloseParentheses);
                    self.cur_col += 1;
                    return Some(result);
                },
                '{' => {
                    let result = self.make_token_at_current_position(TokenType::OpenBrace);
                    self.cur_col += 1;
                    return Some(result);
                },
                '}' => {
                    let result = self.make_token_at_current_position(TokenType::CloseBrace);
                    self.cur_col += 1;
                    return Some(result);
                },
                ';' => {
                    let result = self.make_token_at_current_position(TokenType::Semicolon);
                    self.cur_col += 1;
                    return Some(result);
                },
                '\n' => {
                    self.cur_line += 1;
                    self.cur_col = 1;
                    continue
                },
                '\t' => {
                    self.cur_col = ((self.cur_col + 7) / 8) * 8;
                    continue
                },
                ' ' => {
                    self.cur_col += 1;
                    continue
                },
                _ => {
                    let result = self.make_token_at_current_position(TokenType::Unknown);
                    self.cur_col += 1;
                    return Some(result);
                },
            };
        }
        None
    }
}

impl Iterator for Lexer {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

#[cfg(test)]
mod test {
    use crate::lexer::{Lexer, Token, TokenType};

    #[test]
    fn test_tokenizing_open_and_close_parentheses() {
        let source = "()".to_string();
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenParentheses, line: 1, column: 1},
            Token { token_type: TokenType::CloseParentheses, line: 1, column: 2 },
        ]);
    }

    #[test]
    fn test_tokenizing_open_and_close_braces() {
        let source = "{}".to_string();
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenBrace, line: 1, column: 1 },
            Token { token_type: TokenType::CloseBrace, line: 1, column: 2 },
        ]);
    }

    #[test]
    fn test_tokenizing_with_newlines() {
        let source = "(\n\n)".to_string();
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenParentheses, line: 1, column: 1 },
            Token { token_type: TokenType::CloseParentheses, line: 3, column: 1 },
        ]);
    }

    #[test]
    fn test_tokenizing_with_tabs() {
        let source = "(\t)".to_string();
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenParentheses, line: 1, column: 1 },
            Token { token_type: TokenType::CloseParentheses, line: 1, column: 8 },
        ]);
    }

    #[test]
    fn test_tokenizing_with_whitespaces() {
        let source = "(   )".to_string();
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenParentheses, line: 1, column: 1 },
            Token { token_type: TokenType::CloseParentheses, line: 1, column: 5 },
        ]);
    }

    #[test]
    fn test_tokenizing_semicolon() {
        let source = ";".to_string();
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
            let lexer = Lexer::new(src.to_string());
            let tokens = lexer.into_iter().collect::<Vec<Token>>();
            let expected_tokens = vec![
                Token { token_type: TokenType::Identifier(src.to_string()), line: 1, column: 1 }
            ];
            assert_eq!(tokens, expected_tokens,
                       "lexing identifier {}: expected: {:?}, actual:{:?}",
                        src, expected_tokens, tokens);
        }
    }

    #[test]
    fn test_tokenizing_valid_integers() {
        let integers = vec![
            "1",
            "0",
            "100",
            "189087931798698368761873",
            "0xdeadbeef",
            "-1000",
        ];
        for src in integers.into_iter() {
            let lexer = Lexer::new(src.to_string());
            let tokens = lexer.into_iter().collect::<Vec<Token>>();
            let expected_tokens = vec![
                Token { token_type: TokenType::IntConstant(src.to_string()), line: 1, column: 1 }
            ];
            assert_eq!(tokens, expected_tokens,
                       "lexing identifier {}: expected: {:?}, actual:{:?}",
                       src, expected_tokens, tokens);
        }
    }
}