
#[derive(Debug, Eq, PartialEq)]
pub enum KeywordIdentifier {
    TypeInt,
    TypeVoid,
    Return,
}

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

#[derive(Debug, Eq, PartialEq)]
pub struct Token {
    token_type: TokenType,
    line: usize,
    column: usize,
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
        ])
    }

    #[test]
    fn test_tokenizing_open_and_close_braces() {
        let source = "{}".to_string();
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenBrace, line: 1, column: 1 },
            Token { token_type: TokenType::CloseBrace, line: 1, column: 2 },
        ])
    }

    #[test]
    fn test_tokenizing_with_newlines() {
        let source = "(\n\n)".to_string();
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenParentheses, line: 1, column: 1 },
            Token { token_type: TokenType::CloseParentheses, line: 3, column: 1 },
        ])
    }

    #[test]
    fn test_tokenizing_with_tabs() {
        let source = "(\t)".to_string();
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenParentheses, line: 1, column: 1 },
            Token { token_type: TokenType::CloseParentheses, line: 1, column: 8 },
        ])
    }

    #[test]
    fn test_tokenizing_with_whitespaces() {
        let source = "(   )".to_string();
        let lexer = Lexer::new(source);
        let tokens = lexer.into_iter().collect::<Vec<Token>>();
        assert_eq!(tokens, vec![
            Token { token_type: TokenType::OpenParentheses, line: 1, column: 1 },
            Token { token_type: TokenType::CloseParentheses, line: 1, column: 5 },
        ])
    }
}