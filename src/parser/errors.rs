use thiserror::Error;
use crate::common::Location;
use crate::lexer::{KeywordIdentifier, LexerError, TokenTag};

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