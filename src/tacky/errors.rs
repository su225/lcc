use std::num::ParseIntError;
use thiserror::Error;

#[derive(Error, PartialEq, Debug)]
pub enum TackyError {
    #[error(transparent)]
    IntImmediateParseError(#[from] ParseIntError),
}