use std::num::ParseIntError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodegenError {
    #[error(transparent)]
    IntImmediateParseError(#[from] ParseIntError),
}