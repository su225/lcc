use serde::Serialize;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Location {
    pub(crate) line: usize,
    pub(crate) column: usize,
}

impl From<(usize, usize)> for Location {
    fn from(value: (usize, usize)) -> Self {
        Location { line: value.0, column: value.1 }
    }
}

impl Location {
    pub(crate) fn advance_line(&mut self) {
        self.line += 1;
        self.column = 1;
    }

    pub(crate) fn advance_tab(&mut self) {
        self.column = ((self.column + 7) / 8) * 8;
    }

    pub(crate) fn advance(&mut self) {
        self.column += 1;
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Radix {
    Binary,
    Octal,
    Decimal,
    Hexadecimal,
}