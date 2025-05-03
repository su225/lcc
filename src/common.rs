#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Location {
    pub(crate) line: usize,
    pub(crate) column: usize,
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

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub enum Radix {
    Binary,
    Octal,
    Decimal,
    Hexadecimal,
}