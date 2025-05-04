use std::error::Error;
use std::fs;
use clap::Parser as ClapParser;
use crate::lexer::{Lexer, LexerError, Token};
use crate::parser::Parser;

mod lexer;
mod parser;
mod common;

/// C-compiler for learning real compiler construction
/// and the Rust programming language at the same time
#[derive(ClapParser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input source code for compilation
    input_file: String,

    /// Output object file name
    #[arg(short = 'o', default_value_t = String::new())]
    output: String,

    /// Stop compiler at Lexer (useful for debugging)
    #[arg(short = 'L', long = "lex", default_value_t = false)]
    lex: bool,

    /// Stop compiler at Parsing stage
    /// (Runs Lexer and Parser)
    #[arg(short = 'P', long = "parse", long, default_value_t = false)]
    parse: bool,

    /// Generate assembly code, but stop before emitting .S file
    /// for the assembler and linker.
    #[arg(short = 'C', long = "codegen", long, default_value_t = false)]
    codegen: bool,

    /// Generate and emit .S file for assembler
    #[arg(short = 'S', long = "emit-assembly", long, default_value_t = true)]
    emit_assembly: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let source_code = fs::read_to_string(&args.input_file)?;
    Ok(invoke_compiler_driver(&args, source_code)?)
}

/// invoke_compiler_driver invokes different compiler stages. Depending on
/// the flags, it may stop early in some stage
fn invoke_compiler_driver(args: &Args, source_code: String) -> Result<(), Box<dyn Error>> {
    let lexer = Lexer::new(&source_code);
    if args.lex {
        let tokens: Result<Vec<Token>, LexerError> = lexer.collect();
        println!("{:#?}", tokens);
        if tokens.is_err() {
            return Err(format!("lexer error: {}", tokens.err().unwrap()).into());
        }
        return Ok(());
    }
    let mut parser = Parser::new(lexer);
    let ast = parser.parse();
    if args.parse {
        println!("{:#?}", ast);
        if ast.is_err() {
            return Err(format!("parser error: {}", ast.err().unwrap()).into());
        }
        return Ok(());
    }
    if args.codegen {
        return Ok(());
    }
    Ok(())
}
