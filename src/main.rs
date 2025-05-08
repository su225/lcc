use std::{fs, io};
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::process::{Command, ExitStatus};

use clap::Parser as ClapParser;
use thiserror::Error;

use crate::codegen::CodegenError;
use crate::lexer::{Lexer, LexerError, Token};
use crate::parser::{Parser, ParserError};
use crate::tacky::{TackyError};

mod lexer;
mod parser;
mod common;
mod codegen;
mod asmgen;
mod tacky;

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

    /// Tacky is the intermediate representation
    /// The compiler stops at intermediate-representation generation
    #[arg(short = 'T', long = "tacky", long, default_value_t = false)]
    tacky: bool,

    /// Generate assembly code, but stop before emitting .S file
    /// for the assembler and linker.
    #[arg(short = 'C', long = "codegen", long, default_value_t = false)]
    codegen: bool,

    /// Generate and emit .S file for assembler and stop there
    #[arg(short = 'S', long = "emit-assembly", long, default_value_t = false)]
    emit_assembly: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let source_code = fs::read_to_string(&args.input_file)?;
    Ok(invoke_compiler_driver(&args, source_code)?)
}

#[derive(Error, Debug)]
enum CompilerDriverError {
    #[error("error in lexer: {0}")]
    LexerError(#[from] LexerError),

    #[error("error in parser: {0}")]
    ParserError(#[from] ParserError),

    #[error("error in tacky generation: {0}")]
    TackyGenerationError(#[from] TackyError),

    #[error("error while generating code: {0}")]
    CodeGeneratorError(#[from] CodegenError),

    #[error("error on writing assembly to {0}: {1}")]
    CodeEmitError(String, #[source] io::Error),

    #[error("error on invoking assembler: {0}")]
    SystemAssemblerInvocationError(#[source] io::Error),

    #[error("error from system assembler. exit-status={0}")]
    SystemAssemblerFailedError(ExitStatus),
}

/// invoke_compiler_driver invokes different compiler stages. Depending on
/// the flags, it may stop early in some stage
fn invoke_compiler_driver(args: &Args, source_code: String) -> Result<(), CompilerDriverError> {
    let lexer = Lexer::new(&source_code);
    if args.lex {
        let tokens = lexer.collect::<Result<Vec<Token>, LexerError>>()
            .map_err(|e| CompilerDriverError::LexerError(e))?;
        println!("{:#?}", tokens);
        return Ok(());
    }
    let mut parser = Parser::new(lexer);
    let ast = parser.parse()?;
    if args.parse {
        println!("{:#?}", ast);
        return Ok(());
    }
    let tacky = tacky::emit(&ast)?;
    if args.tacky {
        println!("{:#?}", tacky);
        return Ok(());
    }
    let asm_code = codegen::generate_assembly(tacky)?;
    if args.codegen {
        println!("{:#?}", asm_code);
        return Ok(());
    }
    let output_stem = args.input_file.strip_suffix(".c").unwrap_or(&args.input_file);
    let output_asm_file = format!("{}.s", output_stem);
    let output_file = &output_stem;

    if args.emit_assembly {
        asmgen::emit(asm_code, io::stdout()).
            map_err(|e| CompilerDriverError::CodeEmitError("<stdout>".to_string(), e))?;
        return Ok(());
    }

    OpenOptions::new()
        .create(true)
        .write(true)
        .open(&output_asm_file)
        .and_then(|f| asmgen::emit(asm_code, f))
        .map_err(|e| CompilerDriverError::CodeEmitError(output_asm_file.clone(), e))?;

    invoke_system_assembler(&output_file, &output_asm_file)
        .and_then(|assembler_status| {
            if assembler_status.success() {
                Ok(())
            } else {
                Err(CompilerDriverError::SystemAssemblerFailedError(assembler_status))
            }
        })
}

/// invoke_system_assembler invokes the assembler installed in the system for the assembly
/// code generated. In Mac OS X, this is actually clang.
fn invoke_system_assembler(output_file: &str, assembly_file: &str) -> Result<ExitStatus, CompilerDriverError> {
    Command::new("gcc")
        .args(["-o", output_file, assembly_file])
        .status()
        .map_err(|e| CompilerDriverError::SystemAssemblerInvocationError(e))
}