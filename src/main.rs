use std::error::Error;
use std::fs;
use std::fs::{OpenOptions, Permissions};
use std::os::unix::fs::PermissionsExt;
use std::process::Command;
use clap::Parser as ClapParser;
use crate::lexer::{Lexer, LexerError, Token};
use crate::parser::Parser;

mod lexer;
mod parser;
mod common;
mod codegen;
mod code_emit;

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
    if ast.is_err() {
        println!("{:#?}", ast);
        return Err(format!("parser error: {}", ast.err().unwrap()).into());
    }
    if args.parse {
        println!("{:#?}", ast);
        return Ok(());
    }
    let asm_code = codegen::generate_assembly(ast.unwrap());
    if asm_code.is_err() {
        println!("{:#?}", asm_code);
        return Err(format!("code generation error: {}", asm_code.err().unwrap()).into());
    }
    if args.codegen {
        println!("{:#?}", asm_code);
        return Ok(());
    }
    let output_stem = args.input_file.strip_suffix(".c").unwrap_or(&args.input_file);
    let output_asm_file = format!("{}.s", output_stem);
    let output_file = &output_stem;
    let res = OpenOptions::new().create(true).write(true).open(&output_asm_file)
        .and_then(|f| code_emit::emit(asm_code.unwrap(), f));
    if res.is_err() {
        println!("{:?}", res);
        return Err(format!("error while writing assembly: {}", res.err().unwrap()).into());
    }
    invoke_system_assembler(&output_file, &output_asm_file)?;
    Ok(())
}

fn invoke_system_assembler(output_file: &str, assembly_file: &str) -> Result<(), Box<dyn Error>> {
    let status = Command::new("gcc")
        .args(["-o", output_file, assembly_file])
        .status()
        .expect("failed to execute gcc assembler");

    if !status.success() {
        return Err(format!("assembler failed with status: {}", status).into());
    }
    Ok(())
}