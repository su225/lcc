use clap::Parser;

/// C-compiler for learning real compiler construction
/// and the Rust programming language at the same time
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input source code for compilation
    input_file: String,

    /// Output object file name
    #[arg(short = 'o', long, default_value_t = String::new())]
    output: String,

    /// Stop compiler at Lexer (useful for debugging)
    #[arg(short = 'L', long, default_value_t = false)]
    lexer: bool,

    /// Stop compiler at Parsing stage
    /// (Runs Lexer and Parser)
    #[arg(short = 'P', long, default_value_t = false)]
    parser: bool,

    /// Generate assembly code, but stop before emitting .S file
    /// for the assembler and linker.
    #[arg(short = 'C', long, default_value_t = false)]
    codegen: bool,

    /// Generate and emit .S file for assembler
    #[arg(short = 'S', long, default_value_t = true)]
    emit_assembly: bool,
}

fn main() {
    let args = Args::parse();
    println!("{:?}", args);
}
