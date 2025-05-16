use std::io;
use std::io::Write;
use crate::codegen::x86_64::types::*;
use crate::codegen::x86_64::types::AsmInstruction::*;

pub fn emit<W: Write>(asm_code: AsmProgram, mut w: W) -> io::Result<()> {
    for f in asm_code.functions.into_iter() {
        emit_function(f, &mut w)?;
    }
    Ok(())
}

fn emit_function<W: Write>(f: AsmFunction, w: &mut W) -> io::Result<()> {
    w.write_fmt(format_args!("    .globl _{function_name}\n", function_name = f.name))?;
    w.write_fmt(format_args!("_{function_name}:\n", function_name = f.name))?;
    emit_function_prologue(w)?;
    for instr in f.instructions {
        emit_instruction(&instr, w)?;
    }
    Ok(())
}

macro_rules! emit_instruction {
    ($w:expr, $($arg:tt)*) => {
        writeln!($w, "    {}", format!($($arg)*))
    };
}

fn emit_instruction<W: Write>(instr: &AsmInstruction, w: &mut W) -> io::Result<()> {
    match instr {
        Mov8 { src, dst } => emit_instruction!(w, "movb {src}, {dst}")?,
        Mov32 { src, dst } => emit_instruction!(w, "movl {src}, {dst}")?,
        Ret => {
            emit_function_epilogue(w)?;
            emit_instruction!(w, "ret")?
        },
        Not32 { op } => emit_instruction!(w, "notl {op}")?,
        Neg32 { op } => emit_instruction!(w, "negl {op}")?,
        AllocateStack(stack_size) => emit_instruction!(w, "subq ${stack_size}, %rsp")?,
        Add32 { src, dst } => emit_instruction!(w, "addl {src}, {dst}")?,
        Sub32 { src, dst } => emit_instruction!(w, "subl {src}, {dst}")?,
        IMul32 { src, dst } => emit_instruction!(w, "imull {src}, {dst}")?,
        IDiv32 { divisor } => emit_instruction!(w, "idivl {divisor}")?,
        SignExtendTo64 => emit_instruction!(w, "cdq")?,

        And32 { src, dst } => emit_instruction!(w, "andl {src}, {dst}")?,
        Or32 { src, dst } => emit_instruction!(w, "orl {src}, {dst}")?,
        Xor32 { src, dst } => emit_instruction!(w, "xorl {src}, {dst}")?,
        Shl32 { src, dst } => emit_instruction!(w, "shll {src}, {dst}")?,
        Shr32 { src, dst } => emit_instruction!(w, "shrl {src}, {dst}")?,
        Sal32 { src, dst } => emit_instruction!(w, "sall {src}, {dst}")?,
        Sar32 { src, dst } => emit_instruction!(w, "sarl {src}, {dst}")?,
    };
    Ok(())
}

fn emit_function_prologue<W: Write>(w: &mut W) -> io::Result<()> {
    emit_instruction!(w, "pushq %rbp")?;
    emit_instruction!(w, "movq  %rsp, %rbp")?;
    Ok(())
}

fn emit_function_epilogue<W: Write>(w: &mut W) -> io::Result<()> {
    emit_instruction!(w, "movq %rbp, %rsp")?;
    emit_instruction!(w, "popq %rbp")?;
    Ok(())
}