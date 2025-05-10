use std::io;
use std::io::Write;
use crate::codegen::{AsmFunction, AsmInstruction, AsmProgram};

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
        AsmInstruction::Mov32 { src, dst } => emit_instruction!(w, "movl {src} {dst}")?,
        AsmInstruction::Ret => {
            emit_function_epilogue(w)?;
            emit_instruction!(w, "ret")?
        },
        AsmInstruction::Not32 { op } => emit_instruction!(w, "notl {op}")?,
        AsmInstruction::Neg32 { op } => emit_instruction!(w, "negl {op}")?,
        AsmInstruction::AllocateStack(stack_size) => emit_instruction!(w, "subq %{stack_size}, %rsp")?,
        AsmInstruction::Add32 { src, dst } => emit_instruction!(w, "addl {src}, {dst}")?,
        AsmInstruction::Sub32 { src, dst } => emit_instruction!(w, "subl {src}, {dst}")?,
        AsmInstruction::IMul32 { src, dst } => emit_instruction!(w, "imull {src}, {dst}")?,
        AsmInstruction::IDiv32 { divisor } => emit_instruction!(w, "idivl {divisor}")?,
        AsmInstruction::Cdq => emit_instruction!(w, "cdq")?,
    };
    Ok(())
}

fn emit_function_prologue<W: Write>(w: &mut W) -> io::Result<()> {
    w.write_fmt(format_args!("    pushq %rbp\n"))?;
    w.write_fmt(format_args!("    movq %rsp, %rbp\n"))?;
    Ok(())
}

fn emit_function_epilogue<W: Write>(w: &mut W) -> io::Result<()> {
    w.write_fmt(format_args!("    movq %rbp, %rsp\n"))?;
    w.write_fmt(format_args!("    popq %rbp\n"))?;
    Ok(())
}