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

fn emit_instruction<W: Write>(instr: &AsmInstruction, w: &mut W) -> io::Result<()> {
    match instr {
        AsmInstruction::Mov { src, dst } => {
            let s = src.to_string();
            let d = dst.to_string();
            w.write_fmt(format_args!("    movl {src_operand}, {dst_operand}\n",
                src_operand = s,
                dst_operand = d))?
        },

        AsmInstruction::Ret => {
            emit_function_epilogue(w)?;
            w.write_fmt(format_args!("    ret\n"))?
        },

        AsmInstruction::Unary { op, dst } =>
            w.write_fmt(format_args!("    {unary_op} {operand}\n",
                unary_op = op.to_string(),
                operand = dst.to_string())),

        AsmInstruction::AllocateStack(stack_size) =>
            w.write_fmt(format_args!("    subq ${stack_size}, %rsp\n",
                stack_size = stack_size)),
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