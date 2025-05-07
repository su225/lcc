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
            w.write_fmt(format_args!(
                "    movq {src_operand}, {dst_operand}\n",
                src_operand = s, dst_operand = d))?
        },
        AsmInstruction::Ret => w.write_fmt(format_args!("    ret"))?,
        AsmInstruction::Unary { .. } => todo!(),
        AsmInstruction::AllocateStack(_) => todo!(),
    };
    Ok(())
}