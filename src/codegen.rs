use std::fmt::{Display, Formatter};
use std::num::ParseIntError;

use thiserror::Error;

use crate::parser::{Expression, ExpressionKind, FunctionDefinition, ProgramDefinition, Statement, StatementKind};

#[derive(Debug)]
pub enum Register {
    RAX,
    RBX,
    RCX,
    RDX,
}

impl Display for Register {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", match &self {
            Register::RAX => "rax",
            Register::RBX => "rbx",
            Register::RCX => "rcx",
            Register::RDX => "rdx",
        })
    }
}

#[derive(Debug)]
pub enum Operand {
    Imm(i64),
    Register(Register),
}

impl Display for Operand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match &self {
            Operand::Imm(n) => format!("${}", n),
            Operand::Register(r) => r.to_string(),
        })
    }
}

#[derive(Debug)]
pub enum AsmInstruction {
    Mov { src: Operand, dst: Operand },
    Ret,
}

#[derive(Debug)]
pub struct AsmFunction<'a> {
    pub name: &'a str,
    pub instructions: Vec<AsmInstruction>,
}

#[derive(Debug)]
pub struct AsmProgram<'a> {
    pub functions: Vec<AsmFunction<'a>>,
}

#[derive(Error, Debug)]
pub enum CodegenError {
    #[error(transparent)]
    IntImmediateParseError(#[from] ParseIntError),
}

pub fn generate_assembly<'a>(p: ProgramDefinition<'a>) -> Result<AsmProgram<'a>, CodegenError> {
    let mut asm_functions: Vec<AsmFunction<'a>> = vec![];
    for func in p.functions.iter() {
        let asm = generate_function_assembly(func)?;
        asm_functions.push(asm)
    }
    Ok(AsmProgram { functions: asm_functions })
}

fn generate_function_assembly<'a>(f: &FunctionDefinition<'a>) -> Result<AsmFunction<'a>, CodegenError> {
    let mut body_statements: Vec<AsmInstruction> = vec![];
    for stmt in f.body.iter() {
        let instrs_for_statement = generate_statement_assembly(stmt)?;
        body_statements.extend(instrs_for_statement);
    }
    Ok(AsmFunction {
        name: f.name.name,
        instructions: body_statements,
    })
}

fn generate_statement_assembly(s: &Statement<'_>) -> Result<Vec<AsmInstruction>, CodegenError> {
    match &s.kind {
        StatementKind::Return(expr) => {
            let (mut expr_instrs, res_operand) = generate_expression_assembly(&expr)?;
            expr_instrs.push(AsmInstruction::Mov {
                src: res_operand,
                dst: Operand::Register(Register::RAX),
            });
            expr_instrs.push(AsmInstruction::Ret);
            Ok(expr_instrs)
        }
    }
}

fn generate_expression_assembly(e: &Expression<'_>) -> Result<(Vec<AsmInstruction>, Operand), CodegenError> {
    match e.kind {
        ExpressionKind::IntConstant(num, radix) => {
            let n = i64::from_str_radix(num, radix.value())?;
            Ok((
                vec![
                    AsmInstruction::Mov {
                        src: Operand::Imm(n),
                        dst: Operand::Register(Register::RAX),
                    }
                ],
                Operand::Register(Register::RAX),
            ))
        }
    }
}