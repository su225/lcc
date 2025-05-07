use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::num::ParseIntError;

use thiserror::Error;

use crate::codegen::AsmInstruction::{Mov, Ret, Unary};
use crate::tacky::{Instruction, IRFunction, IRProgram, IRSymbol, IRUnaryOperator, IRValue};

#[derive(Debug, Clone)]
pub enum Register {
    AX,
    BX,
    CX,
    DX,

    EAX,
    EBX,
    ECX,
    EDX,

    RAX,
    RBX,
    RCX,
    RDX,

    R10,
}

impl Display for Register {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", match &self {
            Register::AX => "ax",
            Register::BX => "bx",
            Register::CX => "cx",
            Register::DX => "dx",

            Register::EAX => "eax",
            Register::EBX => "ebx",
            Register::ECX => "ecx",
            Register::EDX => "edx",

            Register::RAX => "rax",
            Register::RBX => "rbx",
            Register::RCX => "rcx",
            Register::RDX => "rdx",

            Register::R10 => "r10",
        })
    }
}

#[derive(Debug)]
pub enum AsmUnaryOperator {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub enum AsmOperand {
    Imm(i64),
    Register(Register),
    Pseudo(IRSymbol),
    Stack { stack_offset: isize },
}

impl Display for AsmOperand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match &self {
            AsmOperand::Imm(n) => format!("${}", n),
            AsmOperand::Register(r) => r.to_string(),
            AsmOperand::Pseudo(_) => todo!("implement for pseudo"),
            AsmOperand::Stack(_) => todo!("implement for stack"),
        })
    }
}

#[derive(Debug)]
pub enum AsmInstruction {
    Mov { src: AsmOperand, dst: AsmOperand },
    Unary { op: AsmUnaryOperator, dst: AsmOperand },
    AllocateStack(usize),
    Ret,
}

#[derive(Debug)]
pub struct AsmFunction {
    pub name: IRSymbol,
    pub instructions: Vec<AsmInstruction>,
}

#[derive(Debug)]
pub struct AsmProgram {
    pub functions: Vec<AsmFunction>,
}

#[derive(Error, Debug)]
pub enum CodegenError {
    #[error(transparent)]
    IntImmediateParseError(#[from] ParseIntError),
}

struct StackAllocationContext {
    cur_offset: isize,
    symbol_offset: HashMap<IRSymbol, isize>,
}

impl StackAllocationContext {
    fn new() -> Self {
        StackAllocationContext {
            cur_offset: 0,
            symbol_offset: HashMap::new(),
        }
    }
}

pub fn generate_assembly(p: IRProgram) -> Result<AsmProgram, CodegenError> {
    let mut asm_functions = Vec::with_capacity(p.functions.len());
    for f in p.functions {
        let asm_func = generate_function_assembly(f)?;
        let mut stack_alloc_ctx = StackAllocationContext::new();
        let stack_alloced = allocate_stack_frame(&stack_alloc_ctx, asm_func)?;
        asm_functions.push(stack_alloced);
    }
    Ok(AsmProgram { functions: asm_functions })
}

fn generate_function_assembly(f: IRFunction) -> Result<AsmFunction, CodegenError> {
    let mut asm_instructions = Vec::with_capacity(f.body.len());
    for tacky_inst in f.body {
        let asm_instrs = generate_instruction_assembly(tacky_inst)?;
        asm_instructions.extend(asm_instrs);
    }
    Ok(AsmFunction {
        name: f.identifier.clone(),
        instructions: asm_instructions,
    })
}

fn generate_instruction_assembly(ti: Instruction) -> Result<Vec<AsmInstruction>, CodegenError> {
    match ti {
        Instruction::Unary { operator, src, dst } => {
            let asm_dst_operand = from_ir_value(dst);
            Ok(vec![
                Mov { src: from_ir_value(src), dst: asm_dst_operand },
                Unary {
                    op: match operator {
                        IRUnaryOperator::Complement => AsmUnaryOperator::Not,
                        IRUnaryOperator::Negate => AsmUnaryOperator::Neg,
                    },
                    dst: asm_dst_operand,
                }
            ])
        },
        Instruction::Return(v) => {
            Ok(vec![
                Mov {
                    src: from_ir_value(v),
                    dst: AsmOperand::Register(Register::AX),
                },
                Ret,
            ])
        }
    }
}

fn allocate_stack_frame(ctx: &mut StackAllocationContext, f: AsmFunction) -> Result<AsmFunction, CodegenError> {
    todo!("allocate stack location to pseudo registers")
}

fn from_ir_value(v: IRValue) -> AsmOperand {
    match v {
        IRValue::Constant(c) => AsmOperand::Imm(c),
        IRValue::Variable(s) => AsmOperand::Pseudo(s),
    }
}