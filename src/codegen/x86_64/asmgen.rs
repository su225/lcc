use std::io;
use std::io::Write;
use crate::codegen::x86_64::*;
use crate::codegen::x86_64::AsmInstruction::*;

pub fn emit_assembly<W: Write>(asm_code: AsmProgram, mut w: W) -> io::Result<()> {
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
        DeallocateStack(stack_size) => emit_instruction!(w, "addq ${stack_size}, %rsp")?,
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

        Cmp32 { op1, op2 } => emit_instruction!(w, "cmpl {op1}, {op2}")?,
        Jmp { target } => emit_instruction!(w, "jmp {target}")?,
        JmpConditional { condition_code, target_if_true } =>
            emit_instruction!(w, "j{condition_code} {target_if_true}")?,
        SetCondition { condition_code, dst } =>
            emit_instruction!(w, "set{condition_code} {dst}")?,
        Label(lbl) => emit_instruction!(w, "{}", format!("{lbl}:"))?,
        Push { op } => emit_instruction!(w, "pushq {op}")?,
        Call(func_name) => emit_instruction!(w, "call {func_label}", func_label = format_args!("_{func_name}"))?,
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

#[cfg(test)]
mod test {
    use std::fs;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};
    use insta::assert_snapshot;
    use rstest::rstest;
    use crate::codegen::x86_64::{emit_assembly, generate_assembly};
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::semantic_analysis::identifier_resolution::resolve_program;
    use crate::semantic_analysis::loop_labeling::loop_label_program_definition;
    use crate::tacky::emit;

    #[rstest]
    #[case("multifunc/simple.c")]
    fn test_generation_for_multiple_functions(#[case] input_path: &str) {
        run_asm_emit_snapshot_test("multiple functions", input_path);
    }

    #[rstest]
    #[case("unary/complement.c")]
    #[case("unary/negation.c")]
    #[case("unary/not.c")]
    fn test_generation_for_unary_operators(#[case] input_path: &str) {
        run_asm_emit_snapshot_test("unary operators", input_path)
    }

    #[rstest]
    #[case("binary/arithmetic_add.c")]
    #[case("binary/arithmetic_add_leftassoc.c")]
    #[case("binary/arithmetic_precedence.c")]
    #[case("binary/arithmetic_precedence_override.c")]
    #[case("binary/arithmetic_multiplication.c")]
    #[case("binary/arithmetic_division.c")]
    #[case("binary/arithmetic_modulo.c")]
    fn test_generation_for_arithmetic_binary_operators(#[case] input_path: &str) {
        run_asm_emit_snapshot_test("arithmetic binary operators", input_path)
    }

    #[rstest]
    #[case("binary/bitwise_and.c")]
    #[case("binary/bitwise_or.c")]
    #[case("binary/bitwise_xor.c")]
    #[case("binary/bitwise_left_shift.c")]
    #[case("binary/bitwise_right_shift.c")]
    fn test_generation_for_bitwise_binary_operators(#[case] input_path: &str) {
        run_asm_emit_snapshot_test("arithmetic bitwise binary operators", input_path)
    }

    #[rstest]
    #[case("binary/relational_eq.c")]
    fn test_generation_for_relational_binary_operators(#[case] input_path: &str) {
        run_asm_emit_snapshot_test("arithmetic relational binary operators", input_path)
    }

    #[rstest]
    #[case("binary/logical_and_false.c")]
    fn test_generation_for_logical_and_false(#[case] input_path: &str) {
        run_asm_emit_snapshot_test("logical operators with short-circuiting", input_path)
    }

    #[rstest]
    #[case("localvars/simple.c")]
    #[case("localvars/declaration_and_assign.c")]
    #[case("localvars/nested_scopes.c")]
    #[case("localvars/expression_with_var.c")]
    fn test_generation_for_local_vars(#[case] input_path: &str) {
        run_asm_emit_snapshot_test("local variables", input_path)
    }

    #[rstest]
    #[case("conditional/if.c")]
    #[case("conditional/if_else.c")]
    #[case("conditional/if_else_if.c")]
    #[case("conditional/dangling_if.c")]
    #[case("conditional/ternary.c")]
    fn test_generation_for_conditional_statements(#[case] input_path: &str) {
        run_asm_emit_snapshot_test("conditional statements", input_path)
    }

    #[rstest]
    #[case("loops/simple_for.c")]
    #[case("loops/for_init_decl.c")]
    #[case("loops/for_init_decl_shadow.c")]
    #[case("loops/for_noinit.c")]
    #[case("loops/for_all_empty.c")]
    #[case("loops/for_with_break.c")]
    #[case("loops/for_with_continue.c")]
    #[case("loops/nested_for.c")]
    #[case("loops/nested_for_with_break.c")]
    #[case("loops/nested_for_with_continue.c")]
    fn test_generation_for_for_loop(#[case] input_path: &str) {
        run_asm_emit_snapshot_test("loops: for", input_path)
    }

    #[rstest]
    #[case("loops/simple_while.c")]
    #[case("loops/while_with_break.c")]
    #[case("loops/while_with_continue.c")]
    fn test_generation_for_while_loop(#[case] input_path: &str) {
        run_asm_emit_snapshot_test("loops: while", input_path)
    }

    #[rstest]
    #[case("loops/simple_dowhile.c")]
    #[case("loops/dowhile_with_break.c")]
    #[case("loops/dowhile_with_continue.c")]
    fn test_generation_for_dowhile_loop(#[case] input_path: &str) {
        run_asm_emit_snapshot_test("loops: do-while", input_path)
    }

    fn run_asm_emit_snapshot_test(suite_description: &str, src_file: &str) {
        let base_dir = file!();
        let src_path = Path::new(base_dir).parent().unwrap().join("input").join(src_file);
        let source = fs::read_to_string(src_path.clone());
        assert!(source.is_ok(), "failed to read {:?}", src_path);

        let src = source.unwrap();
        let lexer = Lexer::new(&src);
        let mut parser = Parser::new(lexer);
        let ast = parser.parse().expect("parsing failed");
        let id_resolved_ast = resolve_program(ast).expect("identifier resolution failed");
        let loop_labeled_ast = loop_label_program_definition(id_resolved_ast).expect("loop labeling failed");
        let tacky = emit(&loop_labeled_ast).expect("tacky gen failed");
        let asm = generate_assembly(tacky).expect("asm gen failed");

        let mut buffer = Cursor::new(Vec::new());
        emit_assembly(asm, &mut buffer).expect("asm emit failed");
        let asm_output = String::from_utf8(buffer.into_inner()).expect("invalid utf-8");

        let (out_dir, snapshot_file) = output_path_parts(src_file);
        insta::with_settings!({
            sort_maps => true,
            prepend_module_to_snapshot => false,
            description => suite_description,
            snapshot_path => out_dir,
            info => &format!("{}", src_file),
        }, {
            assert_snapshot!(snapshot_file, asm_output);
        });
    }

    fn output_path_parts(src_file: &str) -> (PathBuf, String) {
        let input_path = Path::new(src_file);
        let parent = input_path.parent().unwrap_or_else(|| Path::new(""));
        let stem = input_path.file_stem().expect("No file stem").to_string_lossy();
        let output_dir = Path::new("output").join(parent);
        let output_file = format!("{}.s", stem);
        (output_dir, output_file)
    }
}