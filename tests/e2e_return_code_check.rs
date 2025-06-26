use std::{env, fs};
use std::path::{Path, PathBuf};
use std::process::Command;
use assert_cmd::cargo::cargo_bin;
use rstest::rstest;

#[rstest]
#[case("binary/arithmetic_add.c")]
#[case("binary/arithmetic_add_leftassoc.c")]
#[case("binary/arithmetic_division.c")]
#[case("binary/arithmetic_modulo.c")]
#[case("binary/arithmetic_multiplication.c")]
#[case("binary/arithmetic_precedence.c")]
#[case("binary/arithmetic_precedence_override.c")]
#[case("binary/arithmetic_division_assoc.c")]
fn test_e2e_arithmetic_binary_operator(#[case] input_file: &str) {
    run_e2e_against_clang(input_file)
}

#[rstest]
#[case("binary/bitwise_and.c")]
#[case("binary/bitwise_or.c")]
#[case("binary/bitwise_xor.c")]
#[case("binary/bitwise_left_shift.c")]
#[case("binary/bitwise_right_shift.c")]
fn test_e2e_bitwise_binary_operator(#[case] input_file: &str) {
    run_e2e_against_clang(input_file)
}

#[rstest]
#[case("binary/logical_and_false.c")]
fn test_e2e_logical_binary_operator(#[case] input_file: &str) {
    run_e2e_against_clang(input_file)
}

#[rstest]
#[case("localvars/simple.c")]
#[case("localvars/declaration_and_assign.c")]
#[case("localvars/nested_scopes.c")]
#[case("localvars/expression_with_var.c")]
fn test_e2e_local_variables(#[case] input_file: &str) {
    run_e2e_against_clang(input_file)
}

#[rstest]
#[case("compound_assign/add.c")]
#[case("compound_assign/subtract.c")]
#[case("compound_assign/multiply.c")]
#[case("compound_assign/divide.c")]
#[case("compound_assign/modulo.c")]
#[case("compound_assign/bitwise_and.c")]
#[case("compound_assign/bitwise_or.c")]
#[case("compound_assign/bitwise_xor.c")]
#[case("compound_assign/left_shift.c")]
#[case("compound_assign/right_shift.c")]
fn test_e2e_compound_assignment(#[case] input_file: &str) {
    run_e2e_against_clang(input_file)
}

#[rstest]
#[case("unary/complement.c")]
#[case("unary/negation.c")]
#[case("unary/not.c")]
#[case("unary/decr_statement.c")]
#[case("unary/incr_statement.c")]
fn test_e2e_unary_operator(#[case] input_file: &str) {
    run_e2e_against_clang(input_file)
}

#[rstest]
#[case("conditional/if.c")]
#[case("conditional/if_else.c")]
#[case("conditional/if_else_if.c")]
#[case("conditional/dangling_if.c")]
#[case("conditional/ternary.c")]
fn test_e2e_conditional(#[case] input_file: &str) {
    run_e2e_against_clang(input_file)
}

fn run_e2e_against_clang(input_file: &str) {
    println!("current working directory: {}", env::current_dir().unwrap().display());

    println!("compile and run with clang");
    let clang_res = compile_and_run_with_clang(input_file);

    println!("compile and run with lcc (our compiler)");
    let lcc_res = compile_and_run_with_learning_c_compiler(input_file);

    println!("CLANG STDOUT:\n {}", clang_res.clone().compile_stdout);
    println!("LCC STDOUT:\n {}", lcc_res.clone().compile_stdout);
    assert_eq!(clang_res.exit_code, lcc_res.exit_code,
               "{}: clang and lcc generated binary execution should return the same code. clang={:?}, lcc={:?}",
               input_file, clang_res, lcc_res);
}

#[derive(Debug, Clone)]
pub struct CompileRunResult {
    pub exit_code: i32,
    pub compile_stdout: String,
    pub compile_stderr: String,
    pub run_stdout: String,
    pub run_stderr: String,
}

fn compile_and_run_with_clang(input_file: &str) -> CompileRunResult {
    let (input_path, binary_path) = get_input_and_binary_path(input_file, "testout_clang");
    let compiler_output = Command::new("/usr/bin/clang")
        .arg(&input_path)
        .arg("-o").arg(&binary_path)
        .output().unwrap();
    if !compiler_output.status.success() {
        panic!(
            "clang compilation failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&compiler_output.stdout),
            String::from_utf8_lossy(&compiler_output.stderr),
        );
    }
    let run_output = Command::new(&binary_path).output().unwrap();
    let exit_code = run_output.status.code().unwrap();
    CompileRunResult {
        exit_code,
        compile_stdout: String::from_utf8_lossy(&compiler_output.stdout).to_string(),
        compile_stderr: String::from_utf8_lossy(&compiler_output.stderr).to_string(),
        run_stdout: String::from_utf8_lossy(&run_output.stdout).to_string(),
        run_stderr: String::from_utf8_lossy(&run_output.stderr).to_string(),
    }
}

fn compile_and_run_with_learning_c_compiler(input_file: &str) -> CompileRunResult {
    let (input_path, binary_path) = get_input_and_binary_path(input_file, "testout_lcc");
    let mut compiler_bin = Command::new(cargo_bin("lcc"));
    let compiler_cmd = compiler_bin
        .arg(&input_path)
        // .arg("--emit-assembly")
        .arg("-o").arg(&binary_path);
    println!("running command: {:?}", &compiler_cmd);
    let compiler_output = compiler_cmd.output().unwrap();
    if !compiler_output.status.success() {
        panic!(
            "lcc compilation failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&compiler_output.stdout),
            String::from_utf8_lossy(&compiler_output.stderr),
        );
    }
    println!("binary path: {:?}", binary_path);
    println!("working directory: {:?}", env::current_dir());
    let run_output = Command::new(&binary_path).output().unwrap();
    let exit_code = run_output.status.code().unwrap();
    CompileRunResult {
        exit_code,
        compile_stdout: String::from_utf8_lossy(&compiler_output.stdout).to_string(),
        compile_stderr: String::from_utf8_lossy(&compiler_output.stderr).to_string(),
        run_stdout: String::from_utf8_lossy(&run_output.stdout).to_string(),
        run_stderr: String::from_utf8_lossy(&run_output.stderr).to_string(),
    }
}

fn get_input_and_binary_path(input_file: &str, out_dir: &str) -> (PathBuf, PathBuf) {
    let source_dir = Path::new(file!()).parent()
        .expect(&format!("cannot determine source directory for {}", input_file));
    let input_path = source_dir.join("input").join(input_file);
    let out_dir = source_dir.join(out_dir);
    fs::create_dir_all(&out_dir).expect("failed to create out directory");
    let stem = Path::new(input_file)
        .file_stem()
        .expect(&format!("invalid input file name {}", input_file))
        .to_string_lossy();
    let binary_path = out_dir.join(format!("{}", stem));
    return (input_path, binary_path);
}