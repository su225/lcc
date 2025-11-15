use std::cmp::PartialEq;
use std::collections::HashMap;

use thiserror::Error;
use TypecheckError::FunctionUsedAsVariable;
use crate::common::Location;
use crate::parser;
use crate::parser::{Block, BlockItem, Declaration, DeclarationKind, Expression, ExpressionKind, ForInit, Function, PrimitiveKind, Program, Statement, StatementKind, Symbol, TypeExpressionKind, UnaryOperator, VariableDeclaration};
use crate::semantic_analysis::typechecking::TypecheckError::{IncompatibleTypeForUnaryOp, IncompatibleTypes, IncompatibleTypesForTernary};

#[derive(Debug, PartialEq, Clone)]
pub struct Type {
    location: Location,
    descriptor: TypeDescriptor,
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeDescriptor {
    Void,
    Integer,
    Function(FunctionType),
}

impl TypeDescriptor {
    fn is_function(&self) -> bool {
        match self {
            TypeDescriptor::Function(_) => true,
            _ => false,
        }
    }

    fn is_assignable_to(&self, other: &TypeDescriptor) -> bool {
        match (self, other) {
            (TypeDescriptor::Void, TypeDescriptor::Void) => true,
            (TypeDescriptor::Integer, TypeDescriptor::Integer) => true,
            (TypeDescriptor::Function(f1), TypeDescriptor::Function(f2)) => {
                let are_return_types_compatible = f1.return_type.is_assignable_to(&*f2.return_type);
                are_return_types_compatible &&
                    f1.param_types.len() == f2.param_types.len() &&
                    f1.param_types.iter().zip(f2.param_types.iter())
                        .all(|(p,q)| p.is_assignable_to(q))
            }
            _ => false,
        }
    }
}

impl From<&TypeExpressionKind> for TypeDescriptor {
    fn from(v: &TypeExpressionKind) -> Self {
        match &v {
            TypeExpressionKind::Primitive(PrimitiveKind::Integer) => TypeDescriptor::Integer,
            TypeExpressionKind::Primitive(_) => unimplemented!("primitive types other than int not supported"),
        }
    }
}

impl From<&parser::TypeExpression> for Type {
    fn from(v: &parser::TypeExpression) -> Self {
        Type {
            location: v.location.clone(),
            descriptor: TypeDescriptor::from(&v.kind),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct FunctionType {
    param_types: Vec<Box<TypeDescriptor>>,
    return_type: Box<TypeDescriptor>,
}

impl From<&Function> for FunctionType {
    fn from(f: &Function) -> Self {
        let param_types = f.params.iter()
            .map(|fp| Box::new(TypeDescriptor::from(&fp.param_type.kind)))
            .collect::<Vec<Box<TypeDescriptor>>>();
        let return_type = Box::new(TypeDescriptor::Integer); // hardcoded for now
        FunctionType { param_types, return_type }
    }
}

struct BlockScope {
    symbol_types: HashMap<String, Type>,
}

impl BlockScope {
    fn add_declaration_ignore_conflict(&mut self, symbol: Symbol, ty: Type) -> Result<(), TypecheckError> {
        self.symbol_types.insert(symbol.name, ty);
        Ok(())
    }

    fn add_declaration(&mut self, symbol: Symbol, ty: Type) -> Result<(), TypecheckError> {
        if let Some(prev_decl) = self.symbol_types.get(&symbol.name) {
            return Err(TypecheckError::RedeclarationNotAllowed {
                symbol,
                previous_decl: prev_decl.clone(),
                current_decl: ty,
            })
        }
        // we have already ensured that there is no conflict in the previous step.
        // Hence, we can just call with ignore_conflict.
        self.add_declaration_ignore_conflict(symbol, ty)
    }

    fn get_type(&self, symbol: &String) -> Option<Type> {
        self.symbol_types.get(symbol).cloned()
    }
}

impl BlockScope {
    fn new() -> BlockScope {
        BlockScope {
            symbol_types: HashMap::new(),
        }
    }
}

struct TypecheckContext {
    declared_funcs: HashMap<String, Type>,
    scopes: Vec<BlockScope>,
}

impl TypecheckContext {
    fn new() -> Self {
        TypecheckContext {
            declared_funcs: HashMap::new(),
            scopes: vec![BlockScope::new()],
        }
    }

    fn add_function_declaration(&mut self, f: &Function) -> Result<(), TypecheckError> {
        let func_type = FunctionType::from(f);
        let func_decl_location = f.location.clone();
        let symbol_type = Type {
            location: func_decl_location,
            descriptor: TypeDescriptor::Function(func_type),
        };
        // if the function is declared anywhere, then the signature must be
        // the same. Otherwise, it is a compile error.
        if let Some(prev_func_decl_type) = self.declared_funcs.get(&f.name.name) {
            debug_assert!(prev_func_decl_type.descriptor.is_function());
            match (&prev_func_decl_type.descriptor, &symbol_type.descriptor) {
                (TypeDescriptor::Function(prev_func_type), TypeDescriptor::Function(cur_func_type)) => {
                    if prev_func_type != cur_func_type {
                        return Err(TypecheckError::FunctionRedeclaredWithDifferentSignature {
                            func_name: f.name.clone(),
                            prev_declared_type: prev_func_decl_type.clone(),
                            cur_declared_type: symbol_type,
                        })
                    }
                }
                _ => panic!("declared_funcs must only have function type descriptors"),
            };
        }
        // in case of function, we can declare the same signature multiple times in the same
        // scope or even across scopes. This is not a conflict.
        self.declared_funcs.insert(f.name.name.clone(), symbol_type.clone());
        let cur_scope = self.get_current_scope_mut();
        cur_scope.add_declaration_ignore_conflict(f.name.clone(), symbol_type)?;
        Ok(())
    }

    fn add_variable_declaration(&mut self, ident: Symbol, ty: Type) -> Result<(), TypecheckError> {
        let cur_scope = self.get_current_scope_mut();
        cur_scope.add_declaration(ident, ty)
    }

    fn get_type(&self, v: &Symbol) -> Result<Type, TypecheckError> {
        for scope in self.scopes.iter().rev() {
            let ty = scope.get_type(&v.name);
            if let Some(t) = ty {
                return Ok(t)
            }
        }
        Err(TypecheckError::UnknownTypeForSymbol { symbol: v.clone(), location: v.location.clone() })
    }

    #[inline]
    fn with_scope<T, F>(&mut self, f: F) -> Result<T, TypecheckError>
    where
        F: Fn(&mut TypecheckContext) -> Result<T, TypecheckError>
    {
        let scope = BlockScope::new();
        self.scopes.push(scope);
        let result = { f(self) };
        self.scopes.pop();
        result
    }

    fn get_current_scope_mut(&mut self) -> &mut BlockScope {
        self.scopes.last_mut().expect("expected at least one scope")
    }
}

#[derive(Debug, Error)]
pub enum TypecheckError {
    #[error("{location:?}: function '{symbol:?}' used as variable")]
    FunctionUsedAsVariable { symbol: Symbol, location: Location },

    #[error("{location:?}: variable '{symbol:?}' of type '{symbol_type:?}' used as function")]
    VariableUsedAsFunction { symbol: Symbol, symbol_type: TypeDescriptor, location: Location },

    #[error("{location:?}: function '{func_name:?}' called with wrong number of args (expected:{expected_param_count:?}, actual:{actual_param_count:?}")]
    FunctionCalledWithWrongNumberOfArguments {
        location: Location,
        func_name: Symbol,
        expected_param_count: usize,
        actual_param_count: usize,
    },

    #[error("function '{func_name:?}' re-declared with different signature. Prev:{prev_declared_type:?}, Cur:{cur_declared_type:?}")]
    FunctionRedeclaredWithDifferentSignature {
        func_name: Symbol,
        prev_declared_type: Type,
        cur_declared_type: Type,
    },

    #[error("symbol '{symbol:?}' re-declared in scope. Prev:{previous_decl:?}, Cur:{current_decl:?}")]
    RedeclarationNotAllowed { symbol: Symbol, previous_decl: Type, current_decl: Type },

    #[error("{location:?}: type for symbol '{symbol:?}' unknown")]
    UnknownTypeForSymbol { location: Location, symbol: Symbol },

    #[error("{location:?}: incompatible type `{op_type:?}` for unary operator {unary_op:?}")]
    IncompatibleTypeForUnaryOp { location: Location, unary_op: UnaryOperator, op_type: TypeDescriptor },

    #[error("{location:?}: incompatible types: lhs:{lhs_type:?}, rhs:{rhs_type:?}")]
    IncompatibleTypes { location: Location, lhs_type: TypeDescriptor, rhs_type: TypeDescriptor },

    #[error("{location:?}: incompatible types for ternary operator: then:{then_type:?}, else:{else_type:?}")]
    IncompatibleTypesForTernary { location: Location, then_type: TypeDescriptor, else_type: TypeDescriptor },
}

pub fn typecheck_program(p: &Program) -> Result<(), TypecheckError> {
    let mut ctx = TypecheckContext::new();
    for decl in p.declarations.iter() {
        typecheck_declaration(&mut ctx, decl)?;
    }
    Ok(())
}

fn typecheck_declaration(ctx: &mut TypecheckContext, decl: &Declaration) -> Result<TypeDescriptor, TypecheckError> {
    match &decl.kind {
        DeclarationKind::VarDeclaration(var_decl) => typecheck_variable_declaration(ctx, var_decl),
        DeclarationKind::FunctionDeclaration(func_decl) => typecheck_function_declaration(ctx, func_decl)
    }
}

fn typecheck_function_declaration(ctx: &mut TypecheckContext, decl: &Function) -> Result<TypeDescriptor, TypecheckError> {
    ctx.add_function_declaration(decl)?;
    ctx.with_scope(|sub_ctx| {
        if let Some(ref body) = decl.body {
            for formal_param in decl.params.iter() {
                let p = Type::from(&*formal_param.param_type);
                let formal_param_symbol = Symbol {
                    name: formal_param.param_name.name.clone(),
                    location: formal_param.param_name.location.clone(),
                    original_name: None,
                };
                sub_ctx.add_variable_declaration(formal_param_symbol, p)?;
            }
            // typecheck without introducing a new scope. This is because re-declaring
            // the same formal parameter with a different type is an error if it is done
            // in the main scope of the function. However, if it is not, if another scope
            // is introduced within this. Thus, typechecking block indeed needs a different
            // sub-scope, but typechecking function body must be done within the scope.
            do_typecheck_block(sub_ctx, body)?;
        }
        Ok(())
    })?;
    Ok(TypeDescriptor::Void)
}

fn typecheck_block(ctx: &mut TypecheckContext, block: &Block) -> Result<TypeDescriptor, TypecheckError> {
    ctx.with_scope(|sub_ctx| { do_typecheck_block(sub_ctx, block) })
}

fn do_typecheck_block(ctx: &mut TypecheckContext, block: &Block) -> Result<TypeDescriptor, TypecheckError> {
    for blk_item in block.items.iter() {
        match blk_item {
            BlockItem::Statement(stmt) => { typecheck_statement(ctx, stmt)?; },
            BlockItem::Declaration(decl) => {
                let decl_kind = &decl.kind;
                match decl_kind {
                    DeclarationKind::VarDeclaration(vd) => { typecheck_variable_declaration(ctx, vd)?; },
                    DeclarationKind::FunctionDeclaration(fd) => { typecheck_function_declaration(ctx, fd)?; }
                }
            }
        }
    }
    Ok(TypeDescriptor::Void)
}

fn typecheck_variable_declaration(ctx: &mut TypecheckContext, decl: &VariableDeclaration) -> Result<TypeDescriptor, TypecheckError> {
    let var_type_descriptor = TypeDescriptor::Integer;
    let int_variable_type = Type {
        location: decl.identifier.location.clone(),
        descriptor: var_type_descriptor.clone(),
    };
    ctx.add_variable_declaration(decl.identifier.clone(), int_variable_type.clone())?;
    if let Some(ref init_expr) = decl.init_expression {
        let init_expr_type = typecheck_expression(ctx, init_expr)?;
        if !init_expr_type.is_assignable_to(&int_variable_type.descriptor) {
            return Err(IncompatibleTypes {
                location: init_expr.location.clone(),
                lhs_type: int_variable_type.descriptor,
                rhs_type: init_expr_type,
            });
        }
    }
    Ok(var_type_descriptor)
}

fn typecheck_statement(ctx: &mut TypecheckContext, stmt: &Statement) -> Result<TypeDescriptor, TypecheckError> {
    match &stmt.kind {
        StatementKind::Return(ret_expr) => { typecheck_expression(ctx, ret_expr)?; },
        StatementKind::Expression(expr) => { typecheck_expression(ctx, expr)?; },
        StatementKind::SubBlock(blk) => { typecheck_block(ctx, blk)?; },
        StatementKind::If { condition, then_statement, else_statement } => {
            typecheck_expression(ctx, &*condition)?;
            typecheck_statement(ctx, &*then_statement)?;
            if let Some(ref else_stmt) = else_statement {
                typecheck_statement(ctx, &*else_stmt)?;
            }
        },
        StatementKind::While { pre_condition, loop_body, .. } => {
            typecheck_expression(ctx, &*pre_condition)?;
            typecheck_statement(ctx, &*loop_body)?;
        }
        StatementKind::DoWhile { post_condition, loop_body, .. } => {
            typecheck_expression(ctx, &*post_condition)?;
            typecheck_statement(ctx, &*loop_body)?;
        }
        StatementKind::For { init, condition, post, loop_body, .. } => {
            ctx.with_scope(|sub_ctx| {
                match init {
                    ForInit::InitDecl(decl) => { typecheck_variable_declaration(sub_ctx, &*decl)?; }
                    ForInit::InitExpr(expr) => { typecheck_expression(sub_ctx, &*expr)?; }
                    ForInit::Null => {}
                };
                if let Some(ref cond) = condition {
                    typecheck_expression(sub_ctx, cond)?;
                }
                if let Some(loop_post) = post {
                    typecheck_expression(sub_ctx, loop_post)?;
                }
                typecheck_statement(sub_ctx, loop_body)?;
                Ok(())
            })?;
        }
        StatementKind::Break(_) => {}
        StatementKind::Continue(_) => {}
        StatementKind::Goto { .. } => {}
        StatementKind::Null => {}
    };
    Ok(TypeDescriptor::Void)
}

fn typecheck_expression(ctx: &mut TypecheckContext, expr: &Expression) -> Result<TypeDescriptor, TypecheckError> {
    let loc = expr.location.clone();
    match &expr.kind {
        ExpressionKind::Variable(v) => {
            let actual_symbol_type = ctx.get_type(&v)?.descriptor;
            if actual_symbol_type != TypeDescriptor::Integer {
                return Err(FunctionUsedAsVariable { symbol: v.clone(), location: loc })
            }
            Ok(actual_symbol_type.clone())
        }
        ExpressionKind::Assignment { lvalue, rvalue, .. } => {
            let lhs_type = typecheck_expression(ctx, &*lvalue)?;
            let rhs_type = typecheck_expression(ctx, &*rvalue)?;
            if !rhs_type.is_assignable_to(&lhs_type) {
                return Err(IncompatibleTypes {
                    location: loc.clone(),
                    lhs_type: lhs_type.clone(),
                    rhs_type: rhs_type.clone(),
                })
            }
            Ok(lhs_type)
        }
        ExpressionKind::FunctionCall { func_name, actual_params } => {
            let func_name_symbol = Symbol { name: func_name.name.clone(), location: loc.clone(), original_name: None };
            let actual_symbol_type = ctx.get_type(&func_name_symbol)?.descriptor;
            match actual_symbol_type {
                TypeDescriptor::Function(func_type) => {
                    let return_type = func_type.return_type.clone();
                    let expected_num_params = func_type.param_types.len();
                    let actual_num_params = actual_params.len();
                    if expected_num_params != actual_num_params {
                        return Err(TypecheckError::FunctionCalledWithWrongNumberOfArguments {
                            location: loc,
                            func_name: func_name_symbol,
                            expected_param_count: expected_num_params,
                            actual_param_count: actual_num_params,
                        });
                    }
                    // Once we know that expected and actual parameter lists are of the
                    // same length, we can then typecheck the actual parameter against
                    // the expected parameter types. For now, we just want to typecheck
                    // if all of them are int since it is the only supported type
                    for actual_param in actual_params.iter() {
                        typecheck_expression(ctx, &*actual_param)?;
                    }
                    Ok(*return_type)
                }
                other_type_descriptor => {
                    return Err(TypecheckError::VariableUsedAsFunction {
                        symbol: func_name_symbol.clone(),
                        symbol_type: other_type_descriptor,
                        location: loc,
                    });
                }
            }
        }
        ExpressionKind::IntConstant(_, _) => Ok(TypeDescriptor::Integer),
        ExpressionKind::Unary(unary_op, expr) => {
            let etype = typecheck_expression(ctx, &*expr)?;
            if etype != TypeDescriptor::Integer {
                return Err(IncompatibleTypeForUnaryOp {
                    location: loc.clone(),
                    unary_op: unary_op.clone(),
                    op_type: etype,
                });
            }
            Ok(etype)
        },
        ExpressionKind::Binary(_, lexpr, rexpr) => {
            let ltype = typecheck_expression(ctx, &*lexpr)?;
            let rtype = typecheck_expression(ctx, &*rexpr)?;
            if ltype != TypeDescriptor::Integer || rtype != TypeDescriptor::Integer {
                return Err(IncompatibleTypes { location: loc.clone(), lhs_type: ltype, rhs_type: rtype });
            }
            Ok(TypeDescriptor::Integer)
        }
        ExpressionKind::Conditional { condition, then_expr, else_expr } => {
            let cond_type = typecheck_expression(ctx, &*condition)?;
            let then_type = typecheck_expression(ctx, &*then_expr)?;
            let else_type = typecheck_expression(ctx, &*else_expr)?;
            if cond_type != TypeDescriptor::Integer {
                return Err(IncompatibleTypes { location: loc.clone(), lhs_type: TypeDescriptor::Integer, rhs_type: cond_type });
            }
            if !then_type.is_assignable_to(&else_type) || !else_type.is_assignable_to(&then_type) {
                return Err(IncompatibleTypesForTernary { location: loc.clone(), then_type, else_type });
            }
            Ok(then_type)
        }
        ExpressionKind::Increment { e, .. } | ExpressionKind::Decrement { e, .. } => {
            let etype = typecheck_expression(ctx, &*e)?;
            if etype != TypeDescriptor::Integer {
                return Err(IncompatibleTypeForUnaryOp {
                    location: loc.clone(),
                    unary_op: UnaryOperator::Increment,
                    op_type: etype,
                });
            }
            Ok(TypeDescriptor::Integer)
        }
    }
}


#[cfg(test)]
mod test {
    use crate::lexer::Lexer;
    use crate::parser::{Parser};
    use crate::semantic_analysis::identifier_resolution::{resolve_program};
    use crate::semantic_analysis::typechecking::{FunctionType, typecheck_program, TypecheckError, TypeDescriptor};

    #[test]
    fn test_incompatible_func_declarations_in_different_scopes_must_error() {
        let program = r#"
        int bar(void);

        int main(void) {
            int foo(int a);
            return bar() + foo(1);
        }

        int bar(void) {
            int foo(int a, int b);
            return foo(1, 2);
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_cannot_divide_by_function_type() {
        let program = r#"
        int x(void);

        int main(void) {
            int a = 10 / x;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_valid_basic_program() {
        let program = r#"
        int add(int a, int b);
        int multiply(int x, int y);

        int main(void) {
            int x = 5;
            int y = add(2, 3);
            int z = multiply(x, y);
            return z;
        }

        int add(int a, int b) {
            return a + b;
        }

        int multiply(int x, int y) {
            return x * y;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_variable_declarations() {
        let program = r#"
        int main(void) {
            int a;
            int b = 10;
            int c = 20 + 30;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_all_binary_operators() {
        let program = r#"
        int main(void) {
            int a = 10 + 5;
            int b = 10 - 5;
            int c = 10 * 5;
            int d = 10 / 5;
            int e = 10 % 5;
            int f = 10 & 5;
            int g = 10 | 5;
            int h = 10 ^ 5;
            int i = 10 << 2;
            int j = 10 >> 2;
            int k = 10 && 5;
            int l = 10 || 5;
            int m = 10 == 5;
            int n = 10 != 5;
            int o = 10 < 5;
            int p = 10 <= 5;
            int q = 10 > 5;
            int r = 10 >= 5;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_all_unary_operators() {
        let program = r#"
        int main(void) {
            int a = 10;
            int b = -a;
            int c = ~a;
            int d = !a;
            int e = ++a;
            int f = a++;
            int g = --a;
            int h = a--;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_control_flow() {
        let program = r#"
        int main(void) {
            int x = 5;
            if (x > 0) {
                return 1;
            } else {
                return 0;
            }
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_while_loop() {
        let program = r#"
        int main(void) {
            int i = 0;
            while (i < 10) {
                i = i + 1;
            }
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_do_while_loop() {
        let program = r#"
        int main(void) {
            int i = 0;
            do {
                i = i + 1;
            } while (i < 10);
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_for_loop_with_decl() {
        let program = r#"
        int main(void) {
            for (int i = 0; i < 10; i = i + 1) {
                int x = i;
            }
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_for_loop_with_expr() {
        let program = r#"
        int main(void) {
            int i = 0;
            for (i = 0; i < 10; i = i + 1) {
                int x = i;
            }
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_for_loop_empty_parts() {
        let program = r#"
        int main(void) {
            for (;;) {
                break;
            }
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_nested_scopes() {
        let program = r#"
        int main(void) {
            int x = 5;
            {
                int y = 10;
                {
                    int z = 15;
                    x = x + y + z;
                }
            }
            return x;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_variable_shadowing() {
        let program = r#"
        int main(void) {
            int x = 5;
            {
                int x = 10;
                {
                    int x = 15;
                }
            }
            return x;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_function_redeclaration_same_signature() {
        let program = r#"
        int foo(int a);
        int foo(int a);

        int main(void) {
            int foo(int a);
            return foo(5);
        }

        int foo(int a) {
            return a;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_ternary_operator() {
        let program = r#"
        int main(void) {
            int x = 5;
            int y = x > 0 ? 10 : 20;
            return y;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_assignment_expressions() {
        let program = r#"
        int main(void) {
            int a = 5;
            int b = 10;
            a = b = 15;
            a += 5;
            a -= 3;
            a *= 2;
            a /= 4;
            a %= 3;
            return a;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_nested_function_calls() {
        let program = r#"
        int add(int a, int b);
        int multiply(int x, int y);

        int main(void) {
            int result = multiply(add(1, 2), add(3, 4));
            return result;
        }

        int add(int a, int b) {
            return a + b;
        }

        int multiply(int x, int y) {
            return x * y;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_recursive_function() {
        let program = r#"
        int factorial(int n);

        int main(void) {
            return factorial(5);
        }

        int factorial(int n) {
            if (n <= 1) {
                return 1;
            } else {
                return n * factorial(n - 1);
            }
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_break_continue() {
        let program = r#"
        int main(void) {
            int i = 0;
            while (i < 10) {
                if (i == 5) {
                    break;
                }
                if (i == 3) {
                    i = i + 1;
                    continue;
                }
                i = i + 1;
            }
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_function_used_as_variable_in_assignment() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = foo;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_function_used_as_variable_in_binary_op() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = foo + 5;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_function_used_as_variable_in_unary_op() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = -foo;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_function_used_in_increment() {
        let program = r#"
        int foo(void);

        int main(void) {
            foo++;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_function_used_in_ternary() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = 5 > 0 ? foo : 10;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_variable_used_as_function() {
        let program = r#"
        int main(void) {
            int x = 5;
            int y = x();
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_variable_used_as_function_with_args() {
        let program = r#"
        int main(void) {
            int x = 5;
            int y = x(1, 2, 3);
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_function_called_with_too_few_args() {
        let program = r#"
        int add(int a, int b);

        int main(void) {
            int x = add(5);
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_function_called_with_too_many_args() {
        let program = r#"
        int add(int a, int b);

        int main(void) {
            int x = add(5, 10, 15);
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_function_called_with_args_when_expects_none() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = foo(5);
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_function_called_with_no_args_when_expects_some() {
        let program = r#"
        int add(int a, int b);

        int main(void) {
            int x = add();
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_function_redeclared_different_param_count() {
        let program = r#"
        int foo(int a);
        int foo(int a, int b);

        int main(void) {
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_function_redeclared_different_signature_same_scope() {
        let program = r#"
        int main(void) {
            int foo(int a);
            int foo(int a, int b);
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }


    #[test]
    fn test_unary_op_on_function() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = -foo();
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_unary_op_on_function_as_operand() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = ~foo;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_binary_op_with_function_type() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = foo + 5;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_assignment_incompatible_types() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = 5;
            x = foo;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_variable_init_incompatible_type() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = foo;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_ternary_incompatible_branch_types() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = 5 > 0 ? foo : 10;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_ternary_incompatible_branch_types_reverse() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = 5 > 0 ? 10 : foo;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_err(), "{:?}", res);
    }

    #[test]
    fn test_ternary_condition_not_integer() {
        let program = r#"
        int foo(void);

        int main(void) {
            int x = foo() ? 10 : 20;
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_complex_program() {
        let program = r#"
        int add(int a, int b);
        int multiply(int x, int y);
        int factorial(int n);

        int main(void) {
            int result = 0;
            int i = 0;
            while (i < 5) {
                int temp = add(i, 1);
                result = multiply(result, temp);
                i = i + 1;
            }
            if (result > 0) {
                return factorial(result);
            } else {
                return 0;
            }
        }

        int add(int a, int b) {
            return a + b;
        }

        int multiply(int x, int y) {
            return x * y;
        }

        int factorial(int n) {
            if (n <= 1) {
                return 1;
            } else {
                return n * factorial(n - 1);
            }
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_nested_control_flow() {
        let program = r#"
        int main(void) {
            int i = 0;
            while (i < 10) {
                if (i % 2 == 0) {
                    int j = 0;
                    for (j = 0; j < 5; j = j + 1) {
                        if (j == 3) {
                            break;
                        }
                    }
                } else {
                    do {
                        i = i + 1;
                    } while (i < 8);
                }
                i = i + 1;
            }
            return 0;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }


    #[test]
    fn test_valid_deeply_nested_scopes() {
        let program = r#"
        int main(void) {
            int a = 1;
            {
                int b = 2;
                {
                    int c = 3;
                    {
                        int d = 4;
                        a = a + b + c + d;
                    }
                }
            }
            return a;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_for_loop_variable_scope() {
        let program = r#"
        int main(void) {
            int x = 0;
            for (int i = 0; i < 5; i = i + 1) {
                x = x + i;
            }
            return x;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_ternary_nested() {
        let program = r#"
        int main(void) {
            int x = 5;
            int y = x > 0 ? (x > 10 ? 20 : 15) : 0;
            return y;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    #[test]
    fn test_valid_function_call_in_ternary() {
        let program = r#"
        int add(int a, int b);
        int sub(int a, int b);

        int main(void) {
            int x = 5 > 0 ? add(1, 2) : sub(3, 4);
            return x;
        }

        int add(int a, int b) {
            return a + b;
        }

        int sub(int a, int b) {
            return a - b;
        }
        "#;
        let res = run_program_typechecks(program);
        assert!(res.is_ok(), "{:?}", res);
    }

    fn run_program_typechecks(program: &str) -> Result<(), TypecheckError> {
        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok(), "{:?}", parsed);
        let resolved_ast = resolve_program(parsed.unwrap());
        assert!(resolved_ast.is_ok(), "{:?}", resolved_ast);
        let typechecked_ast = typecheck_program(&resolved_ast.unwrap());
        typechecked_ast
    }

    #[test]
    fn test_func_type_equality_returns_true_when_return_type_and_arg_types_and_arity_matches() {
        let f1_descriptor = FunctionType {
            param_types: vec![
                Box::new(TypeDescriptor::Integer),
                Box::new(TypeDescriptor::Integer),
            ],
            return_type: Box::new(TypeDescriptor::Integer),
        };
        let f2_descriptor = FunctionType {
            param_types: vec![
                Box::new(TypeDescriptor::Integer),
                Box::new(TypeDescriptor::Integer),
            ],
            return_type: Box::new(TypeDescriptor::Integer),
        };
        assert_eq!(f1_descriptor, f2_descriptor);
    }

    #[test]
    fn test_func_type_equality_returns_false_when_there_is_arity_mismatch() {
        let f1_descriptor = FunctionType {
            param_types: vec![],
            return_type: Box::new(TypeDescriptor::Integer),
        };
        let f2_descriptor = FunctionType {
            param_types: vec![
                Box::new(TypeDescriptor::Integer),
                Box::new(TypeDescriptor::Integer),
            ],
            return_type: Box::new(TypeDescriptor::Integer),
        };
        assert_ne!(f1_descriptor, f2_descriptor);
    }
}