use std::collections::HashMap;
use std::iter::Map;

use thiserror::Error;

use crate::common::Location;
use crate::parser;
use crate::parser::{Block, BlockItem, Declaration, DeclarationKind, Expression, ExpressionKind, ForInit, Function, PrimitiveKind, Program, Statement, StatementKind, Symbol, TypeExpressionKind, VariableDeclaration};

#[derive(Debug, PartialEq)]
struct Type {
    location: Location,
    descriptor: TypeDescriptor,
}

#[derive(Debug, PartialEq)]
enum TypeDescriptor {
    Integer,
    Function(FunctionType),
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

#[derive(Debug, PartialEq)]
struct FunctionType {
    param_types: Vec<Box<TypeDescriptor>>,
    return_type: Box<TypeDescriptor>,
}

struct BlockScope {
    var_types: HashMap<String, Type>,
}

impl BlockScope {
    fn new() -> BlockScope {
        BlockScope {
            var_types: HashMap::new(),
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
        unimplemented!("add_function_declaration unimplemented")
    }

    fn add_variable_declaration(&mut self, ident: Symbol, ty: Type) -> Result<(), TypecheckError> {
        unimplemented!("add_variable_declaration unimplemented")
    }

    fn get_type(&self, v: &String) -> Result<TypeDescriptor, TypecheckError> {
        unimplemented!("get_type unimplemented")
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

    fn get_current_scope(&self) -> &BlockScope {
        self.scopes.last().expect("expected at least one scope")
    }

    fn get_current_scope_mut(&mut self) -> &BlockScope {
        self.scopes.last_mut().expect("expected at least one scope")
    }
}

#[derive(Debug, Error)]
enum TypecheckError {
    #[error("{location:?}: function '{symbol:?}' used as variable")]
    FunctionUsedAsVariable { symbol: String, location: Location },

    #[error("{location:?}: variable '{symbol:?}' of type '{symbol_type:?}' used as function")]
    VariableUsedAsFunction { symbol: String, symbol_type: TypeDescriptor, location: Location },

    #[error("{location:?}: function '{func_name:?}' called with wrong number of args (expected:{expected_param_count:?}, actual:{actual_param_count:?}")]
    FunctionCalledWithWrongNumberOfArguments {
        location: Location,
        func_name: String,
        expected_param_count: usize,
        actual_param_count: usize,
    },
}

pub fn typecheck_program(p: &Program) -> Result<(), TypecheckError> {
    let mut ctx = TypecheckContext::new();
    for decl in p.declarations.iter() {
        typecheck_declaration(&mut ctx, decl)?;
    }
    Ok(())
}

fn typecheck_declaration(ctx: &mut TypecheckContext, decl: &Declaration) -> Result<(), TypecheckError> {
    match &decl.kind {
        DeclarationKind::VarDeclaration(var_decl) => typecheck_variable_declaration(ctx, var_decl),
        DeclarationKind::FunctionDeclaration(func_decl) => typecheck_function_declaration(ctx, func_decl)
    }
}

fn typecheck_function_declaration(ctx: &mut TypecheckContext, decl: &Function) -> Result<(), TypecheckError> {
    ctx.add_function_declaration(decl)?;
    ctx.with_scope(|sub_ctx| {
        if let Some(ref body) = decl.body {
            for formal_param in decl.params.iter() {
                let p = Type::from(&formal_param.param_type);
                sub_ctx.add_variable_declaration(decl.name.clone(), p)?;
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
    Ok(())
}

fn typecheck_block(ctx: &mut TypecheckContext, block: &Block) -> Result<(), TypecheckError> {
    ctx.with_scope(|sub_ctx| { do_typecheck_block(sub_ctx, block) })
}

fn do_typecheck_block(ctx: &mut TypecheckContext, block: &Block) -> Result<(), TypecheckError> {
    for blk_item in block.items.iter() {
        match blk_item {
            BlockItem::Statement(stmt) => {}
            BlockItem::Declaration(decl) => {
                let decl_kind = &decl.kind;
                match decl_kind {
                    DeclarationKind::VarDeclaration(vd) => typecheck_variable_declaration(ctx, vd)?,
                    DeclarationKind::FunctionDeclaration(fd) => typecheck_function_declaration(ctx, fd)?,
                }
            }
        }
    }
    Ok(())
}

fn typecheck_variable_declaration(ctx: &mut TypecheckContext, decl: &VariableDeclaration) -> Result<(), TypecheckError> {
    let int_variable_type = Type {
        location: decl.identifier.location.clone(),
        descriptor: TypeDescriptor::Integer,
    };
    ctx.add_variable_declaration(decl.identifier.clone(), int_variable_type)?;
    Ok(())
}

fn typecheck_statement(ctx: &mut TypecheckContext, stmt: &Statement) -> Result<(), TypecheckError> {
    let loc = stmt.location.clone();
    match &stmt.kind {
        StatementKind::Return(ret_expr) => typecheck_expression(ctx, ret_expr)?,
        StatementKind::Expression(expr) => typecheck_expression(ctx, expr)?,
        StatementKind::SubBlock(blk) => typecheck_block(ctx, blk)?,
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
                    ForInit::InitDecl(_) => {}
                    ForInit::InitExpr(_) => {}
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
    Ok(())
}

fn typecheck_expression(ctx: &mut TypecheckContext, expr: &Expression) -> Result<(), TypecheckError> {
    let loc = expr.location.clone();
    match &expr.kind {
        ExpressionKind::Variable(v) => {
            let actual_symbol_type = ctx.get_type(v)?;
            if actual_symbol_type != TypeDescriptor::Integer {
                return Err(TypecheckError::FunctionUsedAsVariable {
                    symbol: v.clone(),
                    location: loc,
                })
            }
        }
        ExpressionKind::FunctionCall { func_name, actual_params } => {
            let actual_symbol_type = ctx.get_type(func_name)?;
            match actual_symbol_type {
                TypeDescriptor::Function(func_type) => {
                    let expected_num_params = func_type.param_types.len();
                    let actual_num_params = actual_params.len();
                    if expected_num_params != actual_num_params {
                        return Err(TypecheckError::FunctionCalledWithWrongNumberOfArguments {
                            location: loc,
                            func_name: func_name.clone(),
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
                }
                other_type_descriptor => {
                    return Err(TypecheckError::VariableUsedAsFunction {
                        symbol: func_name.clone(),
                        symbol_type: other_type_descriptor,
                        location: loc,
                    });
                }
            }
        }
        _ => {}
    };
    Ok(())
}
