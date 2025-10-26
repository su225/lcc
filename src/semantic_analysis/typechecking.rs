use std::collections::HashMap;
use std::iter::Map;

use thiserror::Error;

use crate::common::Location;
use crate::parser;
use crate::parser::{Block, Declaration, DeclarationKind, Expression, Function, PrimitiveKind, Program, Statement, Symbol, TypeExpressionKind, VariableDeclaration};

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
            typecheck_block(sub_ctx, body)?;
        }
        Ok(())
    })?;
    Ok(())
}

fn typecheck_variable_declaration(ctx: &mut TypecheckContext, decl: &VariableDeclaration) -> Result<(), TypecheckError> {
    unimplemented!()
}

fn typecheck_block(ctx: &mut TypecheckContext, block: &Block) -> Result<(), TypecheckError> {
    unimplemented!()
}

fn typecheck_statement(ctx: &mut TypecheckContext, stmt: &Statement) -> Result<(), TypecheckError> {
    unimplemented!()
}

fn typecheck_expression(ctx: &mut TypecheckContext, expr: &Expression) -> Result<(), TypecheckError> {
    unimplemented!()
}
