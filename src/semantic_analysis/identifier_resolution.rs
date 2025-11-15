use std::collections::HashMap;

use thiserror::Error;

use crate::common::Location;
use crate::parser::{Block, BlockItem, Declaration, DeclarationKind, Expression, ExpressionKind, ForInit, Function, FunctionParameter, Program, Statement, StatementKind, Symbol, TypeExpression, TypeExpressionKind, VariableDeclaration};

#[derive(Debug, Error)]
pub enum IdentifierResolutionError {
    #[error("identifier '{name:?}' not found")]
    NotFound { name: String },

    #[error("{current_loc:?}: identifier '{name:?}' already declared at {original_loc:?}")]
    AlreadyDeclared { current_loc: Location, original_loc: Location, name: String },

    #[error("{0:?}: lvalue expected")]
    LvalueExpected(Location),

    #[error("{cur_location:?} label '{label:?}' already used at {prev_location:?}")]
    LabelAlreadyUsed { prev_location: Location, cur_location: Location, label: String },

    #[error("label '{label:?}' not declared")]
    LabelNotDeclared { label: String },

    #[error("cannot declare function ({name:?}) inside another function")]
    CannotDefineFunctionInsideAnotherFunction { name: String },

    #[error("cannot redeclare function {name:?} as it is already declared at {location:?}")]
    CannotRedefineFunction { name: String, location: Location },

    #[error("{location:?} cannot redeclare variable as function: {name:?}. Previously declared at {prev_location:?}")]
    CannotRedeclareVariableAsFunction { name: String, location: Location, prev_location: Location },

    #[error("{location:?} cannot redeclare function as variable: {name:?}. Previously declared at {prev_location:?}")]
    CannotRedeclareFunctionAsVariable { name: String, location: Location, prev_location: Location },
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum LinkageType {
    Internal,
    External,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MappingType {
    Variable,
    Function,
}

#[derive(Debug, Clone)]
struct ResolvedIdentifier {
    symbol: Symbol,
    linkage_type: LinkageType,
    mapping_type: MappingType,
}

impl ResolvedIdentifier {
    pub fn location(&self) -> Location {
        self.symbol.location.clone()
    }
}

struct Scope {
    identifier_map: HashMap<String, ResolvedIdentifier>,
    is_function: bool,
}

impl Scope {
    fn new() -> Self {
        return Scope {
            identifier_map: HashMap::new(),
            is_function: false,
        }
    }

    fn with_function() -> Self {
        return Scope {
            identifier_map: HashMap::new(),
            is_function: true,
        }
    }

    fn add_mapping(&mut self, raw_ident: Symbol, mapped_ident: ResolvedIdentifier) -> Result<(), IdentifierResolutionError> {
        let existing_mapping = self.identifier_map.get(&raw_ident.name);
        if let Some(cur_mapping) = existing_mapping {
            let existing_mapping_loc = cur_mapping.location().clone();
            let existing_mapping_type = cur_mapping.mapping_type;
            let incoming_mapping_loc = mapped_ident.location().clone();
            let incoming_mapping_type = mapped_ident.mapping_type;
            return match (existing_mapping_type, incoming_mapping_type) {
                (MappingType::Function, MappingType::Function) => Ok(()),
                (MappingType::Function, MappingType::Variable) => Err(IdentifierResolutionError::CannotRedeclareVariableAsFunction {
                    name: raw_ident.name.clone(),
                    location: incoming_mapping_loc,
                    prev_location: existing_mapping_loc,
                }),
                (MappingType::Variable, MappingType::Variable) => Err(IdentifierResolutionError::AlreadyDeclared {
                    current_loc: incoming_mapping_loc,
                    original_loc: existing_mapping_loc,
                    name: raw_ident.name.clone(),
                }),
                (MappingType::Variable, MappingType::Function) => Err(IdentifierResolutionError::CannotRedeclareFunctionAsVariable {
                    name: raw_ident.name.clone(),
                    location: incoming_mapping_loc,
                    prev_location: existing_mapping_loc,
                })
            }
        }
        self.identifier_map.insert(raw_ident.name, mapped_ident);
        Ok(())
    }

    fn lookup(&self, raw_ident: &String) -> Result<ResolvedIdentifier, IdentifierResolutionError> {
        let existing_mapping = self.identifier_map.get(raw_ident);
        match existing_mapping {
            Some(mapped_ident) => Ok(mapped_ident.clone()),
            None => Err(IdentifierResolutionError::NotFound { name: raw_ident.clone() }),
        }
    }
}

#[derive(Debug, PartialEq)]
struct ResolvedLabel {
    label: String,
    location: Location,
}

struct IdentifierResolutionContext {
    scopes: Vec<Scope>,
    labels: HashMap<String, ResolvedLabel>,
    defined_functions: HashMap<String, Location>,
    next_num_id: u64,
}

impl IdentifierResolutionContext {
    fn new() -> Self {
        return IdentifierResolutionContext {
            scopes: vec![Scope::new()],
            labels: HashMap::new(),
            defined_functions: HashMap::new(),
            next_num_id: 0,
        }
    }

    fn add_identifier_mapping(&mut self, ident: Symbol, linkage_type: LinkageType) -> Result<Symbol, IdentifierResolutionError> {
        let next = self.next_num_id;
        self.next_num_id += 1;
        let mapped_symbol = match linkage_type {
            LinkageType::External => ident.clone(),
            LinkageType::Internal => {
                let mapped_ident = format!("{}${}", ident.name, next);
                Symbol { name: mapped_ident, location: ident.location.clone(), original_name: Some(ident.name.clone()) }
            }
        };
        self.get_current_scope_mut().add_mapping(ident, ResolvedIdentifier {
            symbol: mapped_symbol.clone(),
            linkage_type,
            mapping_type: MappingType::Variable,
        })?;
        Ok(mapped_symbol)
    }

    fn add_function(&mut self, ident: Symbol, is_defined: bool) -> Result<Symbol, IdentifierResolutionError> {
        if is_defined {
            self.add_defined_function(ident.clone())?;
        }
        let mapped_symbol = ident.clone();
        let current_scope = self.get_current_scope_mut();
        current_scope.add_mapping(ident, ResolvedIdentifier {
            symbol: mapped_symbol.clone(),
            linkage_type: LinkageType::External,
            mapping_type: MappingType::Function,
        })?;
        Ok(mapped_symbol)
    }

    fn add_defined_function(&mut self, func: Symbol) -> Result<(), IdentifierResolutionError> {
        if self.is_within_function() {
            return Err(IdentifierResolutionError::CannotDefineFunctionInsideAnotherFunction { name: func.name});
        }
        if let Some(prev_location) = self.defined_functions.get(&func.name) {
            return Err(IdentifierResolutionError::CannotRedefineFunction {
                name: func.name,
                location: *prev_location,
            });
        }
        self.defined_functions.insert(func.name, func.location);
        Ok(())
    }

    fn add_label(&mut self, lbl: String, loc: Location) -> Result<(), IdentifierResolutionError> {
        let next = self.next_num_id;
        self.next_num_id += 1;
        let resolved_label = ResolvedLabel {
            label: format!(".L{lbl}.{next}"),
            location: loc.clone(),
        };
        if let Some(prev_resolved_label) = self.labels.insert(lbl.clone(), resolved_label) {
            return Err(IdentifierResolutionError::LabelAlreadyUsed {
                prev_location: prev_resolved_label.location.clone(),
                cur_location: loc,
                label: lbl.clone(),
            });
        }
        Ok(())
    }

    fn get_resolved_label(&self, lbl: &String) -> Result<String, IdentifierResolutionError> {
        if let Some(resolved) = self.labels.get(lbl) {
            Ok(resolved.label.clone())
        } else {
            Err(IdentifierResolutionError::LabelNotDeclared { label: lbl.to_string() })
        }
    }

    fn get_resolved_identifier(&self, raw_ident: &String) -> Result<ResolvedIdentifier, IdentifierResolutionError> {
        for scope in self.scopes.iter().rev() {
            let lookup_result = scope.lookup(raw_ident);
            match lookup_result {
                Ok(resolved_symbol) => return Ok(resolved_symbol),
                Err(IdentifierResolutionError::NotFound {..}) => {
                    continue; // We cannot resolve in this scope. Move up and try again
                }
                Err(e) => return Err(e),
            };
        }
        Err(IdentifierResolutionError::NotFound { name: raw_ident.clone() })
    }

    fn get_resolved_identifier_in_current_scope(&self, raw_ident: &String) -> Result<ResolvedIdentifier, IdentifierResolutionError> {
        let current_scope = self.get_current_scope();
        current_scope.lookup(raw_ident)
    }

    #[inline]
    fn with_function_scope<T, F>(&mut self, f: F) -> Result<T, IdentifierResolutionError>
    where
        F: Fn(&mut IdentifierResolutionContext) -> Result<T, IdentifierResolutionError>
    {
        self.labels.clear();
        self.with_scope_internal(Scope::with_function(), f)
    }

    #[inline]
    fn with_scope<T, F>(&mut self, f: F) -> Result<T, IdentifierResolutionError>
    where
        F: Fn(&mut IdentifierResolutionContext) -> Result<T, IdentifierResolutionError>
    {
        self.with_scope_internal(Scope::new(), f)
    }

    fn with_scope_internal<T, F>(&mut self, scope: Scope, f: F) -> Result<T, IdentifierResolutionError>
    where
        F: Fn(&mut IdentifierResolutionContext) -> Result<T, IdentifierResolutionError>
    {
        self.scopes.push(scope);
        let result = { f(self) };
        self.scopes.pop();
        result
    }

    fn get_current_scope(&self) -> &Scope {
        self.scopes.last().expect("expected at least one scope")
    }

    fn get_current_scope_mut(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("expected at least one scope")
    }

    fn is_within_function(&self) -> bool {
        self.get_current_scope().is_function
    }
}

pub fn resolve_program(program: Program) -> Result<Program, IdentifierResolutionError> {
    let mut ctx = IdentifierResolutionContext::new();
    let mut resolved_funcs = Vec::with_capacity(program.declarations.len());
    for decl in program.declarations.iter() {
        let decl_loc = decl.location.clone();
        match &decl.kind {
            DeclarationKind::FunctionDeclaration(ref f) => {
                let resolved_f = resolve_function(&mut ctx, f)?;
                resolved_funcs.push(Declaration {
                    location: decl_loc,
                    kind: DeclarationKind::FunctionDeclaration(resolved_f),
                });       
            }
            DeclarationKind::VarDeclaration { .. } => unimplemented!("global variables not implemented")
        }
    }
    Ok(Program { declarations: resolved_funcs })
}

fn resolve_function<'a>(ctx: &mut IdentifierResolutionContext, f: &Function) -> Result<Function, IdentifierResolutionError> {
    ctx.add_function(f.name.clone(), f.body.is_some())?;
    ctx.with_function_scope(|sub_ctx| {
        let mut resolved_params = Vec::with_capacity(f.params.len());
        for p in &f.params {
            resolved_params.push(FunctionParameter {
                loc: p.loc.clone(),
                param_type: Box::new(resolve_type_expression(sub_ctx, &p.param_type)?),
                param_name: p.param_name.clone(),
            });
            sub_ctx.add_identifier_mapping(Symbol{
                name: p.param_name.clone(),
                location: f.location.clone(),
                original_name: None,
            }, LinkageType::Internal)?;
        }
        match &f.body {
            None => {
                Ok(Function {
                    location: f.location.clone(),
                    name: f.name.clone(),
                    params: resolved_params,
                    body: None,
                })
            }
            Some(func_definition) => {
                collect_labels_from_block(sub_ctx, func_definition)?;
                resolve_block(sub_ctx, func_definition)
                    .map(|resolved_block| {
                        Function {
                            location: f.location.clone(),
                            name: f.name.clone(),
                            params: resolved_params,
                            body: Some(resolved_block),
                        }
                    })
            }
        }

    })
}

fn resolve_type_expression(_: &mut IdentifierResolutionContext, ty_expr: &TypeExpression) -> Result<TypeExpression, IdentifierResolutionError> {
    let loc = ty_expr.location.clone();
    match &ty_expr.kind {
        TypeExpressionKind::Primitive(p) => Ok(TypeExpression {
            location: loc,
            kind: TypeExpressionKind::Primitive(p.clone()),
        })
    }
}

fn collect_labels_from_block(ctx: &mut IdentifierResolutionContext, block: &Block) -> Result<(), IdentifierResolutionError> {
    for blk_item in block.items.iter() {
        if let BlockItem::Statement(statement) = blk_item {
            collect_labels_from_statement(ctx, statement)?;
        }
    }
    Ok(())
}

fn collect_labels_from_statement(ctx: &mut IdentifierResolutionContext, statement: &Statement) -> Result<(), IdentifierResolutionError> {
    let loc = statement.location.clone();
    for lbl in statement.labels.iter() {
        ctx.add_label(lbl.clone(), loc)?;
    }
    match &statement.kind {
        StatementKind::If { then_statement, else_statement, .. } => {
            collect_labels_from_statement(ctx, then_statement)?;
            if let Some(ref else_stmt) = else_statement {
                collect_labels_from_statement(ctx, else_stmt)?;
            }
        }
        StatementKind::SubBlock(sub_block) => collect_labels_from_block(ctx, sub_block)?,
        _ => {}
    }
    Ok(())
}

fn resolve_block<'a>(ctx: &mut IdentifierResolutionContext, block: &Block) -> Result<Block, IdentifierResolutionError> {
    let mut resolved_blk_items = Vec::with_capacity(block.items.len());
    for blk_item in block.items.iter() {
        let resolved_blk_item = resolve_block_item(ctx, blk_item)?;
        resolved_blk_items.push(resolved_blk_item);
    }
    Ok(Block {
        start_loc: block.start_loc.clone(),
        end_loc: block.end_loc.clone(),
        items: resolved_blk_items,
    })
}

fn resolve_block_item<'a>(ctx: &mut IdentifierResolutionContext, blk_item: &BlockItem) -> Result<BlockItem, IdentifierResolutionError> {
    match blk_item {
        BlockItem::Statement(stmt) => {
            let resolved_stmt = resolve_statement(ctx, stmt)?;
            Ok(BlockItem::Statement(resolved_stmt))
        },
        BlockItem::Declaration(decl) => {
            let resolved_decl = resolve_declaration(ctx, decl)?;
            Ok(BlockItem::Declaration(resolved_decl))
        },
    }
}

fn resolve_statement(ctx: &mut IdentifierResolutionContext, stmt: &Statement) -> Result<Statement, IdentifierResolutionError> {
    let loc = stmt.location.clone();
    let mut resolved_labels = Vec::with_capacity(stmt.labels.len());
    for lbl in stmt.labels.iter() {
        resolved_labels.push(ctx.get_resolved_label(lbl)?);
    }
    match &stmt.kind {
        StatementKind::Return(ret_val_expr) => {
            let resolved_ret_val_expr = resolve_expression(ctx, ret_val_expr)?;
            Ok(Statement {
                location: loc.clone(),
                labels: resolved_labels,
                kind: StatementKind::Return(resolved_ret_val_expr),
            })
        },
        StatementKind::Expression(expr) => {
            let resolved_expr = resolve_expression(ctx, expr)?;
            Ok(Statement {
                location: loc.clone(),
                labels: resolved_labels,
                kind: StatementKind::Expression(resolved_expr),
            })
        },
        StatementKind::SubBlock(sub_block) => {
            ctx.with_scope(|sub_ctx| {
                let resolved_subblock = resolve_block(sub_ctx, sub_block)?;
                Ok(Statement {
                    location: loc.clone(),
                    labels: resolved_labels.clone(),
                    kind: StatementKind::SubBlock(resolved_subblock),
                })
            })
        },
        StatementKind::If { condition, then_statement, else_statement } => {
            let resolved_condition = resolve_expression(ctx, &*condition)?;
            let resolved_then = resolve_statement(ctx, &*then_statement)?;
            let resolved_else = match else_statement {
                None => None,
                Some(else_stmt) => {
                    let resolved_stmt = resolve_statement(ctx, &*else_stmt)?;
                    Some(Box::new(resolved_stmt))
                },
            };
            Ok(Statement {
                location: loc,
                labels: resolved_labels,
                kind: StatementKind::If {
                    condition: Box::new(resolved_condition),
                    then_statement: Box::new(resolved_then),
                    else_statement: resolved_else,
                },
            })
        },
        StatementKind::Goto { target } => Ok(Statement {
            location: loc,
            labels: resolved_labels,
            kind: StatementKind::Goto {
                target: ctx.get_resolved_label(target)?,
            },
        }),
        StatementKind::Null => Ok(Statement {
            location: loc.clone(),
            labels: resolved_labels,
            kind: StatementKind::Null,
        }),
        StatementKind::Break(lbl) => {
            debug_assert!(lbl.is_none());
            Ok(Statement {
                location: loc.clone(),
                labels: resolved_labels,
                kind: StatementKind::Break(None),
            })
        },
        StatementKind::Continue(lbl) => {
            debug_assert!(lbl.is_none());
            Ok(Statement {
                location: loc.clone(),
                labels: resolved_labels,
                kind: StatementKind::Continue(None),
            })
        },
        StatementKind::While { pre_condition, loop_body, loop_label } => {
            debug_assert!(loop_label.is_none());
            let resolved_precondition = resolve_expression(ctx, &*pre_condition)?;
            let resolved_loop_body = resolve_statement(ctx, &*loop_body)?;
            Ok(Statement {
                location: loc.clone(),
                labels: resolved_labels,
                kind: StatementKind::While {
                    pre_condition: Box::new(resolved_precondition),
                    loop_body: Box::new(resolved_loop_body),
                    loop_label: None,
                }
            })
        },
        StatementKind::DoWhile { loop_body, post_condition, loop_label } => {
            debug_assert!(loop_label.is_none());
            let resolved_loop_body = resolve_statement(ctx, &*loop_body)?;
            let resolved_post_condition = resolve_expression(ctx, &*post_condition)?;
            Ok(Statement {
                location: loc.clone(),
                labels: resolved_labels,
                kind: StatementKind::DoWhile {
                    loop_body: Box::new(resolved_loop_body),
                    post_condition: Box::new(resolved_post_condition),
                    loop_label: None,
                }
            })
        },
        StatementKind::For { init, condition, post, loop_body, loop_label } => {
            debug_assert!(loop_label.is_none());
            ctx.with_scope(|sub_ctx| {
                let resolved_init = match &init {
                    ForInit::InitDecl(decl) => {
                        let resolved_declaration = resolve_variable_declaration(sub_ctx, loc.clone(), &*decl)?;
                        ForInit::InitDecl(Box::new(resolved_declaration))
                    }
                    ForInit::InitExpr(expr) => {
                        let resolved_expression = resolve_expression(sub_ctx, &*expr)?;
                        ForInit::InitExpr(Box::new(resolved_expression))
                    }
                    ForInit::Null => ForInit::Null,
                };
                let resolved_condition = match &condition {
                    None => None,
                    Some(cond_expr) => {
                        let resolved_cond_expr = resolve_expression(sub_ctx, &*cond_expr)?;
                        Some(Box::new(resolved_cond_expr))
                    }
                };
                let resolved_post_loop = match &post {
                    None => None,
                    Some(post_expr) => {
                        let resolved_post_expr = resolve_expression(sub_ctx, &*post_expr)?;
                        Some(Box::new(resolved_post_expr))
                    }
                };
                let resolved_loop_body = resolve_statement(sub_ctx, &*loop_body)?;
                Ok(Statement {
                    location: loc.clone(),
                    labels: resolved_labels.clone(),
                    kind: StatementKind::For {
                        init: resolved_init,
                        condition: resolved_condition,
                        post: resolved_post_loop,
                        loop_body: Box::new(resolved_loop_body),
                        loop_label: None,
                    }
                })
            })
        },
    }
}

fn resolve_declaration(ctx: &mut IdentifierResolutionContext, decl: &Declaration) -> Result<Declaration, IdentifierResolutionError> {
    let decl_loc = decl.location.clone();
    match &decl.kind {
        DeclarationKind::VarDeclaration(var_decl) => {
            let resolved = resolve_variable_declaration(ctx, decl_loc.clone(), var_decl)?;
            Ok(Declaration { location: decl_loc, kind: DeclarationKind::VarDeclaration(resolved) })
        }
        DeclarationKind::FunctionDeclaration(func_decl) => {
            let resolved = resolve_function(ctx, func_decl)?;
            Ok(Declaration {
                location: decl_loc,
                kind: DeclarationKind::FunctionDeclaration(resolved),
            })
        },
    }
}

fn resolve_variable_declaration(ctx: &mut IdentifierResolutionContext, loc: Location, var_decl: &VariableDeclaration) -> Result<VariableDeclaration, IdentifierResolutionError> {
    let identifier = &var_decl.identifier;
    let prev_decl = ctx.get_resolved_identifier_in_current_scope(&identifier.name);
    if let Ok(prev_mapped) = prev_decl {
        return Err(IdentifierResolutionError::AlreadyDeclared {
            current_loc: loc.clone(),
            original_loc: prev_mapped.location(),
            name: identifier.name.clone(),
        });
    }
    let mapped = ctx.add_identifier_mapping(identifier.clone(), LinkageType::Internal)?;
    Ok(VariableDeclaration {
        identifier: mapped,
        init_expression: match &var_decl.init_expression {
            None => None,
            Some(expr) => Some(resolve_expression(ctx, expr)?),
        },
    })
}

fn resolve_expression<'a>(ctx: &mut IdentifierResolutionContext, expr: &Expression) -> Result<Expression, IdentifierResolutionError> {
    let loc = expr.location.clone();
    Ok(Expression {
        location: loc.clone(),
        kind: match &expr.kind {
            ExpressionKind::IntConstant(x, radix) => ExpressionKind::IntConstant(x.to_string(), *radix),
            ExpressionKind::Variable(v) => {
                let resolved = ctx.get_resolved_identifier(&v.name)?;
                ExpressionKind::Variable(resolved.symbol)
            },
            ExpressionKind::Unary(unary_op, op_expr) => {
                let resolved_expr = resolve_expression(ctx, op_expr)?;
                ExpressionKind::Unary(*unary_op, Box::new(resolved_expr))
            },
            ExpressionKind::Binary(binary_op, left_oper, right_oper) => {
                let resolved_left_oper = resolve_expression(ctx, left_oper)?;
                let resolved_right_oper = resolve_expression(ctx, right_oper)?;
                ExpressionKind::Binary(*binary_op, Box::new(resolved_left_oper), Box::new(resolved_right_oper))
            },
            ExpressionKind::Assignment { lvalue, rvalue, op } => {
                let result = if !lvalue.kind.is_lvalue_expression() {
                    Err(IdentifierResolutionError::LvalueExpected(loc.clone()))
                } else {
                    let resolved_rhs = {
                        let raw_rhs = resolve_expression(ctx, rvalue)?;
                        let rhs_loc = raw_rhs.location.clone();
                        match op {
                            None => raw_rhs,
                            Some(cat) => {
                                // todo: clean up double resolution of the same lvalue expression
                                //  This is done this way to make borrow-checker happy and not to
                                //  implement Clone on Expression type.
                                let desugared_rhs = ExpressionKind::Binary(
                                    (*cat).into(), Box::new(resolve_expression(ctx, lvalue)?), Box::new(raw_rhs));
                                Expression { location: rhs_loc, kind: desugared_rhs }
                            },
                        }
                    };
                    Ok(ExpressionKind::Assignment {
                        lvalue: Box::new(resolve_expression(ctx, lvalue)?),
                        rvalue: Box::new(resolved_rhs),
                        op: None,
                    })
                }?;
                result
            },
            ExpressionKind::Increment { is_post, e } => {
                if !e.kind.is_lvalue_expression() {
                    return Err(IdentifierResolutionError::LvalueExpected(e.location));
                }
                ExpressionKind::Increment {
                    is_post: is_post.clone(),
                    e: Box::new(resolve_expression(ctx, e)?),
                }
            },
            ExpressionKind::Decrement { is_post, e } => {
                if !e.kind.is_lvalue_expression() {
                    return Err(IdentifierResolutionError::LvalueExpected(e.location));
                }
                ExpressionKind::Decrement {
                    is_post: is_post.clone(),
                    e: Box::new(resolve_expression(ctx, e)?),
                }
            },
            ExpressionKind::Conditional { condition, then_expr, else_expr } => {
                let resolved_condition = resolve_expression(ctx, condition)?;
                let resolved_then_expr = resolve_expression(ctx, then_expr)?;
                let resolved_else_expr = resolve_expression(ctx, else_expr)?;
                ExpressionKind::Conditional {
                    condition: Box::new(resolved_condition),
                    then_expr: Box::new(resolved_then_expr),
                    else_expr: Box::new(resolved_else_expr),
                }
            },
            ExpressionKind::FunctionCall { func_name, actual_params } => {
                let resolved_func_name = ctx.get_resolved_identifier(&func_name.name.clone())?;
                let mut resolved_actual_params = Vec::with_capacity(actual_params.len());
                for ap in actual_params.iter() {
                    let resolved_param_expr = resolve_expression(ctx, ap)?;
                    resolved_actual_params.push(Box::new(resolved_param_expr));
                }
                ExpressionKind::FunctionCall {
                    func_name: resolved_func_name.symbol,
                    actual_params: resolved_actual_params,
                }
            },
        },
    })
}

#[cfg(test)]
mod test {
    use indoc::indoc;

    use crate::lexer::Lexer;
    use crate::parser::{Parser, Program};
    use crate::semantic_analysis::identifier_resolution::{IdentifierResolutionError, resolve_program};
    use crate::semantic_analysis::desugaring_verifier::desugared_compound_assignment;
    use crate::semantic_analysis::unique_identifier_verifier::program_identifiers_are_unique;

    #[test]
    fn test_should_error_on_use_before_declaration() {
        let program = indoc!{r#"
        int main(void) {
            a = 1;
            int a;
            return a;
        }
        "#};
        let resolved_ast = run_program_resolution(program);
        assert!(resolved_ast.is_err());

        let IdentifierResolutionError::NotFound { name } = resolved_ast.unwrap_err() else { panic!("unexpected error") };
        assert_eq!(name, "a".to_string());
    }

    #[test]
    fn test_should_error_on_redeclaration_in_same_scope() {
        let program = indoc!{r#"
        int main(void) {
            int a = 1;
            int a = 2;
            return a;
        }
        "#};
        let resolved_ast = run_program_resolution(program);
        let IdentifierResolutionError::AlreadyDeclared { name, current_loc: _, original_loc: _ } = resolved_ast.unwrap_err() else {
            panic!("unexpected error");
        };
        assert_eq!(name, "a".to_string());
    }

    #[test]
    fn test_should_resolve_simple_declaration() {
        let program = indoc! {r#"
        int main(void) {
            int a;
            a = 1;
            return a;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_resolve_shadowed_variable_in_inner_scope() {
        let program = indoc! {r#"
        int main(void) {
            int a = 1;
            {
                int a = 2;
                return a;
            }
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_resolve_multiple_variables() {
        let program = indoc! {r#"
        int main(void) {
            int a = 1;
            int b = 2;
            return a + b;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_error_on_undeclared_variable_use() {
        let program = indoc! {r#"
        int main(void) {
            return x;
        }
        "#};
        let resolved_ast = run_program_resolution(program);
        let IdentifierResolutionError::NotFound { name } = resolved_ast.unwrap_err() else {
            panic!("unexpected error");
        };
        assert_eq!(name, "x");
    }

    #[test]
    fn test_should_error_on_lvalue_expected() {
        let program = indoc! {r#"
        int main(void) {
            1 = 2;
            return 0;
        }
        "#};
        let resolved_ast = run_program_resolution(program);
        let IdentifierResolutionError::LvalueExpected(location) = resolved_ast.unwrap_err() else {
            panic!("unexpected error");
        };
        assert_eq!(location, (2,5).into());
    }

    #[test]
    fn test_use_after_scope_exit() {
        let program = indoc! {r#"
        int main(void) {
            {
                int a = 1;
            }
            a = 2;
        }
        "#};
        let resolved_ast = run_program_resolution(program);
        assert!(resolved_ast.is_err());

        let IdentifierResolutionError::NotFound { name } = resolved_ast.unwrap_err() else {
            panic!("unexpected error type");
        };
        assert_eq!(name, "a");
    }

    #[test]
    fn test_increment_nonlvalue_int_is_not_allowed() {
        let program = indoc! {r#"
        int main(void) {
            10++;
        }
        "#};
        let resolved_ast = run_program_resolution(program);
        let IdentifierResolutionError::LvalueExpected(_location) = resolved_ast.unwrap_err() else {
            panic!("unexpected error");
        };
    }

    #[test]
    fn test_increment_nonlvalue_expression_is_not_allowed() {
        let program = indoc! {r#"
        int main(void) {
            int a = 10;
            int b = 20;
            (a+b)++;
            return a+b;
        }
        "#};

        let resolved_ast = run_program_resolution(program);
        let IdentifierResolutionError::LvalueExpected(_location) = resolved_ast.unwrap_err() else {
            panic!("unexpected error");
        };
    }

    #[test]
    fn test_increment_nonlvalue_unary_expression_is_not_allowed() {
        let program = indoc! {r#"
        int main(void) {
            (!10)++;
        }
        "#};
        let resolved_ast = run_program_resolution(program);
        let IdentifierResolutionError::LvalueExpected(_location) = resolved_ast.unwrap_err() else {
            panic!("unexpected error");
        };
    }

    #[test]
    fn test_should_allow_shadowing_in_multiple_scopes() {
        let program = indoc!{r#"
        int main(void) {
            int a = 10;
            {
                int a = 20;
            }
            return a;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_resolve_as_different_variables_in_different_scopes() {
        let program = indoc!{r#"
        int main(void) {
            int a = 10;
            {
                int b = 20;
            }
            {
                int b = 30;
            }
            return a;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_resolve_as_different_variables_in_different_functions() {
        let program = indoc!{r#"
        int main(void) {
            int a = 10;
            return a;
        }

        int foo(void) {
            int a = 20;
            return a;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_resolve_previously_seen_function_names() {
        let program = indoc!{r#"
        int foo(void) {
            return 10;
        }

        int bar(void) {
            int a = 2;
            return a;
        }

        int main(void) {
            foo; bar;
            return 0;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_desugar_compound_assignment_add() {
        let program = indoc!{r#"
        int main(void) {
            int a = 10;
            a += 10;
            return a;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_error_if_any_one_arm_of_ternary_is_not_lvalue() {
        let program = indoc!{r#"
        int main(void) {
            int a = 10;
            int b = 20;
            a > b ? a : a+b = 100;
            return a;
        }
        "#};
        let resolved_ast = run_program_resolution(program);
        let IdentifierResolutionError::LvalueExpected(_location) = resolved_ast.unwrap_err() else {
            panic!("unexpected error");
        };
    }

    #[test]
    fn test_should_resolve_correctly_if_both_arms_of_ternary_in_assignment_are_lvalue() {
        let program = indoc!{r#"
        int main(void) {
            int a = 10;
            int b = 20;
            a > b ? a : b = 100;
            return a;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_resolve_identifiers_in_if_blocks_correctly() {
        let program = indoc!{r#"
        int main(void) {
            int a = 10;
            if (a > 10) {
                int b = 20;
                return a + b;
            } else {
                int c = 10;
                return a + c;
            }
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_resolve_shadowing_in_different_arms_of_if_statement_correctly() {
        let program = indoc!{r#"
        int main(void) {
            int a = 10;
            if ( a > 10 ) {
                int a = 20;
                int x = 20;
                return a + x + 1;
            }
            if (a >= 10) {
                int a = 2;
                return a;
            } else {
                int b = 2;
                a += b;
            }
            return a;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_resolve_labels_correctly_simple() {
        let program = indoc!{r#"
        int main(void) {
            int x = 10;
            assign_1: x = 1;
            return x;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_resolve_variable_name_and_labels_correctly_when_they_are_the_same() {
        let program = indoc!{r#"
        int main(void) {
            int x = 10;
            x: x = 1;
            return x;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_resolve_labels_correctly_forward_ref() {
        let program = indoc!{r#"
        int main(void) {
            int x = 10;
            int errno = 0;
            if (x < 10)
                goto err;
            return x;
        err:
            errno = -1;
            return errno;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_resolve_same_label_in_different_functions_correctly() {
        let program = indoc!{r#"
        int foo(void) {
        x: return 10;
        }

        int bar(void) {
        x: return 1;
        }
        "#};
        assert_successful_identifier_resolution(program);
    }

    #[test]
    fn test_should_error_when_same_label_is_reused_within_a_function() {
        let program = indoc!{r#"
        int main(void) {
            int x = 10;
            a: x += 1;
            a: x += 2;
            return x;
        }
        "#};
        let resolv_result = run_program_resolution(program);
        assert!(resolv_result.is_err());
        let IdentifierResolutionError::LabelAlreadyUsed { .. } = resolv_result.unwrap_err() else {
            panic!("unexpected error")
        };
    }

    #[test]
    fn test_should_error_when_same_label_is_reused_in_a_different_block_in_the_same_function() {
        let program = indoc!{r#"
        int main(void) {
            int x = 10;
            if (x > 10) {
                a: x += 1;
            } else {
                a: x += 2;
            }
            return x;
        }
        "#};
        let resolv_result = run_program_resolution(program);
        println!("error: {:?}", resolv_result);
        assert!(resolv_result.is_err());
        let IdentifierResolutionError::LabelAlreadyUsed { .. } = resolv_result.unwrap_err() else {
            panic!("unexpected error");
        };
    }

    #[test]
    fn test_should_error_when_undeclared_label_is_used_in_the_goto_statement() {
        let program = indoc!{r#"
        int main(void) {
            int x = 10;
            goto y;
            return x;
        }
        "#};
        let resolv_result = run_program_resolution(program);
        assert!(resolv_result.is_err());
        let IdentifierResolutionError::LabelNotDeclared { .. } = resolv_result.unwrap_err() else {
            panic!("unexpected error")
        };
    }

    #[test]
    fn test_should_error_when_goto_target_is_declared_in_a_different_function() {
        let program = indoc!{r#"
        int main(void) {
            int x = 10;
            goto y;
            return x;
        }

        int foo(void) {
            y: return 1;
        }
        "#};
        let resolv_result = run_program_resolution(program);
        assert!(resolv_result.is_err());
        let IdentifierResolutionError::LabelNotDeclared { .. } = resolv_result.unwrap_err() else {
            panic!("unexpected error")
        };
    }

    #[test]
    fn test_should_error_when_function_is_defined_inside_another_function() {
        let program = indoc!{r#"
        int main(void) {
            int foo(void) {
                return 10;
            }
            return 0;
        }
        "#};
        let resolv_result = run_program_resolution(program);
        assert!(resolv_result.is_err());
        let IdentifierResolutionError::CannotDefineFunctionInsideAnotherFunction { .. } = resolv_result.unwrap_err() else {
            panic!("unexpected error")
        };
    }

    #[test]
    fn test_should_error_when_function_is_declared_more_than_once() {
        let program = indoc!{r#"
        int main(void) {
            return 10;
        }

        int foo(void) {
            return 10;
        }

        int foo(void) {
            return 20;
        }
        "#};
        let resolv_result = run_program_resolution(program);
        assert!(resolv_result.is_err());
        let IdentifierResolutionError::CannotRedefineFunction { .. } = resolv_result.unwrap_err() else {
            panic!("unexpected error")
        };
    }

    #[test]
    fn test_multiple_function_declarations_are_allowed() {
        let program = indoc!{r#"
        int foo(int a, int b);

        int main(void) {
            return 0;
        }

        int foo(int a0, int b0);
        "#};
        let resolv_result = run_program_resolution(program);
        assert!(resolv_result.is_ok(), "{:#?}", resolv_result);
    }

    #[test]
    fn test_function_declaration_inside_function_is_allowed() {
        let program = indoc! {r#"
        int foo(int a);

        int main(void) {
            int foo(int a);  // inner declaration
            return 0;
        }

        int foo(int a) { return a; }
        "#};
        let resolv_result = run_program_resolution(program);
        assert!(resolv_result.is_ok(), "{:#?}", resolv_result);
    }

    #[test]
    fn test_multiple_duplicate_function_declarations_inside_function() {
        let program = indoc! {r#"
        int bar(int x);

        int main(void) {
            int bar(int y);
            int bar(int y0);  // duplicate declaration
            return 0;
        }
        "#};
        let resolv_result = run_program_resolution(program);
        assert!(resolv_result.is_ok(), "{:#?}", resolv_result);
    }

    #[test]
    fn test_redefining_variable_as_function_in_the_same_scope_is_illegal() {
        let program = indoc! {r#"
        int main(void) {
            int foo = 1;
            int foo(void);
            return foo;
        }

        int foo(void) {
            return 1;
        }
        "#};
        let resolv_result = run_program_resolution(program);
        assert!(resolv_result.is_err());
        let IdentifierResolutionError::CannotRedeclareVariableAsFunction { .. } = resolv_result.unwrap_err() else {
            panic!("unexpected error")
        };
    }

    #[test]
    fn test_redefining_variable_as_function_in_a_nested_scope_is_legal() {
        let program = indoc! {r#"
        int main(void) {
            int foo = 1;
            {
                int foo(void);
            }
            return foo;
        }

        int foo(void) {
            return 1;
        }
        "#};
        let resolv_result = run_program_resolution(program);
        assert!(resolv_result.is_ok());
    }

    fn assert_successful_identifier_resolution(program: &str) {
        let resolved = run_program_resolution(program);
        assert!(resolved.is_ok(), "{:#?}", resolved);

        let resolved_program = resolved.unwrap();
        assert!(program_identifiers_are_unique(&resolved_program));
        assert!(desugared_compound_assignment(&resolved_program));
    }

    fn run_program_resolution(program: &str) -> Result<Program, IdentifierResolutionError> {
        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());
        let resolved_ast = resolve_program(parsed.unwrap());
        resolved_ast
    }
}