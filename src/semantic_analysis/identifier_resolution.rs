use std::collections::HashMap;

use thiserror::Error;

use crate::common::Location;
use crate::parser::{Block, BlockItem, Declaration, DeclarationKind, Expression, ExpressionKind, ForInit, FunctionDefinition, ProgramDefinition, Statement, StatementKind, Symbol};

#[derive(Debug, Error)]
pub enum IdentifierResolutionError {
    #[error("{location:?}: identifier '{name:?}' not found")]
    NotFound { location: Location, name: String },

    #[error("{current_loc:?}: identifier '{name:?}' already declared at {original_loc:?}")]
    AlreadyDeclared { current_loc: Location, original_loc: Location, name: String },

    #[error("{0:?}: lvalue expected")]
    LvalueExpected(Location),

    #[error("{cur_location:?} label '{label:?}' already used at {prev_location:?}")]
    LabelAlreadyUsed { prev_location: Location, cur_location: Location, label: String },

    #[error("label '{label:?}' not declared")]
    LabelNotDeclared { label: String }
}

struct Scope {
    identifier_map: HashMap<String, Symbol>,
}

impl Scope {
    fn new() -> Self {
        return Scope {
            identifier_map: HashMap::new(),
        }
    }

    fn add_mapping(&mut self, raw_ident: Symbol, mapped_ident: Symbol) -> Result<(), IdentifierResolutionError> {
        let existing_mapping = self.identifier_map.get(&raw_ident.name);
        if existing_mapping.is_some() {
            return Err(IdentifierResolutionError::AlreadyDeclared {
                current_loc: raw_ident.location.clone(),
                original_loc: existing_mapping.unwrap().location.clone(),
                name: raw_ident.name.clone(),
            })
        }
        self.identifier_map.insert(raw_ident.name, mapped_ident);
        Ok(())
    }

    fn lookup(&self, raw_ident: &Symbol) -> Result<Symbol, IdentifierResolutionError> {
        let existing_mapping = self.identifier_map.get(&raw_ident.name);
        match existing_mapping {
            Some(mapped_ident) => Ok(mapped_ident.clone()),
            None => Err(IdentifierResolutionError::NotFound {
                location: raw_ident.location.clone(),
                name: raw_ident.name.clone(),
            }),
        }
    }
}

struct ResolvedLabel {
    label: String,
    location: Location,
}

struct IdentifierResolutionContext {
    scopes: Vec<Scope>,
    labels: HashMap<String, ResolvedLabel>,
    next_num_id: u64,
}

impl IdentifierResolutionContext {
    fn new() -> Self {
        return IdentifierResolutionContext {
            scopes: vec![Scope::new()],
            labels: HashMap::new(),
            next_num_id: 0,
        }
    }

    fn add_identifier_mapping(&mut self, ident: Symbol) -> Result<Symbol, IdentifierResolutionError> {
        let next = self.next_num_id;
        self.next_num_id += 1;
        let mapped_ident = format!("{}${}", ident.name, next);
        let mapped_symbol = Symbol {
            name: mapped_ident,
            location: ident.location.clone(),
        };
        let current_scope = self.get_current_scope_mut();
        current_scope.add_mapping(ident, mapped_symbol.clone())?;
        Ok(mapped_symbol)
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

    fn get_resolved_identifier(&self, raw_ident: &Symbol) -> Result<Symbol, IdentifierResolutionError> {
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
        Err(IdentifierResolutionError::NotFound {
            location: raw_ident.location.clone(),
            name: raw_ident.name.clone(),
        })
    }

    fn get_resolved_identifier_in_current_scope(&self, raw_ident: &Symbol) -> Result<Symbol, IdentifierResolutionError> {
        let current_scope = self.get_current_scope();
        current_scope.lookup(raw_ident)
    }

    fn with_scope<T, F>(&mut self, f: F) -> Result<T, IdentifierResolutionError>
    where
        F: Fn(&mut IdentifierResolutionContext) -> Result<T, IdentifierResolutionError>
    {
        self.scopes.push(Scope::new());
        let result = { f(self) };
        self.scopes.pop();
        result
    }

    #[inline]
    fn with_function_scope<T, F>(&mut self, f: F) -> Result<T, IdentifierResolutionError>
    where
        F: Fn(&mut IdentifierResolutionContext) -> Result<T, IdentifierResolutionError>
    {
        self.labels.clear();
        self.with_scope(f)
    }

    fn get_current_scope(&self) -> &Scope {
        self.scopes.last().expect("expected at least one scope")
    }

    fn get_current_scope_mut(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("expected at least one scope")
    }
}

pub fn resolve_program(program: ProgramDefinition) -> Result<ProgramDefinition, IdentifierResolutionError> {
    let mut ctx = IdentifierResolutionContext::new();
    let mut resolved_funcs = Vec::with_capacity(program.functions.len());
    for f in program.functions.iter() {
        ctx.add_identifier_mapping(f.name.clone())?;
        let resolved_f = resolve_function(&mut ctx, f)?;
        resolved_funcs.push(resolved_f);
    }
    Ok(ProgramDefinition { functions: resolved_funcs })
}

fn resolve_function<'a>(ctx: &mut IdentifierResolutionContext, f: &FunctionDefinition) -> Result<FunctionDefinition, IdentifierResolutionError> {
    ctx.with_function_scope(|sub_ctx| {
        collect_labels_from_block(sub_ctx, &f.body)?;
        resolve_block(sub_ctx, &f.body).map(|resolved_block| {
            FunctionDefinition {
                location: f.location.clone(),
                name: f.name.clone(),
                body: resolved_block,
            }
        })
    })
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
                        let resolved_declaration = resolve_declaration(sub_ctx, &*decl)?;
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
        DeclarationKind::Declaration { identifier, init_expression } => {
            let prev_decl = ctx.get_resolved_identifier_in_current_scope(&identifier);
            if let Ok(prev_mapped) = prev_decl {
                return Err(IdentifierResolutionError::AlreadyDeclared {
                    current_loc: decl_loc.clone(),
                    original_loc: prev_mapped.location.clone(),
                    name: identifier.name.clone(),
                });
            }
            let mapped = ctx.add_identifier_mapping(identifier.clone())?;
            Ok(Declaration {
                location: decl_loc.clone(),
                kind: DeclarationKind::Declaration {
                    identifier: mapped,
                    init_expression: match init_expression {
                        None => None,
                        Some(expr) => Some(resolve_expression(ctx, expr)?),
                    },
                },
            })
        }
    }
}

fn resolve_expression<'a>(ctx: &mut IdentifierResolutionContext, expr: &Expression) -> Result<Expression, IdentifierResolutionError> {
    let loc = expr.location.clone();
    Ok(Expression {
        location: loc.clone(),
        kind: match &expr.kind {
            ExpressionKind::IntConstant(x, radix) => ExpressionKind::IntConstant(x.to_string(), *radix),
            ExpressionKind::Variable(v) => {
                let ident = Symbol { location: loc.clone(), name: v.to_string() };
                let resolved = ctx.get_resolved_identifier(&ident)?;
                ExpressionKind::Variable(resolved.name.clone())
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
        },
    })
}

#[cfg(test)]
mod test {
    use indoc::indoc;

    use crate::lexer::Lexer;
    use crate::parser::{Parser, ProgramDefinition};
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

        let IdentifierResolutionError::NotFound { location, name } = resolved_ast.unwrap_err() else { panic!("unexpected error") };
        assert_eq!(name, "a".to_string());
        assert_eq!(location, (2,5).into());
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
        let IdentifierResolutionError::NotFound { name, location } = resolved_ast.unwrap_err() else {
            panic!("unexpected error");
        };
        assert_eq!(name, "x");
        assert_eq!(location, (2,12).into());
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

        let IdentifierResolutionError::NotFound { location, name } = resolved_ast.unwrap_err() else {
            panic!("unexpected error type");
        };
        assert_eq!(name, "a");
        assert_eq!(location, (5,5).into()); // location of `a = 2;`
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

    fn assert_successful_identifier_resolution(program: &str) {
        let resolved = run_program_resolution(program);
        assert!(resolved.is_ok(), "{:#?}", resolved);

        let resolved_program = resolved.unwrap();
        assert!(program_identifiers_are_unique(&resolved_program));
        assert!(desugared_compound_assignment(&resolved_program));
    }

    fn run_program_resolution(program: &str) -> Result<ProgramDefinition, IdentifierResolutionError> {
        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());
        let resolved_ast = resolve_program(parsed.unwrap());
        resolved_ast
    }
}