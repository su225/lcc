use std::collections::HashMap;
use thiserror::Error;
use crate::common::Location;
use crate::parser::{Block, BlockItem, Declaration, DeclarationKind, Expression, ExpressionKind, FunctionDefinition, ProgramDefinition, Statement, StatementKind, Symbol};

#[derive(Debug, Error)]
pub enum IdentifierResolutionError {
    #[error("{location:?}: identifier '{name:?}' not found")]
    NotFound { location: Location, name: String },

    #[error("{current_loc:?}: identifier '{name:?}' already declared at {original_loc:?}")]
    AlreadyDeclared { current_loc: Location, original_loc: Location, name: String },

    #[error("{0:?}: lvalue expected")]
    LvalueExpected(Location),
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

struct IdentifierResolutionContext {
    scopes: Vec<Scope>,
    labels: Scope,
    next_num_id: u64,
}

impl IdentifierResolutionContext {
    fn new() -> Self {
        return IdentifierResolutionContext {
            scopes: vec![Scope::new()],
            labels: Scope::new(),
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
    ctx.with_scope(|sub_ctx| {
        resolve_block(sub_ctx, &f.body).map(|resolved_block| {
            FunctionDefinition {
                location: f.location.clone(),
                name: f.name.clone(),
                body: resolved_block,
            }
        })
    })
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

fn resolve_statement<'a>(ctx: &mut IdentifierResolutionContext, stmt: &Statement) -> Result<Statement, IdentifierResolutionError> {
    let loc = stmt.location.clone();
    match &stmt.kind {
        StatementKind::Return(ret_val_expr) => {
            let resolved_ret_val_expr = resolve_expression(ctx, ret_val_expr)?;
            Ok(Statement { location: loc.clone(), label: None, kind: StatementKind::Return(resolved_ret_val_expr) })
        },
        StatementKind::Expression(expr) => {
            let resolved_expr = resolve_expression(ctx, expr)?;
            Ok(Statement { location: loc.clone(), label: None, kind: StatementKind::Expression(resolved_expr) })
        },
        StatementKind::SubBlock(sub_block) => {
            ctx.with_scope(|sub_ctx| {
                let resolved_subblock = resolve_block(sub_ctx, sub_block)?;
                Ok(Statement { location: loc.clone(), label: None, kind: StatementKind::SubBlock(resolved_subblock) })
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
                location: loc.clone(),
                label: None,
                kind: StatementKind::If {
                    condition: Box::new(resolved_condition),
                    then_statement: Box::new(resolved_then),
                    else_statement: resolved_else,
                },
            })
        },
        StatementKind::Goto { .. } => todo!("label must be defined within the function"),
        StatementKind::Null => Ok(Statement { location: loc.clone(), label: None, kind: StatementKind::Null })
    }
}

fn resolve_declaration<'a>(ctx: &mut IdentifierResolutionContext, decl: &Declaration) -> Result<Declaration, IdentifierResolutionError> {
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
    use std::collections::HashSet;
    use indoc::indoc;
    use crate::lexer::Lexer;
    use crate::parser::{Block, BlockItem, Declaration, DeclarationKind, Expression, ExpressionKind, FunctionDefinition, Parser, ProgramDefinition, Statement, StatementKind};
    use crate::semantic_analysis::identifier_resolution::{IdentifierResolutionError, resolve_program};

    #[test]
    fn test_should_error_on_use_before_declaration() {
        let program = indoc!{r#"
        int main(void) {
            a = 1;
            int a;
            return a;
        }
        "#};
        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());

        let resolved_ast = resolve_program(parsed.unwrap());
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
        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());

        let resolved_ast = resolve_program(parsed.unwrap());
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
        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());

        let resolved_ast = resolve_program(parsed.unwrap());
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
        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());

        let resolved_ast = resolve_program(parsed.unwrap());
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
        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());

        let resolved_ast = resolve_program(parsed.unwrap());
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

        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());
        let resolved_ast = resolve_program(parsed.unwrap());
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

        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());
        let resolved_ast = resolve_program(parsed.unwrap());
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

        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());
        let resolved_ast = resolve_program(parsed.unwrap());
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
        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());
        let resolved_ast = resolve_program(parsed.unwrap());
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

    fn assert_successful_identifier_resolution(program: &str) {
        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());
        let resolved = resolve_program(parsed.unwrap());
        assert!(resolved.is_ok(), "{:#?}", resolved);

        let resolved_program = resolved.unwrap();
        assert!(program_identifiers_are_unique(&resolved_program));
        assert!(desugared_compound_assignment(&resolved_program));
    }
    
    fn desugared_compound_assignment(prog: &ProgramDefinition) -> bool {
        prog.functions.iter().all(|f| desugared_compound_assignment_in_functions(f))
    }

    fn desugared_compound_assignment_in_functions(func: &FunctionDefinition) -> bool {
        desugared_compound_assignment_in_blocks(&func.body)
    }

    fn desugared_compound_assignment_in_blocks(block: &Block) -> bool {
        block.items.iter().all(|blk_item| desugared_compound_assignment_in_block_items(blk_item))
    }

    fn desugared_compound_assignment_in_block_items(block_item: &BlockItem) -> bool {
        match block_item {
            BlockItem::Statement(stmt) => desugared_compound_assignment_in_statement(stmt),
            BlockItem::Declaration(decl) => desugared_compound_assignment_in_declaration(decl),
        }
    }

    fn desugared_compound_assignment_in_statement(stmt: &Statement) -> bool {
        match &stmt.kind {
            StatementKind::Return(ret_expr) => desugared_compound_assignment_in_expression(ret_expr),
            StatementKind::Expression(expr) => desugared_compound_assignment_in_expression(expr),
            StatementKind::SubBlock(blk) => desugared_compound_assignment_in_blocks(blk),
            StatementKind::If { condition, then_statement, else_statement } => {
                desugared_compound_assignment_in_expression(condition)
                    && desugared_compound_assignment_in_statement(then_statement)
                    && else_statement.as_ref().map(
                        |else_stmt| desugared_compound_assignment_in_statement(else_stmt))
                        .unwrap_or(true)
            }
            StatementKind::Goto {..} => true,
            StatementKind::Null => true,
        }
    }

    fn desugared_compound_assignment_in_declaration(decl: &Declaration) -> bool {
        match &decl.kind {
            DeclarationKind::Declaration { init_expression: Some(init_expr), .. } =>
                desugared_compound_assignment_in_expression(init_expr),
            _ => true,
        }
    }

    fn desugared_compound_assignment_in_expression(expr: &Expression) -> bool {
        match &expr.kind {
            ExpressionKind::Assignment { op: Some(_), .. } => false,
            _ => true,
        }
    }

    fn program_identifiers_are_unique(prog: &ProgramDefinition) -> bool {
        let mut function_names = HashSet::with_capacity(prog.functions.len());
        let mut identifiers = HashSet::new();
        for f in prog.functions.iter() {
            if function_names.contains(&f.name.name) {
                return false;
            }
            function_names.insert(f.name.name.clone());
            let unique_func_identifiers = function_identifiers_are_unique(&mut identifiers, f);
            if !unique_func_identifiers {
                return false;
            }
        }
        return true;
    }

    fn function_identifiers_are_unique(identifiers: &mut HashSet<String>, f: &FunctionDefinition) -> bool {
        block_identifiers_are_unique(identifiers, &f.body)
    }

    fn block_identifiers_are_unique(identifiers: &mut HashSet<String>, b: &Block) -> bool {
        for bi in b.items.iter() {
            let are_unique = block_item_identifiers_are_unique(identifiers, bi);
            if !are_unique {
                return false;
            }
        }
        return true;
    }

    fn block_item_identifiers_are_unique(identifiers: &mut HashSet<String>, bi: &BlockItem) -> bool {
        match bi {
            BlockItem::Statement(s) => statement_identifiers_are_unique(identifiers, s),
            BlockItem::Declaration(d) => declaration_identifiers_are_unique(identifiers, d),
        }
    }

    fn statement_identifiers_are_unique(identifiers: &mut HashSet<String>, s: &Statement) -> bool {
        match &s.kind {
            StatementKind::Return(_)
            | StatementKind::Expression(_)
            | StatementKind::Null => true,
            StatementKind::If {then_statement, else_statement, ..} => {
                let then_uniq = statement_identifiers_are_unique(identifiers, then_statement);
                let else_uniq = match else_statement {
                    None => true,
                    Some(e) => statement_identifiers_are_unique(identifiers, e),
                };
                then_uniq && else_uniq
            },
            StatementKind::Goto {..} => true,
            StatementKind::SubBlock(sb) => block_identifiers_are_unique(identifiers, sb),
        }
    }

    fn declaration_identifiers_are_unique(identifiers: &mut HashSet<String>, d: &Declaration) -> bool {
        return match &d.kind {
            DeclarationKind::Declaration { identifier: ident, .. } => {
                if identifiers.contains(&ident.name) {
                    false
                } else {
                    identifiers.insert(ident.name.clone());
                    true
                }
            },
        }
    }
}