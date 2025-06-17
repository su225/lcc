use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use crate::common::Location;
use crate::parser::{Block, BlockItem, Declaration, Expression, ExpressionKind, FunctionDefinition, ProgramDefinition, Statement, StatementKind, Symbol};

#[derive(Debug, Error)]
enum IdentifierResolutionError<'a> {
    #[error("{location:?}: identifier '{name:?}' not found")]
    NotFound { location: Location, name: &'a str },

    #[error("{current_loc:?}: identifier '{name:?}' already declared at {original_loc:?}")]
    AlreadyDeclared { current_loc: Location, original_loc: Location, name: &'a str },
}

struct IdentifierResolutionContext<'a> {
    parent: Option<Arc<IdentifierResolutionContext<'a>>>,
    identifier_map: HashMap<Symbol<'a>, Symbol<'a>>,
    next_num_id: u64,
}

impl<'a> IdentifierResolutionContext<'a> {
    fn new() -> Self {
        return IdentifierResolutionContext {
            parent: None,
            identifier_map: HashMap::new(),
            next_num_id: 0,
        }
    }

    fn with_parent(p: Arc<IdentifierResolutionContext<'a>>) -> Self {
        return IdentifierResolutionContext {
            parent: Some(p),
            identifier_map: HashMap::new(),
            next_num_id: 0,
        }
    }

    fn add_identifier_mapping(&mut self, ident: Symbol<'a>) -> Result<Symbol<'a>, IdentifierResolutionError<'a>> {
        todo!()
    }

    fn get_resolved_identifier(&self, raw_ident: Symbol<'a>) -> Result<Symbol<'a>, IdentifierResolutionError<'a>> {
        todo!()
    }
}

fn resolve_program(program: ProgramDefinition) -> Result<ProgramDefinition, IdentifierResolutionError> {
    let mut ctx = Arc::new(IdentifierResolutionContext::new());
    let mut resolved_funcs = Vec::with_capacity(program.functions.len());
    for f in program.functions.iter() {
        let resolved_f = resolve_function(ctx.clone(), f)?;
        resolved_funcs.push(resolved_f);
    }
    Ok(ProgramDefinition { functions: resolved_funcs })
}

fn resolve_function<'a>(ctx: Arc<IdentifierResolutionContext<'a>>, f: &FunctionDefinition<'a>) -> Result<FunctionDefinition<'a>, IdentifierResolutionError<'a>> {
    let mut f_ctx = IdentifierResolutionContext::with_parent(ctx);
    f_ctx.add_identifier_mapping(f.name.clone())?;

    let f_ctx_arc = Arc::new(f_ctx);
    resolve_block(f_ctx_arc, &f.body).map(|resolved_block| {
        FunctionDefinition {
            location: f.location.clone(),
            name: f.name.clone(),
            body: resolved_block,
        }
    })

}

fn resolve_block<'a>(ctx: Arc<IdentifierResolutionContext<'a>>, block: &Block<'a>) -> Result<Block<'a>, IdentifierResolutionError<'a>> {
    let mut resolved_blk_items = Vec::with_capacity(block.items.len());
    for blk_item in block.items.iter() {
        let resolved_blk_item = resolve_block_item(ctx.clone(), blk_item)?;
        resolved_blk_items.push(resolved_blk_item);
    }
    Ok(Block {
        start_loc: block.start_loc.clone(),
        end_loc: block.end_loc.clone(),
        items: resolved_blk_items,
    })
}

fn resolve_block_item<'a>(ctx: Arc<IdentifierResolutionContext<'a>>, blk_item: &BlockItem<'a>) -> Result<BlockItem<'a>, IdentifierResolutionError<'a>> {
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

fn resolve_statement<'a>(ctx: Arc<IdentifierResolutionContext<'a>>, stmt: &Statement<'a>) -> Result<Statement<'a>, IdentifierResolutionError<'a>> {
    let loc = stmt.location.clone();
    match &stmt.kind {
        StatementKind::Return(ret_val_expr) => {
            let resolved_ret_val_expr = resolve_expression(ctx, ret_val_expr)?;
            Ok(Statement { location: loc.clone(), kind: StatementKind::Return(resolved_ret_val_expr) })
        },
        StatementKind::Expression(expr) => {
            let resolved_expr = resolve_expression(ctx, expr)?;
            Ok(Statement { location: loc.clone(), kind: StatementKind::Expression(resolved_expr) })
        },
        StatementKind::SubBlock(sub_block) => {
            let sub_block_ctx = Arc::new(IdentifierResolutionContext::with_parent(ctx));
            let resolved_subblock = resolve_block(sub_block_ctx, sub_block)?;
            Ok(Statement { location: loc.clone(), kind: StatementKind::SubBlock(resolved_subblock) })
        },
        StatementKind::Null => Ok(Statement { location: loc.clone(), kind: StatementKind::Null })
    }
}

fn resolve_declaration<'a>(ctx: Arc<IdentifierResolutionContext<'a>>, decl: &Declaration<'a>) -> Result<Declaration<'a>, IdentifierResolutionError<'a>> {
    todo!()
}

fn resolve_expression<'a>(ctx: Arc<IdentifierResolutionContext<'a>>, expr: &Expression<'a>) -> Result<Expression<'a>, IdentifierResolutionError<'a>> {
    let loc = expr.location.clone();
    match &expr.kind {
        ExpressionKind::IntConstant(num, radix) => Ok(Expression {
            location: loc.clone(),
            kind: ExpressionKind::IntConstant(num, radix.clone()),
        }),
        ExpressionKind::Variable(v) => todo!(),
        ExpressionKind::Unary(unary_op, expr) => todo!(),
        ExpressionKind::Binary(binary_op, lhs, rhs) => todo!(),
        ExpressionKind::Assignment { lvalue, rvalue } => todo!(),
    }
}