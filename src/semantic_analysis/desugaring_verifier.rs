use crate::parser::{Block, BlockItem, Declaration, DeclarationKind, Expression, ExpressionKind, ForInit, Function, Program, Statement, StatementKind};

pub fn desugared_compound_assignment(prog: &Program) -> bool {
    prog.declarations.iter().all(|f| desugared_compound_assignment_in_declaration(f))
}

fn desugared_compound_assignment_in_functions(func: &Function) -> bool {
    func.body.as_ref().is_none_or(|func_body| desugared_compound_assignment_in_blocks(func_body))
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
        StatementKind::Break(_) => true,
        StatementKind::Continue(_) => true,

        StatementKind::While { pre_condition, loop_body, .. } =>
            desugared_compound_assignment_in_expression(pre_condition)
                && desugared_compound_assignment_in_statement(loop_body),

        StatementKind::DoWhile { loop_body, post_condition, .. } =>
            desugared_compound_assignment_in_expression(post_condition)
                && desugared_compound_assignment_in_statement(loop_body),

        StatementKind::For { init, condition, post, loop_body, .. } =>
            desugared_compound_assignment_in_for_init(init)
                && condition.as_ref().is_none_or(|e| desugared_compound_assignment_in_expression(&*e))
                && post.as_ref().is_none_or(|e| desugared_compound_assignment_in_expression(&*e))
                && desugared_compound_assignment_in_statement(&*loop_body),
    }
}

fn desugared_compound_assignment_in_for_init(for_init: &ForInit) -> bool {
    match &for_init {
        ForInit::InitDecl(_) => true,
        ForInit::InitExpr(e) => desugared_compound_assignment_in_expression(e),
        ForInit::Null => true,
    }
}

fn desugared_compound_assignment_in_declaration(decl: &Declaration) -> bool {
    match &decl.kind {
        DeclarationKind::VarDeclaration { init_expression: Some(init_expr), .. } =>
            desugared_compound_assignment_in_expression(init_expr),
        DeclarationKind::VarDeclaration { .. } => true,
        DeclarationKind::FunctionDeclaration(ref f) => desugared_compound_assignment_in_functions(f),
    }
}

fn desugared_compound_assignment_in_expression(expr: &Expression) -> bool {
    match &expr.kind {
        ExpressionKind::Assignment { op: Some(_), .. } => false,
        _ => true,
    }
}