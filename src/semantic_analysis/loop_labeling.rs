use thiserror::Error;
use crate::common::Location;
use crate::parser::{Block, BlockItem, FunctionDefinition, ProgramDefinition, Statement, StatementKind};

#[derive(Debug, Error)]
pub enum LoopLabelingError {
    #[error("{location:?}: break outside loop")]
    BreakOutsideLoop { location: Location },

    #[error("{location:?}: continue outside loop")]
    ContinueOutsideLoop { location: Location },
}

struct LoopLabelingContext {
    next_id: usize,
    loop_labels: Vec<String>,
}

impl LoopLabelingContext {
    fn new() -> Self {
        LoopLabelingContext {
            next_id: 0,
            loop_labels: vec![],
        }
    }

    fn with_loop<T, F>(&mut self, f: F) -> Result<T, LoopLabelingError>
    where
        F: FnOnce(&mut LoopLabelingContext) -> Result<T, LoopLabelingError>
    {
        let label = self.generate_loop_label();
        self.loop_labels.push(label);
        let result = { f(self) };
        debug_assert!(self.loop_labels.pop().is_some());
        result
    }

    fn current_loop(&self) -> Option<String> {
        self.loop_labels.last().cloned()
    }

    fn generate_loop_label(&mut self) -> String {
        let num_id = self.next_id;
        self.next_id += 1;
        format!(".loop.{}", num_id)
    }
}

pub fn loop_label_program_definition(program: ProgramDefinition) -> Result<ProgramDefinition, LoopLabelingError> {
    let mut ctx = LoopLabelingContext::new();
    let mut labeled_funcs = Vec::with_capacity(program.functions.len());
    for f in program.functions {
        let labeled_f = loop_label_function(&mut ctx, f)?;
        labeled_funcs.push(labeled_f);
    }
    Ok(ProgramDefinition { functions: labeled_funcs })
}

fn loop_label_function(ctx: &mut LoopLabelingContext, function: FunctionDefinition) -> Result<FunctionDefinition, LoopLabelingError> {
    let mut labeled_blk_item = Vec::with_capacity(function.body.items.len());
    for blk_item in function.body.items {
        if let BlockItem::Statement(stmt) = blk_item {
            let labeled_stmt = loop_label_statement(ctx, stmt)?;
            debug_assert!(loop_statement_is_labeled(&labeled_stmt));
            labeled_blk_item.push(BlockItem::Statement(labeled_stmt));
        } else {
            labeled_blk_item.push(blk_item);
        }
    }
    Ok(FunctionDefinition {
        location: function.location,
        name: function.name,
        body: Block {
            start_loc: function.body.start_loc,
            end_loc: function.body.end_loc,
            items: labeled_blk_item,
        },
    })
}

fn loop_statement_is_labeled(stmt: &Statement) -> bool {
    match &stmt.kind {
        StatementKind::For { loop_label, .. }
        | StatementKind::DoWhile { loop_label, .. }
        | StatementKind::While { loop_label, .. }
        | StatementKind::Break(loop_label)
        | StatementKind::Continue(loop_label) => loop_label.is_some(),
        _ => true,
    }
}

fn loop_label_statement(ctx: &mut LoopLabelingContext, stmt: Statement) -> Result<Statement, LoopLabelingError> {
    let stmt_loc = stmt.location;
    if !is_loop_labeling_required(&stmt) {
        return Ok(stmt);
    }
    match stmt.kind {
        StatementKind::While { pre_condition, loop_body, .. } => {
            ctx.with_loop(move |sub_ctx| {
                let labeled_body = loop_label_statement(sub_ctx, *loop_body)?;
                Ok(Statement {
                    location: stmt_loc,
                    labels: stmt.labels,
                    kind: StatementKind::While {
                        pre_condition,
                        loop_body: Box::new(labeled_body),
                        loop_label: sub_ctx.current_loop(),
                    },
                })
            })
        },
        StatementKind::DoWhile { loop_body, post_condition, .. } => {
            ctx.with_loop(move |sub_ctx| {
                let labeled_body = loop_label_statement(sub_ctx, *loop_body)?;
                Ok(Statement {
                    location: stmt_loc,
                    labels: stmt.labels,
                    kind: StatementKind::DoWhile {
                        loop_body: Box::new(labeled_body),
                        post_condition,
                        loop_label: sub_ctx.current_loop(),
                    },
                })
            })
        },
        StatementKind::For { init, condition, post, loop_body, .. } => {
            ctx.with_loop(move |sub_ctx| {
                let labeled_body = loop_label_statement(sub_ctx, *loop_body)?;
                Ok(Statement {
                    location: stmt_loc,
                    labels: stmt.labels,
                    kind: StatementKind::For {
                        init,
                        condition,
                        post,
                        loop_body: Box::new(labeled_body),
                        loop_label: sub_ctx.current_loop(),
                    }
                })
            })
        },
        StatementKind::Break(_) => {
            ctx.current_loop()
                .map(|loop_label| Statement {
                    location: stmt_loc,
                    labels: stmt.labels,
                    kind: StatementKind::Break(Some(loop_label)),
                })
                .ok_or(LoopLabelingError::BreakOutsideLoop {location: stmt_loc})
        },
        StatementKind::Continue(_) => {
            ctx.current_loop()
                .map(|loop_label| Statement {
                    location: stmt_loc,
                    labels: stmt.labels,
                    kind: StatementKind::Continue(Some(loop_label)),
                })
                .ok_or(LoopLabelingError::ContinueOutsideLoop {location: stmt_loc})
        },
        _ => unreachable!("is_loop_labeling_required() should have returned false"),
    }
}

fn is_loop_labeling_required(stmt: &Statement) -> bool {
    match &stmt.kind {
        StatementKind::Break(_)
        | StatementKind::Continue(_)
        | StatementKind::For {..}
        | StatementKind::While {..}
        | StatementKind::DoWhile {..} => true,
        _ => false,
    }
}