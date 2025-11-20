use thiserror::Error;
use crate::common::Location;
use crate::parser::{Block, BlockItem, Declaration, DeclarationKind, Function, Program, Statement, StatementKind};

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

pub fn loop_label_program_definition(program: Program) -> Result<Program, LoopLabelingError> {
    let mut ctx = LoopLabelingContext::new();
    let mut labeled_decls = Vec::with_capacity(program.declarations.len());
    for decl in program.declarations {
        let decl_loc = decl.location.clone();
        if let DeclarationKind::FunctionDeclaration(f) = decl.kind {
            let labeled_f = loop_label_function(&mut ctx, f)?;
            labeled_decls.push(Declaration {
                location: decl_loc,
                kind: DeclarationKind::FunctionDeclaration(labeled_f),
            });
        } else {
            labeled_decls.push(decl);
        }
    }
    Ok(Program { declarations: labeled_decls })
}

fn loop_label_function(ctx: &mut LoopLabelingContext, function: Function) -> Result<Function, LoopLabelingError> {
    let loop_labeled_body = function.body
        .map(|func_body| loop_label_block(ctx, func_body))
        .transpose()?;
    Ok(Function {
        location: function.location,
        name: function.name,
        params: function.params,
        body: loop_labeled_body,
    })
}

fn loop_label_block(ctx: &mut LoopLabelingContext, block: Block) -> Result<Block, LoopLabelingError> {
    let mut labeled_blk_item = Vec::with_capacity(block.items.len());
    for blk_item in block.items.into_iter() {
        if let BlockItem::Statement(stmt) = blk_item {
            let labeled_stmt = loop_label_statement(ctx, stmt)?;
            debug_assert!(loop_statement_is_labeled(&labeled_stmt));
            labeled_blk_item.push(BlockItem::Statement(labeled_stmt));
        } else {
            labeled_blk_item.push(blk_item);
        }
    }
    Ok(Block {
        start_loc: block.start_loc,
        end_loc: block.end_loc,
        items: labeled_blk_item,
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
        StatementKind::If { condition, then_statement, else_statement } => {
            let labeled_then = loop_label_statement(ctx, *then_statement)?;
            let labeled_else = else_statement
                .map(|else_stmt| loop_label_statement(ctx, *else_stmt).map(|s| Box::new(s)))
                .transpose()?;
            Ok(Statement {
                location: stmt_loc,
                labels: stmt.labels,
                kind: StatementKind::If {
                    condition,
                    then_statement: Box::new(labeled_then),
                    else_statement: labeled_else,
                },
            })
        },
        StatementKind::SubBlock(sub_block) => {
            Ok(Statement {
                location: stmt_loc,
                labels: stmt.labels,
                kind: StatementKind::SubBlock(loop_label_block(ctx, sub_block)?),
            })
        }
        _ => unreachable!("is_loop_labeling_required() should have returned false"),
    }
}

fn is_loop_labeling_required(stmt: &Statement) -> bool {
    match &stmt.kind {
        StatementKind::Break(_)
        | StatementKind::Continue(_)
        | StatementKind::For {..}
        | StatementKind::While {..}
        | StatementKind::DoWhile {..}
        | StatementKind::If {..}
        | StatementKind::SubBlock(_) => true,
        _ => false,
    }
}

#[cfg(test)]
mod test {
    use indoc::indoc;
    use crate::lexer::Lexer;
    use crate::parser::{Parser, Program};
    use crate::semantic_analysis::identifier_resolution::{resolve_program};
    use crate::semantic_analysis::loop_label_verifier::loop_labels_are_complete_and_unique;
    use crate::semantic_analysis::loop_labeling::{loop_label_program_definition, LoopLabelingError};

    #[test]
    fn test_label_for_loop_correctly() {
        let program = indoc!{r#"
        int main(void) {
            int x = 0;
            for (int i = 0; i < 10; i++)
                x += i;
            return 0;
        }
        "#};
        assert_successful_loop_labeling(program);
    }

    #[test]
    fn test_label_nested_for_loop_correctly() {
        let program = indoc!{r#"
        int main(void) {
            int x = 0;
            for (int i = 0; i < 10; i++)
                for (int j = 0; j < 10; j++)
                    x += (i + j);
            return 0;
        }
        "#};
        assert_successful_loop_labeling(program);
    }

    #[test]
    fn test_label_while_loop_correctly() {
        let program = indoc!{r#"
        int main(void) {
            int x = 0;
            while (x < 10) x++;
            return 0;
        }
        "#};
        assert_successful_loop_labeling(program);
    }

    #[test]
    fn test_label_do_while_loop_correctly() {
        let program = indoc!{r#"
        int main(void) {
            int x = 0;
            do x++; while (x < 10);
            return 0;
        }
        "#};
        assert_successful_loop_labeling(program);
    }

    #[test]
    fn test_label_break_statement_inside_loop() {
        let program = indoc!{r#"
        int main(void) {
            int x = 0;
            for (int i = 0; i < 10; i++)
                break;
            return 0;
        }
        "#};
        assert_successful_loop_labeling(program);
    }

    #[test]
    fn test_label_continue_statement_inside_loop() {
        let program = indoc!{r#"
        int main(void) {
            int x = 0;
            for (int i = 0; i < 10; i++)
                continue;
            return 0;
        }
        "#};
        assert_successful_loop_labeling(program);
    }

    #[test]
    fn test_label_break_statement_outside_loop_must_error() {
        let program = indoc!{r#"
        int main(void) {
            if (1)
                break;
            return 0;
        }
        "#};
        let res = run_program_loop_labeling(program);
        assert!(res.is_err(), "{:#?}", res);
        let LoopLabelingError::BreakOutsideLoop { .. } = res.unwrap_err() else {
            panic!("unexpected error")
        };
    }

    #[test]
    fn test_label_continue_statement_outside_loop_must_error() {
        let program = indoc!{r#"
        int main(void) {
            int x = 0;
            continue;
            return 0;
        }
        "#};
        let res = run_program_loop_labeling(program);
        assert!(res.is_err());
        let LoopLabelingError::ContinueOutsideLoop { .. } = res.unwrap_err() else {
            panic!("unexpected error")
        };
    }

    fn assert_successful_loop_labeling(program: &str) {
        let labeled = run_program_loop_labeling(program);
        assert!(labeled.is_ok(), "{:#?}", labeled);

        let lbl = labeled.unwrap();
        assert!(loop_labels_are_complete_and_unique(&lbl));
    }

    fn run_program_loop_labeling(program: &str) -> Result<Program, LoopLabelingError> {
        let lexer = Lexer::new(program);
        let mut parser = Parser::new(lexer);
        let parsed = parser.parse();
        assert!(parsed.is_ok());
        let resolved_ast = resolve_program(parsed.unwrap());
        let loop_labeled_ast = loop_label_program_definition(resolved_ast.unwrap());
        loop_labeled_ast
    }
}