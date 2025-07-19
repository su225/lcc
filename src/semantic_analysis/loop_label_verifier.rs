use std::collections::HashSet;
use crate::parser::{BlockItem, Declaration, DeclarationKind, Function, Program, Statement, StatementKind};

pub fn loop_labels_are_complete_and_unique(p: &Program) -> bool {
    let mut label_set: HashSet<String> = HashSet::new();
    p.declarations.iter().all(|f| declaration_loop_labels_are_complete_and_unique(&mut label_set, f))
}

fn declaration_loop_labels_are_complete_and_unique(label_set: &mut HashSet<String>, decl: &Declaration) -> bool {
    match &decl.kind {
        DeclarationKind::FunctionDeclaration(ref f) => function_loop_labels_are_complete_and_unique(label_set, f),
        _ => true,
    }
}

fn function_loop_labels_are_complete_and_unique(label_set: &mut HashSet<String>, f: &Function) -> bool {
    f.body.as_ref().is_some_and(|func_body| func_body.items.iter().all(|bi| block_item_loop_labels_are_complete_and_unique(label_set, bi)))
}

fn block_item_loop_labels_are_complete_and_unique(label_set: &mut HashSet<String>, block_item: &BlockItem) -> bool {
    if let BlockItem::Statement(ref stmt) = block_item {
        statement_loop_labels_are_complete_and_unique(label_set, stmt)
    } else {
        true
    }
}

fn statement_loop_labels_are_complete_and_unique(label_set: &mut HashSet<String>, stmt: &Statement) -> bool {
    match &stmt.kind {
        StatementKind::Break(loop_label)
        | StatementKind::Continue(loop_label)
        | StatementKind::While { loop_label, .. }
        | StatementKind::DoWhile { loop_label, .. }
        | StatementKind::For { loop_label, .. } => {
            if let Some(ref lbl) = loop_label {
                label_set.insert(lbl.clone())
            } else {
                false
            }
        },
        _ => true,
    }
}
