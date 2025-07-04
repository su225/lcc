use std::collections::HashSet;
use crate::parser::{BlockItem, FunctionDefinition, ProgramDefinition, Statement, StatementKind};

pub fn loop_labels_are_complete_and_unique(p: &ProgramDefinition) -> bool {
    let mut label_set: HashSet<String> = HashSet::new();
    p.functions.iter().all(|f| function_loop_labels_are_complete_and_unique(&mut label_set, f))
}

fn function_loop_labels_are_complete_and_unique(label_set: &mut HashSet<String>, f: &FunctionDefinition) -> bool {
    f.body.items.iter().all(|bi| block_item_loop_labels_are_complete_and_unique(label_set, bi))
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
