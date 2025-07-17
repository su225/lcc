use std::collections::HashSet;
use crate::parser::{Block, BlockItem, Declaration, DeclarationKind, ForInit, Function, ProgramDefinition, Statement, StatementKind};

pub fn program_identifiers_are_unique(prog: &ProgramDefinition) -> bool {
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

fn function_identifiers_are_unique(identifiers: &mut HashSet<String>, f: &Function) -> bool {
    f.body.as_ref().is_some_and(|func_body| block_identifiers_are_unique(identifiers, func_body))
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
        | StatementKind::Null
        | StatementKind::Goto {..}
        | StatementKind::Break(_)
        | StatementKind::Continue(_) => true,
        StatementKind::If {then_statement, else_statement, ..} => {
            let then_uniq = statement_identifiers_are_unique(identifiers, then_statement);
            let else_uniq = else_statement.as_ref()
                .is_none_or(|e| statement_identifiers_are_unique(identifiers, &*e));
            then_uniq && else_uniq
        },
        StatementKind::SubBlock(sb) => block_identifiers_are_unique(identifiers, sb),
        StatementKind::While { loop_body, .. } => statement_identifiers_are_unique(identifiers, loop_body),
        StatementKind::For { init, loop_body, .. } => {
            let is_init_uniq = match &init {
                ForInit::InitDecl(d) => declaration_identifiers_are_unique(identifiers, &*d),
                ForInit::InitExpr(_) | ForInit::Null => true,
            };
            let is_loop_body_uniq = statement_identifiers_are_unique(identifiers, &*loop_body);
            is_init_uniq && is_loop_body_uniq
        },
        StatementKind::DoWhile { loop_body, .. } => statement_identifiers_are_unique(identifiers, loop_body),
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
        DeclarationKind::FunctionDeclaration(_) => todo!(),
    }
}