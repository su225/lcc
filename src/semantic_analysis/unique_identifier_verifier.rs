use std::collections::HashSet;
use crate::parser::{Block, BlockItem, Declaration, DeclarationKind, ForInit, Function, Program, Statement, StatementKind, VariableDeclaration};

pub fn program_identifiers_are_unique(prog: &Program) -> bool {
    let mut identifier_names = HashSet::with_capacity(prog.declarations.len());
    let mut identifiers = HashSet::new();
    for decl in prog.declarations.iter() {
        let ident_name = decl.identifier().name.clone();
        if identifier_names.contains(&ident_name) {
            return false;
        }
        identifier_names.insert(ident_name);
        let unique_func_identifiers = declaration_identifiers_are_unique(&mut identifiers, decl);
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
                ForInit::InitDecl(d) => {
                    let decl = Declaration {
                        location: (0,0).into(),
                        kind: DeclarationKind::VarDeclaration(VariableDeclaration {
                            identifier: d.identifier.clone(),
                            init_expression: None, // init expression does not matter in this case
                        }),
                    };
                    declaration_identifiers_are_unique(identifiers, &decl)
                },
                ForInit::InitExpr(_) | ForInit::Null => true,
            };
            let is_loop_body_uniq = statement_identifiers_are_unique(identifiers, &*loop_body);
            is_init_uniq && is_loop_body_uniq
        },
        StatementKind::DoWhile { loop_body, .. } => statement_identifiers_are_unique(identifiers, loop_body),
    }
}

fn declaration_identifiers_are_unique(identifiers: &mut HashSet<String>, d: &Declaration) -> bool {
    let ident = d.identifier().name.clone();
    let is_unique = identifiers.insert(ident);
    if !is_unique {
        return false;
    }
    match &d.kind {
        DeclarationKind::FunctionDeclaration(ref f) => function_identifiers_are_unique(identifiers, f),
        _ => true,
    }
}