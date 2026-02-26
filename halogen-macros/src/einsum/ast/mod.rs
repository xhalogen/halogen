use syn::{Expr, ExprIndex, Ident};

pub mod folder;
pub mod viewer;

pub struct TensorValue {
    pub left_tensors: Box<Expr>,
    pub right_indices: Vec<Ident>,
}

pub fn is_tensorvalue(expridx: &ExprIndex) -> Option<TensorValue> {
    let tensor = expridx.expr.clone();

    let indices = match expridx.index.as_ref() {
        Expr::Tuple(x) => x.elems.iter().map(is_index).collect::<Option<Vec<_>>>()?,
        x => vec![is_index(x)?],
    };

    Some(TensorValue {
        left_tensors: tensor,
        right_indices: indices,
    })
}

fn is_index(expr: &Expr) -> Option<Ident> {
    match expr {
        Expr::Paren(x) => is_index(&x.expr),
        Expr::Group(x) => is_index(&x.expr),
        Expr::Path(x) if x.qself.is_none() && x.path.segments.len() == 1 => {
            Some(x.path.segments[0].ident.clone())
        }
        _ => None,
    }
}
