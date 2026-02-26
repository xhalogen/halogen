use super::{TensorIndices, defv::*, is_tensorvalue};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::HashMap;
use syn::{
    Error, Expr, Ident, Result,
    fold::{self, Fold},
    parse_quote,
};

pub fn get_einsum_expr(indices: &TensorIndices, expr: &Expr) -> Result<TokenStream2> {
    let mut idx_prc = HashMap::<String, Ident>::new();
    for (i, idx) in indices.inner.iter().enumerate() {
        idx_prc.insert(idx.to_string(), get_inner_i(i));
    }
    for (i, idx) in indices.outer.iter().enumerate() {
        idx_prc.insert(idx.to_string(), get_outer_i(i));
    }
    let mut folder = TensorValueFolder {
        idx_prc: &idx_prc,
        error: None,
    };
    let folded = folder.fold_expr(expr.clone());
    if let Some(err) = folder.error {
        return Err(err);
    }
    Ok(quote! { #folded })
}

struct TensorValueFolder<'a> {
    idx_prc: &'a HashMap<String, Ident>,
    error: Option<Error>,
}

impl<'a> Fold for TensorValueFolder<'a> {
    fn fold_expr(&mut self, expr: Expr) -> Expr {
        let tensor_value = if let Expr::Index(expridx) = &expr {
            is_tensorvalue(expridx)
        } else {
            None
        };

        if let Some(t) = tensor_value {
            let result: Result<Vec<Ident>> =
                t.right_indices
                    .iter()
                    .map(|x| {
                        self.idx_prc.get(&x.to_string()).cloned().ok_or_else(|| {
                            Error::new_spanned(x, "index not found in einsum index map")
                        })
                    })
                    .collect();

            match result {
                Err(e) => {
                    self.error = Some(e);
                    return expr;
                }
                Ok(indices) => {
                    let tensor = t.left_tensors;
                    return parse_quote! {
                        (#tensor).at(&[ #(#indices),* ])
                    };
                }
            }
        }

        fold::fold_expr(self, expr)
    }
}
