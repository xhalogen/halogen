use crate::einsum::{TensorIndices, is_tensorvalue};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{
    Error, Expr, Ident, Result,
    fold::{self, Fold},
    parse_quote,
};

pub fn get_einsum_expr(indices: &TensorIndices, expr: &Expr) -> Result<TokenStream2> {
    let mut idx_prc = HashMap::<String, Ident>::new();
    for (i, idx) in indices.input.iter().enumerate() {
        idx_prc.insert(
            idx.to_string(),
            format_ident!("__halogen_einsum_iter_in{i}"),
        );
    }
    for (i, idx) in indices.output.iter().enumerate() {
        idx_prc.insert(
            idx.to_string(),
            format_ident!("__halogen_einsum_iter_out{i}"),
        );
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
                t.indices
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
                    let tensor = t.tensor;
                    return parse_quote! {
                        (#tensor).at(&[ #(#indices),* ])
                    };
                }
            }
        }

        fold::fold_expr(self, expr)
    }
}
