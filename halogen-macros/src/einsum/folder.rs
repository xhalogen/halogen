use crate::einsum::{TensorIndices, is_tensorvalue};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{
    Expr, Ident,
    fold::{self, Fold},
    parse_quote,
};

pub fn get_einsum_expr(indices: &TensorIndices, expr: &Expr) -> TokenStream2 {
    let mut idx_prc = HashMap::<Ident, Ident>::new();
    for (i, idx) in indices.input.iter().enumerate() {
        idx_prc.insert(idx.clone(), format_ident!("__halogen_einsum_iter_in{i}"));
    }
    for (i, idx) in indices.output.iter().enumerate() {
        idx_prc.insert(idx.clone(), format_ident!("__halogen_einsum_iter_out{i}"));
    }
    let mut folder = TensorValueFolder { idx_prc: &idx_prc };
    let folded = folder.fold_expr(expr.clone());
    quote! { #folded }
}

struct TensorValueFolder<'a> {
    idx_prc: &'a HashMap<Ident, Ident>,
}

impl<'a> Fold for TensorValueFolder<'a> {
    fn fold_expr(&mut self, expr: Expr) -> Expr {
        let expr = fold::fold_expr(self, expr);
        if let Expr::Index(expridx) = &expr
            && let Some(t) = is_tensorvalue(expridx)
        {
            let tensor = t.tensor;
            let indices: Vec<Ident> = t
                    .indices
                    .iter()
                    .map(|x| {
                        self.idx_prc
                            .get(x)
                            .cloned()
                            .expect(
                                "fold_expr: something went wrong I will write err msg later (input idx_rep)"
                            )
                    })
                    .collect();
            return parse_quote! {
                (#tensor).at(&[ #(#indices),* ])
            };
        }
        expr
    }
}
