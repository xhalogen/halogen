use super::*;
use proc_macro2::Ident as Ident2;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::HashMap;
use syn::{Error, Result};

pub fn get_size(idx: &Ident, size_map: &HashMap<String, (Ident2, usize)>) -> Option<TokenStream2> {
    if let Some((rep_shval, rep_axis)) = size_map.get(&idx.to_string()) {
        return Some(quote! {(#rep_shval)[#rep_axis]});
    }
    None
}

pub fn get_size_map(tensors: &[TensorValue]) -> HashMap<String, (Ident2, usize)> {
    let mut size_map = HashMap::<String, (Ident2, usize)>::new();
    for (i, t) in tensors.iter().enumerate() {
        let shval = get_shape_i(i);
        for (axis, idx) in t.right_indices.iter().enumerate() {
            if get_size(idx, &size_map).is_none() {
                size_map.insert(idx.to_string(), (shval.clone(), axis));
            }
        }
    }
    size_map
}

pub fn gen_size_checks(
    tensors: &[TensorValue],
    size_map: &HashMap<String, (Ident2, usize)>,
) -> Result<TokenStream2> {
    let mut ret = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        let shval = get_shape_i(i);
        for (axis, idx) in t.right_indices.iter().enumerate() {
            let idx_len = get_size(idx, size_map).ok_or_else(|| {
                Error::new_spanned(idx, "internal error: input index not found in size map")
            })?;
            let idx_name = idx.to_string();
            ret.push(quote! {
                ::core::assert!(
                    (#shval)[#axis] == #idx_len,
                    "einsum: dimension mismatch for index `{}` ({} != {})",
                    #idx_name,
                    (#shval)[#axis],
                    #idx_len
                );
            });
        }
    }
    Ok(quote! { #(#ret)* })
}
