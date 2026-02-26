use super::{TensorIndices, TensorValue, defv::*};
use proc_macro2::Ident as Ident2;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::HashMap;
use syn::{Error, Ident, Result};

pub mod vars;
pub use vars::*;
pub mod size;
pub use size::*;

pub fn gen_einsum_loop(
    indices: &TensorIndices,
    einsum_expr: TokenStream2,
    size_map: &HashMap<String, (Ident2, usize)>,
) -> Result<TokenStream2> {
    let offset_v = get_offset_v();
    let data_v = get_data_v();
    // updating result tensor datas
    let mut quotes = quote! {
        #data_v[#offset_v] += #einsum_expr;
    };
    // for loops for inner indices (indices which is not indices of result tensor)
    for (i, idx) in indices.inner.iter().enumerate().rev() {
        let idx_len = get_size(idx, size_map).ok_or_else(|| {
            Error::new_spanned(idx, "internal error: input index not found in size map")
        })?;
        let idx_v = get_inner_i(i);
        quotes = quote! {
            for #idx_v in (0..#idx_len) {
                #quotes
            }
        };
    }
    // for loops for outer indices (indices of result tensor)
    for (i, idx) in indices.outer.iter().enumerate().rev() {
        let idx_len = get_size(idx, size_map).ok_or_else(|| {
            Error::new_spanned(idx, "internal error: output index not found in size map")
        })?;
        let idx_v = get_outer_i(i);
        let stride_v = get_stride_i(i);
        quotes = quote! {
            for #idx_v in (0..#idx_len) {
                #quotes
                #offset_v += #stride_v;
            }
        };
    }
    // returning result tensor
    Ok(quotes)
}

pub fn gen_einsum_output(code: TokenStream2) -> TokenStream2 {
    let offset_v = get_offset_v();
    let shape_v = get_shape_v();
    let data_v = get_data_v();
    let from_vec_v = get_from_vec_v();
    let t_v = get_t_v();
    let input0 = get_input_i(0);
    quote! {
        let mut #offset_v: usize = 0;
        #code
        fn #from_vec_v<#t_v: crate::core::tensor::Tensor>(
            _: &#t_v,
            shape: &[usize],
            data: ::std::vec::Vec<#t_v::Elem>,
        ) -> ::std::result::Result<#t_v, crate::core::TensorError> {
            #t_v::from_vec(shape, data)
        }
        #from_vec_v(
            #input0,
            &#shape_v,
            #data_v,
        ).unwrap()
    }
}
