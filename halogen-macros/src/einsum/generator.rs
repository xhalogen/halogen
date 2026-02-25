use crate::einsum::{TensorIndices, TensorValue};
use proc_macro2::Ident as Ident2;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{Error, Ident, Result};

pub fn def_inputvalues(tensors: &[TensorValue]) -> TokenStream2 {
    let ret = tensors.iter().enumerate().map(|(i, t)| {
        let tsval = format_ident!("__halogen_einsum_tensor{i}");
        let shval = format_ident!("__halogen_einsum_shape{i}");
        let expr = &t.left_tensors;
        quote! {
            let #tsval = &(#expr);
            let #shval = (#tsval).shape();
        }
    });
    quote! {#(#ret)*}
}

fn get_size(idx: &Ident, size_map: &HashMap<String, (Ident2, usize)>) -> Option<TokenStream2> {
    if let Some((rep_shval, rep_axis)) = size_map.get(&idx.to_string()) {
        return Some(quote! {(#rep_shval)[#rep_axis]});
    }
    None
}

pub fn get_size_map(tensors: &[TensorValue]) -> HashMap<String, (Ident2, usize)> {
    let mut size_map = HashMap::<String, (Ident2, usize)>::new();
    for (i, t) in tensors.iter().enumerate() {
        let shval = format_ident!("__halogen_einsum_shape{i}");
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
        let shval = format_ident!("__halogen_einsum_shape{i}");
        for (axis, idx) in t.right_indices.iter().enumerate() {
            let idxlen = get_size(idx, &size_map).ok_or_else(|| {
                Error::new_spanned(idx, "internal error: input index not found in size map")
            })?;
            let idx_name = idx.to_string();
            ret.push(quote! {
                ::core::assert!(
                    (#shval)[#axis] == #idxlen,
                    "einsum: dimension mismatch for index `{}` ({} != {})",
                    #idx_name,
                    (#shval)[#axis],
                    #idxlen
                );
            });
        }
    }
    Ok(quote! { #(#ret)* })
}

pub fn def_outputvalues(
    indices: &TensorIndices,
    right_indices: &[Ident],
    size_map: &HashMap<String, (Ident2, usize)>,
) -> Result<TokenStream2> {
    // shape와 data 정의
    let mut ret = Vec::new();
    ret.push(quote! {
        let mut __halogen_einsum_output_shape: ::std::vec::Vec<usize> = ::std::vec::Vec::new();
    });
    for idx in right_indices {
        let idxlen = get_size(idx, size_map)
            .ok_or_else(|| Error::new_spanned(idx, "expression must contain every output index"))?;
        ret.push(quote! {
            __halogen_einsum_output_shape.push(#idxlen);
        });
    }
    ret.push(quote! {
        let mut __halogen_einsum_output_data = ::std::vec![::core::default::Default::default(); __halogen_einsum_output_shape.iter().product()];
    });

    // 스트라이드 정의부
    let mut idx_prc = HashMap::<String, usize>::new();
    for (i, idx) in indices.outer.iter().enumerate() {
        idx_prc.insert(idx.to_string(), i);
    }
    // 스트라이드 계산 도중 누적되는 변수
    ret.push(quote! {
        let mut __halogen_einsum_stride_tmp: usize = 1;
    });
    // 스트라이드 변수 정의
    for i in (0..indices.outer.len()).rev() {
        let strd = format_ident!("__halogen_einsum_stride{i}");
        ret.push(quote! {
            let mut #strd: usize = 0;
        });
    }
    // 스트라이드 계산
    for idx in right_indices.iter().rev() {
        let key = *idx_prc
            .get(&idx.to_string())
            .expect("internal error: something went wrong while calculating stride variables");
        let strd = format_ident!("__halogen_einsum_stride{key}");
        let idxlen = get_size(idx, size_map).expect("expression must contain every output index");
        ret.push(quote! {
            #strd += __halogen_einsum_stride_tmp;
            __halogen_einsum_stride_tmp *= #idxlen;
        });
    }
    Ok(quote! { #(#ret)* })
}

pub fn gen_einsum_loop(
    indices: &TensorIndices,
    einsum_expr: TokenStream2,
    size_map: &HashMap<String, (Ident2, usize)>,
) -> Result<TokenStream2> {
    // updating result tensor datas
    let mut quotes = quote! {
        __halogen_einsum_output_data[__halogen_einsum_loop_index] += #einsum_expr;
    };
    // for loops for inner indices (indices which is not indices of result tensor)
    for (i, idx) in indices.inner.iter().enumerate().rev() {
        let idx_v = format_ident!("__halogen_einsum_inner{i}");
        let idxlen = get_size(idx, size_map).ok_or_else(|| {
            Error::new_spanned(idx, "internal error: input index not found in size map")
        })?;
        quotes = quote! {
            for #idx_v in (0..#idxlen) {
                #quotes
            }
        };
    }
    // for loops for outer indices (indices of result tensor)
    for (i, idx) in indices.outer.iter().enumerate().rev() {
        let idx_v = format_ident!("__halogen_einsum_outer{i}");
        let idxlen = get_size(idx, size_map).ok_or_else(|| {
            Error::new_spanned(idx, "internal error: output index not found in size map")
        })?;
        let stride_v = format_ident!("__halogen_einsum_stride{i}");
        let tmp_v = format_ident!("__halogen_einsum_memo{i}");

        quotes = quote! {
            for #idx_v in (0..#idxlen) {
                let #tmp_v = __halogen_einsum_loop_index;
                #quotes
                __halogen_einsum_loop_index = #tmp_v + #stride_v;
            }
        };
    }
    // returning result tensor
    quotes = quote! {
        let mut __halogen_einsum_loop_index: usize = 0;
        #quotes
        fn __halogen_einsum_make_output<__T: crate::core::tensor::Tensor>(
            _: &__T,
            shape: &[usize],
            data: ::std::vec::Vec<__T::Elem>,
        ) -> ::std::result::Result<__T, crate::core::TensorError> {
            __T::from_vec(shape, data)
        }
        let __halogen_einsum_output_tensor = __halogen_einsum_make_output(
            __halogen_einsum_tensor0,
            &__halogen_einsum_output_shape,
            __halogen_einsum_output_data,
        );
    };
    Ok(quotes)
}
