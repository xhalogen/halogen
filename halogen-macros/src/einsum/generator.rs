use crate::einsum::{TensorIndices, TensorValue, defv::*};
use proc_macro2::Ident as Ident2;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::HashMap;
use syn::{Error, Ident, Result};

pub fn def_inputvalues(tensors: &[TensorValue]) -> TokenStream2 {
    let ret = tensors.iter().enumerate().map(|(i, t)| {
        let tsval = get_input_i(i);
        let shval = get_shape_i(i);
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

pub fn def_outputvalues(
    indices: &TensorIndices,
    size_map: &HashMap<String, (Ident2, usize)>,
) -> Result<TokenStream2> {
    let shape_v = get_shape_v();
    let data_v = get_data_v();
    let acc_v = get_acc_v();
    let tmp_v = get_tmp_v();
    // shape와 data 정의
    let mut ret = Vec::new();
    let mut shape_tok = Vec::new();
    for idx in &indices.outer {
        let idx_len = get_size(idx, size_map)
            .ok_or_else(|| Error::new_spanned(idx, "expression must contain every output index"))?;
        shape_tok.push(quote! {#idx_len,});
    }
    ret.push(quote! {
        let #shape_v = [#(#shape_tok)*];
    });
    ret.push(quote! {
        let mut #data_v = ::std::vec![::core::default::Default::default(); #shape_v.iter().product()];
    });
    // 스트라이드 계산 도중 누적되는 변수
    ret.push(quote! {
        let mut #acc_v: usize = 1;
    });
    // 스트라이드 변수 정의
    for (i, _) in indices.outer.iter().enumerate().rev() {
        let stride_v = get_stride_i(i);
        ret.push(quote! {
            let mut #stride_v: usize = 0;
        });
    }
    // 스트라이드 계산
    for (i, idx) in indices.outer.iter().enumerate().rev() {
        let stride_v = get_stride_i(i);
        let idx_len = get_size(idx, size_map).expect("expression must contain every output index");
        ret.push(quote! {
            #stride_v += #acc_v;
            #acc_v *= #idx_len;
        });
    }
    // 스트라이드 보정 (검증 필요)
    ret.push(quote! {
        let mut #tmp_v: usize = 0;
        #acc_v = 0;
    });
    for (i, idx) in indices.outer.iter().enumerate().rev() {
        let stride_v = get_stride_i(i);
        let idx_len = get_size(idx, size_map).expect("expression must contain every output index");
        ret.push(quote! {
            #tmp_v = #stride_v;
            #stride_v -= #acc_v;
            #acc_v = #idx_len * (#acc_v + #tmp_v);
        });
    }
    Ok(quote! { #(#ret)* })
}

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
