use super::*;
use proc_macro2::Ident as Ident2;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::HashMap;
use syn::{Error, Result};

pub fn def_inputvars(tensors: &[TensorValue]) -> TokenStream2 {
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

pub fn def_outputvars(
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
