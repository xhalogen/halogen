use crate::einsum::{TensorIndices, TensorValue};
use proc_macro2::Ident as Ident2;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::Ident;

pub fn def_tensorvalues(tensors: &[TensorValue]) -> TokenStream2 {
    let ret = tensors.iter().enumerate().map(|(i, t)| {
        let tsval = format_ident!("__halogen_einsum_tensor{i}");
        let shval = format_ident!("__halogen_einsum_shape{i}");
        let expr = &t.tensor;
        quote! {
            let #tsval = &(#expr);
            let #shval = (#tsval).shape();
        }
    });
    quote! {#(#ret)*}
}

pub fn get_rep_size(
    idx: &Ident,
    idx_rep: &HashMap<String, (Ident2, usize)>,
) -> Option<TokenStream2> {
    if let Some((rep_shval, rep_axis)) = idx_rep.get(&idx.to_string()) {
        return Some(quote! {(#rep_shval)[#rep_axis]});
    }
    None
}

pub fn gen_size_checks(
    tensors: &[TensorValue],
) -> (TokenStream2, HashMap<String, (Ident2, usize)>) {
    let mut idx_rep = HashMap::<String, (Ident2, usize)>::new();
    let mut ret = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        let shval = format_ident!("__halogen_einsum_shape{i}");
        for (axis, idx) in t.indices.iter().enumerate() {
            if let Some(idxlen) = get_rep_size(idx, &idx_rep) {
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
            } else {
                idx_rep.insert(idx.to_string(), (shval.clone(), axis));
            }
        }
    }
    (quote! { #(#ret)* }, idx_rep)
}

pub fn def_outputvalues(
    exprresult: &[Ident],
    idx_rep: &HashMap<String, (Ident2, usize)>,
) -> TokenStream2 {
    let mut ret = Vec::new();
    ret.push(quote! {
        let mut __halogen_einsum_output_shape: ::std::vec::Vec<usize> = ::std::vec::Vec::new();
    });
    for idx in exprresult {
        let idxlen =
            get_rep_size(idx, idx_rep).expect("expression must contain every output index");
        ret.push(quote! {
            __halogen_einsum_output_shape.push(#idxlen);
        });
    }
    ret.push(quote! {
        let __halogen_einsum_output_data_len: usize = __halogen_einsum_output_shape.iter().product();
        let mut __halogen_einsum_output_data = ::std::vec![0; __halogen_einsum_output_data_len];
        // let mut __halogen_einsum_output_stride ::std::vec::Vec<usize> = ::std::vec::Vec::new();
        // if !__halogen_einsum_output_shape.is_empty() {
        //     __halogen_einsum_output_stride.resize(__halogen_einsum_output_shape.len(), 0);
        //     __halogen_einsum_output_stride[__halogen_einsum_output_shape.len()-1] = 1;
        //     for i in (0..__halogen_einsum_output_shape.len()-1).rev() {
        //         __halogen_einsum_output_stride[i]
        //             = __halogen_einsum_output_stride[i+1]
        //             * __halogen_einsum_output_shape[i+1];
        //     }
        // }
    });
    quote! { #(#ret)* }
}

pub fn gen_run_einsum(
    indices: &TensorIndices,
    einsum_expr: TokenStream2,
    idx_rep: &HashMap<String, (Ident2, usize)>,
) -> TokenStream2 {
    // println!("tmp");
    let mut quotes = quote! {
        // println!("{__halogen_einsum_iter_index}");
        //
        __halogen_einsum_output_data[0] += #einsum_expr; // !!!!!!!!!!!!!!
        // __halogen_einsum_iter_index += __halogen_einsum_iter_stride;
    };
    for (i, idx) in indices.input.iter().enumerate().rev() {
        let idxval = format_ident!("__halogen_einsum_iter_in{i}");
        let idxlen = get_rep_size(idx, idx_rep).expect(
            "gen_run_einsum: something went wrong I will write err msg later (input idx_rep)",
        );
        quotes = quote! {
            for #idxval in (0..#idxlen) {
                #quotes
            }
        };
    }
    for (i, idx) in indices.output.iter().enumerate().rev() {
        let idxval = format_ident!("__halogen_einsum_iter_out{i}");
        let idxlen = get_rep_size(idx, idx_rep).expect(
            "gen_run_einsum: something went wrong I will write err msg later (output idx_rep)",
        );
        quotes = quote! {
            for #idxval in (0..#idxlen) {
                #quotes
                // __halogen_einsum_iter_index += __halogen_einsum_iter_stride; // update index
            }
            // __halogen_einsum_iter_stride *= #idxlen; // update stride
        };
        if i == indices.output.len() - 1 {
            quotes = quote! {
                // __halogen_einsum_iter_stride = 1; // init stride
                #quotes
            };
        }
    }
    // println!("tmp");
    quotes = quote! {
        // let mut __halogen_einsum_iter_index: usize = 0; // init index
        // let mut __halogen_einsum_iter_stride: usize = 1; // init stride
        #quotes
        let __halogen_einsum_output_tensor = crate::core::tensor::Tensor::from_vec(
            &__halogen_einsum_output_shape,
            __halogen_einsum_output_data
        );
    };
    quotes
}
