use crate::einsum::{TensorIndices, TensorValue};
use proc_macro2::Ident as Ident2;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{Error, Ident, Result};

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
    indices: &TensorIndices,
    exprresult: &[Ident],
    idx_rep: &HashMap<String, (Ident2, usize)>,
) -> Result<TokenStream2> {
    let mut ret = Vec::new();
    ret.push(quote! {
        let mut __halogen_einsum_output_shape: ::std::vec::Vec<usize> = ::std::vec::Vec::new();
    });
    for idx in exprresult {
        let idxlen = get_rep_size(idx, idx_rep)
            .ok_or_else(|| Error::new_spanned(idx, "expression must contain every output index"))?;
        ret.push(quote! {
            __halogen_einsum_output_shape.push(#idxlen);
        });
    }
    ret.push(quote! {
        let __halogen_einsum_output_data_len: usize = __halogen_einsum_output_shape.iter().product();
        let mut __halogen_einsum_output_data = ::std::vec![::core::default::Default::default(); __halogen_einsum_output_data_len];
    });

    let mut idx_prc = HashMap::<String, usize>::new();
    for (i, idx) in indices.output.iter().enumerate() {
        idx_prc.insert(idx.to_string(), i);
    }
    ret.push(quote! {
        let mut __halogen_einsum_iter_stride_tmp = 1;
        let mut __halogen_einsum_iter_stride_sum = 0;
    });
    for i in (0..indices.output.len()).rev() {
        let strd = format_ident!("__halogen_einsum_iter_stride{i}");
        ret.push(quote! {
            let mut #strd = 0;
        });
    }
    for idx in exprresult.iter().rev() {
        let key = *idx_prc
            .get(&idx.to_string())
            .expect("def_outputvalues: something went wrong I will write err msg later");
        let strd = format_ident!("__halogen_einsum_iter_stride{key}");
        let idxlen =
            get_rep_size(idx, idx_rep).expect("expression must contain every output index");
        ret.push(quote! {
            #strd += __halogen_einsum_iter_stride_tmp;
            __halogen_einsum_iter_stride_tmp *= #idxlen;
            __halogen_einsum_iter_stride_sum += #strd;
        });
    }
    ret.push(quote! {
        __halogen_einsum_iter_stride_tmp = 0;
    });
    for i in 0..indices.output.len() {
        let strd = format_ident!("__halogen_einsum_iter_stride{i}");
        ret.push(quote! {
            __halogen_einsum_iter_stride_sum -= #strd;
            #strd -= __halogen_einsum_iter_stride_sum;
        });
    }
    ret.push(quote! {
        __halogen_einsum_iter_stride_tmp = 0;
    });
    for i in (0..indices.output.len()).rev() {
        let strd = format_ident!("__halogen_einsum_iter_stride{i}");
        if i == indices.output.len() - 1 {
            ret.push(quote! {
                __halogen_einsum_iter_stride_tmp = #strd;
            });
        } else {
            ret.push(quote! {
                #strd -= __halogen_einsum_iter_stride_tmp;
                __halogen_einsum_iter_stride_tmp += #strd;
            });
        }
    }
    Ok(quote! { #(#ret)* })
}

pub fn gen_run_einsum(
    indices: &TensorIndices,
    einsum_expr: TokenStream2,
    idx_rep: &HashMap<String, (Ident2, usize)>,
) -> Result<TokenStream2> {
    let mut quotes = quote! {
        __halogen_einsum_output_data[__halogen_einsum_iter_index] += #einsum_expr;
    };
    for (i, idx) in indices.input.iter().enumerate().rev() {
        let idxval = format_ident!("__halogen_einsum_iter_in{i}");
        let idxlen = get_rep_size(idx, idx_rep).ok_or_else(|| {
            Error::new_spanned(idx, "internal error: input index not found in size map")
        })?;
        quotes = quote! {
            for #idxval in (0..#idxlen) {
                #quotes
            }
        };
    }
    for (i, idx) in indices.output.iter().enumerate().rev() {
        let idxval = format_ident!("__halogen_einsum_iter_out{i}");
        let idxlen = get_rep_size(idx, idx_rep).ok_or_else(|| {
            Error::new_spanned(idx, "internal error: output index not found in size map")
        })?;
        let strideval = format_ident!("__halogen_einsum_iter_stride{i}");

        quotes = quote! {
            for #idxval in (0..#idxlen) {
                #quotes
                __halogen_einsum_iter_index += #strideval;
            }
        };
    }
    quotes = quote! {
        let mut __halogen_einsum_iter_index: usize = 0;
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
