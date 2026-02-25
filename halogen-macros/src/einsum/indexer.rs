use crate::einsum::TensorValue;
use std::collections::HashMap;
use syn::{Error, Ident, Result};

pub struct TensorIndices {
    pub inner: Vec<Ident>,
    pub outer: Vec<Ident>,
}

fn get_unique_indices(indices: &[Ident]) -> (Vec<Ident>, HashMap<Ident, i32>) {
    let mut count = HashMap::<Ident, i32>::new();
    let mut ret = Vec::new();
    for idx in indices {
        let tmp = count.entry(idx.clone()).or_insert(0);
        if *tmp == 0 {
            ret.push(idx.clone());
        }
        *tmp += 1;
    }
    (ret, count)
}

pub fn get_indices(tensors: &[TensorValue], right_indices: &[Ident]) -> Result<TensorIndices> {
    let (expr_idxs, expr_count) = {
        let tmp: Vec<Ident> = tensors
            .iter()
            .flat_map(|x| x.right_indices.iter())
            .cloned()
            .collect();
        get_unique_indices(&tmp)
    };
    let (res_idxs, res_count) = get_unique_indices(right_indices);
    for idx in &res_idxs {
        if !expr_count.contains_key(idx) {
            return Err(Error::new_spanned(
                idx,
                "expression must contain every output index",
            ));
        }
    }
    let output = res_idxs;
    let mut input = Vec::new();
    for idx in &expr_idxs {
        if !res_count.contains_key(idx) {
            input.push(idx.clone());
        }
    }
    Ok(TensorIndices {
        inner: input,
        outer: output,
    })
}
