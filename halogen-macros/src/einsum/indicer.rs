use super::TensorValue;
use std::collections::HashSet;
use syn::{Error, Ident, Result};

pub struct TensorIndices {
    pub inner: Vec<Ident>,
    pub outer: Vec<Ident>,
}

fn get_unique_indices(
    indices: &[Ident],
    force_unique: bool,
) -> Result<(Vec<Ident>, HashSet<Ident>)> {
    let mut count = HashSet::<Ident>::new();
    let mut ret = Vec::new();
    for idx in indices {
        match count.contains(idx) {
            false => {
                count.insert(idx.clone());
                ret.push(idx.clone());
            }
            true if force_unique => {
                return Err(Error::new_spanned(
                    idx,
                    "every output indices should be unique",
                ));
            }
            _ => (),
        };
    }
    Ok((ret, count))
}

pub fn get_indices(tensors: &[TensorValue], right_indices: &[Ident]) -> Result<TensorIndices> {
    let (expr_idxs, expr_count) = {
        let tmp: Vec<Ident> = tensors
            .iter()
            .flat_map(|x| x.right_indices.iter())
            .cloned()
            .collect();
        get_unique_indices(&tmp, false)?
    };
    let (res_idxs, res_count) = get_unique_indices(right_indices, true)?;
    for idx in &res_idxs {
        if !expr_count.contains(idx) {
            return Err(Error::new_spanned(
                idx,
                "expression must contain every output indices",
            ));
        }
    }
    let output = res_idxs;
    let mut input = Vec::new();
    for idx in &expr_idxs {
        if !res_count.contains(idx) {
            input.push(idx.clone());
        }
    }
    Ok(TensorIndices {
        inner: input,
        outer: output,
    })
}
