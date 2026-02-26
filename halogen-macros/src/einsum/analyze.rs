use super::generator::get_size_map;
use super::indexer::{TensorIndices, get_indices};
use super::parser::EinsumInput;
use super::viewer::{TensorValue, get_tensorvalues};
use proc_macro2::Ident;
use std::collections::HashMap;

pub struct EinsumAnalysis {
    pub left_tensors: Vec<TensorValue>,
    pub indices: TensorIndices,
    pub size_map: HashMap<String, (Ident, usize)>,
}

pub fn analyze(input: &EinsumInput) -> syn::Result<EinsumAnalysis> {
    let left_tensors = get_tensorvalues(&input.left_exprs);
    let size_map = get_size_map(&left_tensors);
    let indices = get_indices(&left_tensors, &input.right_indices)?;
    Ok(EinsumAnalysis {
        left_tensors,
        indices,
        size_map,
    })
}
