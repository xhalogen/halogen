use proc_macro::TokenStream;

mod analyze;
mod codegen;
mod defv;
mod folder;
mod generator;
mod indexer;
mod parser;
mod viewer;

use analyze::analyze;
use codegen::codegen;
use indexer::*;
use parser::EinsumInput;
use viewer::*;

fn einsum_inner(tokens: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let input = syn::parse2::<EinsumInput>(tokens)?;
    let analysis = analyze(&input)?;
    codegen(&analysis, &input.left_exprs)
}

pub(crate) fn einsum(tokens: TokenStream) -> TokenStream {
    match einsum_inner(tokens.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}
