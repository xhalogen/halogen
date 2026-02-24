use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

mod parser;
pub use parser::*;
mod folder;
pub use folder::*;
mod viewer;
pub use viewer::*;
mod indexer;
pub use indexer::*;
mod generator;
pub use generator::*;

pub fn einsum(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as EinsumInput);
    let tensors = get_tensorvalues(&input.expr);

    let (size_checks, idx_rep) = gen_size_checks(&tensors);

    let indices = match get_indices(&tensors, &input.output) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };

    let def_tensorvalues = def_tensorvalues(&tensors);
    let def_outputvalues = match def_outputvalues(&indices, &input.output, &idx_rep) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };

    // println!("tmp");

    let gen_run_einsum = {
        // println!("tmp");
        // let tmp = &input.expr;
        // println!("{}", quote! { #tmp }.to_string());
        let einsum_expr = match get_einsum_expr(&indices, &input.expr) {
            Ok(v) => v,
            Err(e) => return e.to_compile_error().into(),
        };
        match gen_run_einsum(&indices, einsum_expr, &idx_rep) {
            Ok(v) => v,
            Err(e) => return e.to_compile_error().into(),
        }
    };
    let ret = quote! {
        {
            #def_tensorvalues
            #size_checks
            #def_outputvalues
            #gen_run_einsum
            __halogen_einsum_output_tensor.unwrap()
        }
    };
    // println!("{}", ret.to_string());
    ret.into()
}
