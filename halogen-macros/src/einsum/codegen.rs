use super::analyze::EinsumAnalysis;
use super::folder::*;
use super::generator::*;
use quote::quote;
use syn::Expr;

pub(super) fn codegen(
    analysis: &EinsumAnalysis,
    left_exprs: &Expr,
) -> syn::Result<proc_macro2::TokenStream> {
    let EinsumAnalysis {
        left_tensors,
        indices,
        size_map,
    } = analysis;

    let inputvalues_code = def_inputvalues(left_tensors);
    let size_checking_code = gen_size_checks(left_tensors, size_map)?;
    let outputvalues_code = def_outputvalues(indices, size_map)?;

    let einsum_expr = get_einsum_expr(indices, left_exprs)?;
    let einsum_loop = gen_einsum_loop(indices, einsum_expr, size_map)?;
    let einsum_output_code = gen_einsum_output(einsum_loop);

    Ok(quote! {
        {
            #inputvalues_code
            #outputvalues_code
            #size_checking_code
            #einsum_output_code
        }
    })
}
