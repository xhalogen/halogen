use proc_macro::TokenStream;

mod einsum;

#[proc_macro]
pub fn einsum(input: TokenStream) -> TokenStream {
    einsum::einsum(input)
}
