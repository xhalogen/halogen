use proc_macro2::{TokenStream as TokenStream2, TokenTree as TokenTree2};
use quote::quote;
use std::iter::once;
use syn::{
    Expr, Ident, Result, Token, bracketed,
    parse::{Parse, ParseStream},
    parse2,
};

pub struct EinsumInput {
    pub expr: Expr,
    pub output: Vec<Ident>,
}

impl Parse for EinsumInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut tokens = TokenStream2::new();
        while !input.is_empty() && !input.peek(Token![=>]) {
            tokens.extend(once(
                input
                    .parse::<TokenTree2>()
                    .expect("einsum/parser: something went wrong while extending token"),
            ));
        }
        let expr = parse2(preprocess_einsum(tokens))
            .expect("einsum/parser: something went wrong while preprocessing expression");

        input.parse::<Token![=>]>()?;
        let index;
        bracketed!(index in input);
        let output = index
            .parse_terminated(Ident::parse, Token![,])
            .expect("einsum/parser: expected ,")
            .into_iter()
            .collect();

        Ok(EinsumInput { expr, output })
    }
}

fn preprocess_einsum(tokens: TokenStream2) -> TokenStream2 {
    let tokens: Vec<_> = tokens.into_iter().collect();
    let mut ret = TokenStream2::new();
    let mut now = 0;
    while now < tokens.len() {
        if let Some(TokenTree2::Ident(_)) = tokens.get(now) {
            if let Some(TokenTree2::Group(g)) = tokens.get(now + 1) {
                if g.delimiter() == proc_macro2::Delimiter::Bracket {
                    let indices = g.stream();
                    let tensor = tokens[now].clone();
                    ret.extend(quote! {#tensor[(#indices)]});
                    now += 2;
                    continue;
                }
            }
        }
        ret.extend(once(tokens[now].clone()));
        now += 1;
    }
    ret
}
