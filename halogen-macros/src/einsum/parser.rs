use syn::{
    Expr, Ident, Result, Token, bracketed,
    parse::{Parse, ParseStream},
};

pub struct EinsumInput {
    pub expr: Expr,
    pub output: Vec<Ident>,
}

impl Parse for EinsumInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let expr = input.parse()?;
        input.parse::<Token![=>]>()?;
        let index;
        bracketed!(index in input);
        let output = index
            .parse_terminated(Ident::parse, Token![,])?
            .into_iter()
            .collect();
        Ok(EinsumInput { expr, output })
    }
}
