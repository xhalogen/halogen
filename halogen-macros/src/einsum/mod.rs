use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

mod parser;
use parser::*;
mod folder;
use folder::*;
mod viewer;
use viewer::*;
mod indexer;
use indexer::*;
mod generator;
use generator::*;
mod defv;

pub(crate) fn einsum(tokens: TokenStream) -> TokenStream {
    // 파싱
    let EinsumInput {
        left_exprs,
        right_indices,
    } = parse_macro_input!(tokens as EinsumInput);
    // 입력값들
    let left_tensors = get_tensorvalues(&left_exprs);
    // 인덱스 크기에 관한 사상
    let size_map = get_size_map(&left_tensors);
    // 인덱스들
    let indices = match get_indices(&left_tensors, &right_indices) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };
    // 코드 생성부
    // 합규약의 입력 텐서들을 정의
    let inputvalues_code = def_inputvalues(&left_tensors);
    // 같은 이름의 인덱스가 같은 크기를 가지는지 검증
    let size_checking_code = match gen_size_checks(&left_tensors, &size_map) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };
    // 합규약의 출력 텐서 관련 변수들을 정의
    let outputvalues_code = match def_outputvalues(&indices, &size_map) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };
    // 합 규약의 계산 루프
    let einsum_output_code = {
        let einsum_expr = match get_einsum_expr(&indices, &left_exprs) {
            Ok(v) => v,
            Err(e) => return e.to_compile_error().into(),
        };
        let einsum_loop = match gen_einsum_loop(&indices, einsum_expr, &size_map) {
            Ok(v) => v,
            Err(e) => return e.to_compile_error().into(),
        };
        gen_einsum_output(einsum_loop)
    };
    let ret = quote! {
        {
            #inputvalues_code
            #outputvalues_code
            #size_checking_code
            #einsum_output_code
        }
    };
    ret.into()
}
