use crate::core::tensor::Tensor;

mod ext;
mod ops;
pub use ext::*;
pub use ops::*;

pub fn zipwith<A, B, C, F, const N: usize>(a: &A, b: &B, mut f: F) -> Option<C>
where
    A: Tensor<N>,
    B: Tensor<N>,
    C: Tensor<N>,
    A::Elem: Copy,
    B::Elem: Copy,
    F: FnMut(A::Elem, B::Elem) -> C::Elem,
{
    if a.shape() != b.shape() {
        return None;
    }
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    let mut ret = Vec::with_capacity(a_slice.len());
    for i in 0..a_slice.len() {
        ret.push(f(a_slice[i], b_slice[i]));
    }
    C::from_vec(*a.shape(), ret)
}
