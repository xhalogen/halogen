use crate::core::tensor::Tensor;

mod ext;
mod ops;
pub use ext::*;
pub use ops::*;

pub fn map<A, C, F, const N: usize>(a: &A, mut f: F) -> Option<C>
where
    A: Tensor<N>,
    C: Tensor<N>,
    A::Elem: Copy,
    F: FnMut(A::Elem) -> C::Elem,
{
    let a_slice = a.as_slice();
    let mut ret = Vec::with_capacity(a_slice.len());
    for &x in a_slice {
        ret.push(f(x));
    }
    C::from_vec(*a.shape(), ret)
}
