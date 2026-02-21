use crate::core::tensor::Tensor;

pub trait TensorZipwithExt<const N: usize>: Tensor<N> {
    fn zipwith<B, C, F>(&self, b: &B, f: F) -> Option<C>
    where
        Self: Sized,
        B: Tensor<N>,
        C: Tensor<N>,
        Self::Elem: Copy,
        B::Elem: Copy,
        F: FnMut(Self::Elem, B::Elem) -> C::Elem,
    {
        zipwith::<Self, B, C, F, N>(self, b, f)
    }
}

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

impl<T, const N: usize> TensorZipwithExt<N> for T where T: Tensor<N> {}
