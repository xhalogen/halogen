use crate::core::tensor::Tensor;

pub trait TensorMapExt<const N: usize>: Tensor<N> {
    fn map<C, F>(&self, f: F) -> Option<C>
    where
        Self: Sized,
        C: Tensor<N>,
        Self::Elem: Copy,
        F: FnMut(Self::Elem) -> C::Elem,
    {
        map::<Self, C, F, N>(self, f)
    }
}

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

impl<T, const N: usize> TensorMapExt<N> for T where T: Tensor<N> {}
