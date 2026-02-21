use super::*;
use crate::core::tensor::Tensor;
use std::ops::*;

macro_rules! def_zipwith_ext {
    ($op:ident, $trait:tt) => {
        fn $op<B, C>(&self, b: &B) -> Option<C>
        where
            Self: Sized,
            B: Tensor<N>,
            C: Tensor<N, Elem = <Self::Elem as $trait<B::Elem>>::Output>,
            Self::Elem: Copy + $trait<B::Elem>,
            B::Elem: Copy,
        {
            $op(self, b)
        }
    };
}

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

    def_zipwith_ext!(add, Add);
    def_zipwith_ext!(sub, Sub);
    def_zipwith_ext!(mul, Mul);
    def_zipwith_ext!(div, Div);
    def_zipwith_ext!(rem, Rem);
    def_zipwith_ext!(bitand, BitAnd);
    def_zipwith_ext!(bitor, BitOr);
    def_zipwith_ext!(bitxor, BitXor);
    def_zipwith_ext!(shl, Shl);
    def_zipwith_ext!(shr, Shr);
}
impl<T, const N: usize> TensorZipwithExt<N> for T where T: Tensor<N> {}
