use super::*;
use crate::core::TensorError;
use crate::core::tensor::Tensor;
use std::ops::*;

macro_rules! def_zipwith_ext {
    ($op:ident, $trait:tt) => {
        fn $op<B, C>(&self, b: &B) -> Result<C, TensorError>
        where
            Self: Sized,
            B: Tensor,
            C: Tensor<Elem = <Self::Elem as $trait<B::Elem>>::Output>,
            Self::Elem: Copy + $trait<B::Elem>,
            B::Elem: Copy,
        {
            $op(self, b)
        }
    };
}

pub trait TensorZipwithExt: Tensor {
    fn zipwith<B, C, F>(&self, b: &B, f: F) -> Result<C, TensorError>
    where
        Self: Sized,
        B: Tensor,
        C: Tensor,
        Self::Elem: Copy,
        B::Elem: Copy,
        F: FnMut(Self::Elem, B::Elem) -> C::Elem,
    {
        zipwith::<Self, B, C, F>(self, b, f)
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
impl<T> TensorZipwithExt for T where T: Tensor {}
