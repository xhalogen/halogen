use super::*;
use crate::core::tensor::Tensor;
use std::ops::*;

macro_rules! def_fmap_ext {
    ($op:ident, $trait:tt) => {
        fn $op<C>(&self) -> Option<C>
        where
            Self: Sized,
            C: Tensor<Elem = <Self::Elem as $trait>::Output>,
            Self::Elem: Copy + $trait,
        {
            $op(self)
        }
    };
}

impl<T> TensorFmapExt for T where T: Tensor {}

pub trait TensorFmapExt: Tensor {
    fn fmap<C, F>(&self, f: F) -> Option<C>
    where
        Self: Sized,
        C: Tensor,
        Self::Elem: Copy,
        F: FnMut(Self::Elem) -> C::Elem,
    {
        fmap::<Self, C, F>(self, f)
    }

    def_fmap_ext!(neg, Neg);
    def_fmap_ext!(not, Not);
}
