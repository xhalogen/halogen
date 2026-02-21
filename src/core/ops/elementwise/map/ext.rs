use super::*;
use crate::core::tensor::Tensor;
use std::ops::*;

macro_rules! def_map_ext {
    ($op:ident, $trait:tt) => {
        fn $op<C>(&self) -> Option<C>
        where
            Self: Sized,
            C: Tensor<N, Elem = <Self::Elem as $trait>::Output>,
            Self::Elem: Copy + $trait,
        {
            $op(self)
        }
    };
}

impl<T, const N: usize> TensorMapExt<N> for T where T: Tensor<N> {}

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

    def_map_ext!(neg, Neg);
    def_map_ext!(not, Not);
}
