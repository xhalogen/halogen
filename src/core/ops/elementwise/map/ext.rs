use super::*;
use crate::core::tensor::Tensor;
use std::ops::*;

macro_rules! def_map_ext {
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

impl<T> TensorMapExt for T where T: Tensor {}

pub trait TensorMapExt: Tensor {
    fn map<C, F>(&self, f: F) -> Option<C>
    where
        Self: Sized,
        C: Tensor,
        Self::Elem: Copy,
        F: FnMut(Self::Elem) -> C::Elem,
    {
        map::<Self, C, F>(self, f)
    }

    def_map_ext!(neg, Neg);
    def_map_ext!(not, Not);
}
