use super::*;
use crate::core::tensor::Tensor;
use std::ops::*;

macro_rules! impl_map {
    ($op:ident, $trait:tt) => {
        pub fn $op<A, C, const N: usize>(a: &A) -> Option<C>
        where
            A: Tensor<N>,
            C: Tensor<N, Elem = <A::Elem as $trait>::Output>,
            A::Elem: Copy + $trait,
        {
            map::<A, C, _, N>(a, |x| x.$op())
        }
    };
}

impl_map!(neg, Neg);
impl_map!(not, Not);
