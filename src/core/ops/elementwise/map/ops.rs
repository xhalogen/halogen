use super::*;
use crate::core::tensor::Tensor;
use std::ops::*;

macro_rules! def_map_ops {
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

def_map_ops!(neg, Neg);
def_map_ops!(not, Not);
