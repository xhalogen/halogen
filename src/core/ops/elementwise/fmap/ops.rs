use super::*;
use crate::core::tensor::Tensor;
use std::ops::*;

macro_rules! def_fmap_ops {
    ($op:ident, $trait:tt) => {
        pub fn $op<A, C>(a: &A) -> Option<C>
        where
            A: Tensor,
            C: Tensor<Elem = <A::Elem as $trait>::Output>,
            A::Elem: Copy + $trait,
        {
            fmap::<A, C, _>(a, |x| x.$op())
        }
    };
}

def_fmap_ops!(neg, Neg);
def_fmap_ops!(not, Not);
