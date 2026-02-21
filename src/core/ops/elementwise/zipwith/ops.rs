use super::*;
use crate::core::tensor::Tensor;
use std::ops::*;

macro_rules! def_zipwith_ops {
    ($op:ident, $trait:tt) => {
        pub fn $op<A, B, C, const N: usize>(a: &A, b: &B) -> Option<C>
        where
            A: Tensor<N>,
            B: Tensor<N>,
            C: Tensor<N, Elem = <A::Elem as $trait<B::Elem>>::Output>,
            A::Elem: Copy + $trait<B::Elem>,
            B::Elem: Copy,
        {
            zipwith::<A, B, C, _, N>(a, b, |x, y| x.$op(y))
        }
    };
}

def_zipwith_ops!(add, Add);
def_zipwith_ops!(sub, Sub);
def_zipwith_ops!(mul, Mul);
def_zipwith_ops!(div, Div);
def_zipwith_ops!(rem, Rem);
def_zipwith_ops!(bitand, BitAnd);
def_zipwith_ops!(bitor, BitOr);
def_zipwith_ops!(bitxor, BitXor);
def_zipwith_ops!(shl, Shl);
def_zipwith_ops!(shr, Shr);
