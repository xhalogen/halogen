use super::*;
use crate::core::tensor::Tensor;
use std::ops::*;

macro_rules! impl_zipwith {
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

impl_zipwith!(add, Add);
impl_zipwith!(sub, Sub);
impl_zipwith!(mul, Mul);
impl_zipwith!(div, Div);
impl_zipwith!(rem, Rem);
impl_zipwith!(bitand, BitAnd);
impl_zipwith!(bitor, BitOr);
impl_zipwith!(bitxor, BitXor);
impl_zipwith!(shl, Shl);
impl_zipwith!(shr, Shr);
