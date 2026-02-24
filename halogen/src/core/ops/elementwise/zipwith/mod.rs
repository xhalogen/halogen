use crate::core::TensorError;
use crate::core::tensor::Tensor;

mod ext;
mod ops;
pub use ext::*;
pub use ops::*;

pub fn zipwith<A, B, C, F>(a: &A, b: &B, mut f: F) -> Result<C, TensorError>
where
    A: Tensor,
    B: Tensor,
    C: Tensor,
    A::Elem: Copy,
    B::Elem: Copy,
    F: FnMut(A::Elem, B::Elem) -> C::Elem,
{
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            left: a.shape().to_vec(),
            right: b.shape().to_vec(),
        });
    }
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    let mut ret = Vec::with_capacity(a_slice.len());
    for i in 0..a_slice.len() {
        ret.push(f(a_slice[i], b_slice[i]));
    }
    C::from_vec(a.shape(), ret)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::DenseTensor;

    #[test]
    fn zipwith_combines_elementwise() {
        let a = DenseTensor::<i32>::from_vec(&[2, 2], vec![1, 2, 3, 4]).unwrap();
        let b = DenseTensor::<i32>::from_vec(&[2, 2], vec![5, 6, 7, 8]).unwrap();
        let c: DenseTensor<i32> = zipwith(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.as_slice(), &[6, 8, 10, 12]);
        assert_eq!(c.shape(), a.shape());
    }

    #[test]
    fn zipwith_multiplies_elementwise() {
        let a = DenseTensor::<i32>::from_vec(&[3], vec![1, 2, 3]).unwrap();
        let b = DenseTensor::<i32>::from_vec(&[3], vec![4, 5, 6]).unwrap();
        let c: DenseTensor<i32> = zipwith(&a, &b, |x, y| x * y).unwrap();
        assert_eq!(c.as_slice(), &[4, 10, 18]);
    }

    #[test]
    fn zipwith_rejects_shape_mismatch() {
        let a = DenseTensor::<i32>::from_vec(&[2, 2], vec![1, 2, 3, 4]).unwrap();
        let b = DenseTensor::<i32>::from_vec(&[4], vec![1, 2, 3, 4]).unwrap();
        let result: Result<DenseTensor<i32>, _> = zipwith(&a, &b, |x, y| x + y);
        assert!(result.is_err());
    }
}
