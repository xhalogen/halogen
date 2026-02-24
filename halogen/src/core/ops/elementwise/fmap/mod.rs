use crate::core::TensorError;
use crate::core::tensor::Tensor;

mod ext;
mod ops;
pub use ext::*;
use ops::*;

pub fn fmap<A, C, F>(a: &A, mut f: F) -> Result<C, TensorError>
where
    A: Tensor,
    C: Tensor,
    A::Elem: Copy,
    F: FnMut(A::Elem) -> C::Elem,
{
    let a_slice = a.as_slice();
    let mut ret = Vec::with_capacity(a_slice.len());
    for &x in a_slice {
        ret.push(f(x));
    }
    C::from_vec(a.shape(), ret)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::DenseTensor;

    #[test]
    fn fmap_applies_function_elementwise() {
        let a = DenseTensor::<i32>::from_vec(&[2, 2], vec![1, 2, 3, 4]).unwrap();
        let b: DenseTensor<i32> = fmap(&a, |x| x * 2).unwrap();
        assert_eq!(b.as_slice(), &[2, 4, 6, 8]);
        assert_eq!(b.shape(), a.shape());
    }

    #[test]
    fn fmap_preserves_shape() {
        let a = DenseTensor::<i32>::from_vec(&[3, 4], (0..12).collect()).unwrap();
        let b: DenseTensor<i32> = fmap(&a, |x| x + 1).unwrap();
        assert_eq!(b.shape(), &[3, 4]);
        assert_eq!(b.as_slice(), &(1..13).collect::<Vec<_>>());
    }

    #[test]
    fn neg_negates_all_elements() {
        let a = DenseTensor::<i32>::from_vec(&[3], vec![1, -2, 3]).unwrap();
        let b: DenseTensor<i32> = fmap(&a, |x| -x).unwrap();
        assert_eq!(b.as_slice(), &[-1, 2, -3]);
    }
}
