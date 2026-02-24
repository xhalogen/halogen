use crate::core::TensorError;

use super::Tensor;

pub struct DenseTensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> DenseTensor<T> {
    fn offset_of(&self, idx: &[usize]) -> Result<usize, TensorError> {
        if idx.len() != self.shape.len() {
            return Err(TensorError::IndexOutOfBounds {
                idx: idx.to_vec(),
                shape: self.shape.clone(),
            });
        }

        let mut offset = 0usize;
        for (i, &idx_size) in idx.iter().enumerate() {
            let shape_size = self.shape[i];
            if shape_size <= idx_size {
                return Err(TensorError::IndexOutOfBounds {
                    idx: idx.to_vec(),
                    shape: self.shape.clone(),
                });
            }
            offset = offset * shape_size + idx_size;
        }
        Ok(offset)
    }
}

impl<T> Tensor for DenseTensor<T> {
    type Elem = T;

    fn from_vec(shape: &[usize], data: Vec<Self::Elem>) -> Result<Self, TensorError> {
        let shape_size: usize = shape.iter().product();
        if shape_size != data.len() {
            return Err(TensorError::DataLengthMismatch {
                shape: shape.to_vec(),
                expected: shape_size,
                provided: data.len(),
            });
        }
        Ok(Self {
            shape: shape.to_vec(),
            data,
        })
    }

    fn rank(&self) -> usize {
        self.shape.len()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn get(&self, idx: &[usize]) -> Result<&Self::Elem, TensorError> {
        let offset = self.offset_of(idx)?;
        self.data.get(offset).ok_or_else(|| {
            let expected = self.shape.iter().product();
            TensorError::DataLengthMismatch {
                shape: self.shape.clone(),
                expected,
                provided: self.data.len(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_creates_correct_shape_and_data() {
        let t = DenseTensor::<i32>::from_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.as_slice(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(t.rank(), 2);
    }

    #[test]
    fn from_vec_rejects_length_mismatch() {
        assert!(DenseTensor::<i32>::from_vec(&[2, 3], vec![1, 2, 3]).is_err());
    }

    #[test]
    fn get_returns_correct_elements() {
        let t = DenseTensor::<i32>::from_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(t.get(&[0, 0]).unwrap(), &1);
        assert_eq!(t.get(&[0, 2]).unwrap(), &3);
        assert_eq!(t.get(&[1, 0]).unwrap(), &4);
        assert_eq!(t.get(&[1, 2]).unwrap(), &6);
    }

    #[test]
    fn get_rejects_out_of_bounds() {
        let t = DenseTensor::<i32>::from_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert!(t.get(&[2, 0]).is_err());
        assert!(t.get(&[0, 3]).is_err());
    }

    #[test]
    fn get_rejects_wrong_rank() {
        let t = DenseTensor::<i32>::from_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert!(t.get(&[0]).is_err());
        assert!(t.get(&[0, 0, 0]).is_err());
    }
}
