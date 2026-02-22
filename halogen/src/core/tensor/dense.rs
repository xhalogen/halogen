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
