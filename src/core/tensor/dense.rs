use super::Tensor;

pub struct DenseTensor<T, const N: usize> {
    data: Vec<T>,
    shape: [usize; N],
}

impl<T, const N: usize> DenseTensor<T, N> {
    fn offset_of(&self, idx: [usize; N]) -> Option<usize> {
        let mut offset = 0usize;
        for (i, &idx_size) in idx.iter().enumerate() {
            let shape_size = self.shape[i];
            if shape_size <= idx_size {
                return None;
            }
            offset = offset * shape_size + idx_size;
        }
        Some(offset)
    }
}

impl<T, const N: usize> Tensor<N> for DenseTensor<T, N> {
    type Elem = T;

    fn from_vec(shape: [usize; N], data: Vec<Self::Elem>) -> Option<Self> {
        let shape_len: usize = shape.iter().product();
        if shape_len != data.len() {
            return None;
        }
        Some(Self { data, shape })
    }

    fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn get(&self, idx: [usize; N]) -> Option<&Self::Elem> {
        let idx = self.offset_of(idx)?;
        self.data.get(idx)
    }
}
