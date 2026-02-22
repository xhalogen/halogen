use super::Tensor;

pub struct DenseTensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> DenseTensor<T> {
    fn offset_of(&self, idx: &[usize]) -> Option<usize> {
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

impl<T> Tensor for DenseTensor<T> {
    type Elem = T;

    fn from_vec(shape: &[usize], data: Vec<Self::Elem>) -> Option<Self> {
        let shape_size: usize = shape.iter().product();
        if shape_size != data.len() {
            return None;
        }
        Some(Self {
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

    fn get(&self, idx: &[usize]) -> Option<&Self::Elem> {
        let idx = self.offset_of(idx)?;
        self.data.get(idx)
    }
}
