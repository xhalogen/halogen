mod dense;
mod into;
mod scalar;

use crate::core::TensorError;

pub trait Tensor {
    type Elem;
    fn rank(&self) -> usize;
    fn as_slice(&self) -> &[Self::Elem];
    fn from_vec(shape: &[usize], data: Vec<Self::Elem>) -> Result<Self, TensorError>
    where
        Self: Sized;
    fn shape(&self) -> &[usize];
    fn get(&self, idx: &[usize]) -> Result<&Self::Elem, TensorError>;
    fn at(&self, idx: &[usize]) -> &Self::Elem {
        self.get(idx).unwrap_or_else(|err| panic!("{err}"))
    }
}

pub use dense::*;
pub use scalar::*;
