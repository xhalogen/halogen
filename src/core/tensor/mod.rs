mod dense;
mod into;
mod scalar;

pub trait Tensor {
    type Elem;
    fn rank(&self) -> usize;
    fn as_slice(&self) -> &[Self::Elem];
    fn from_vec(shape: &[usize], data: Vec<Self::Elem>) -> Option<Self>
    where
        Self: Sized;
    fn shape(&self) -> &[usize];
    fn get(&self, idx: &[usize]) -> Option<&Self::Elem>;
    fn at(&self, idx: &[usize]) -> &Self::Elem {
        self.get(idx).expect("Index out of bounds")
    }
}

pub use dense::*;
pub use scalar::*;
