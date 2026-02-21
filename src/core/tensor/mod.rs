pub trait Tensor<const N: usize> {
    type Elem;
    fn rank(&self) -> usize {
        N
    }
    fn as_slice(&self) -> &[Self::Elem];
    fn from_vec(shape: [usize; N], data: Vec<Self::Elem>) -> Option<Self>
    where
        Self: Sized;
    fn shape(&self) -> &[usize; N];
    fn get(&self, idx: [usize; N]) -> Option<&Self::Elem>;
    fn at(&self, idx: [usize; N]) -> &Self::Elem {
        self.get(idx).expect("Index out of bounds")
    }
}

pub mod dense;
pub use dense::*;
