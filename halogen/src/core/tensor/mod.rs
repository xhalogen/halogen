mod dense;
mod graph;
mod into;
mod scalar;
use core::borrow::Borrow;

use crate::core::TensorError;

pub trait Tensor: Sized {
    type Elem;
    type ElemRet<'a>: Borrow<Self::Elem>
    where
        Self: 'a;
    fn rank(&self) -> usize;
    fn as_slice(&self) -> &[Self::Elem];
    fn from_vec(shape: &[usize], data: Vec<Self::Elem>) -> Result<Self, TensorError>
    where
        Self: Sized;
    fn shape(&self) -> &[usize];
    fn get<'a>(&'a self, idx: &[usize]) -> Result<Self::ElemRet<'a>, TensorError>;
    fn at<'a>(&'a self, idx: &[usize]) -> Self::ElemRet<'a> {
        self.get(idx).unwrap_or_else(|err| panic!("{err}"))
    }
    fn reshape(&self, shape: &[usize]) -> Result<Self, TensorError>
    where
        Self: Sized;
}

pub use dense::*;
pub use graph::*;
pub use scalar::*;
