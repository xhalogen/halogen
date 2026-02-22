use super::{DenseTensor, Scalar};

pub trait IntoTensor {
    type Elem;
    fn shape(&self) -> Vec<usize>;
    fn flatten_into(self, buf: &mut Vec<Self::Elem>);
}

impl<T: Scalar> IntoTensor for T {
    type Elem = T;
    fn shape(&self) -> Vec<usize> {
        vec![]
    }
    fn flatten_into(self, buf: &mut Vec<T>) {
        buf.push(self);
    }
}

impl<A: IntoTensor, const N: usize> IntoTensor for [A; N] {
    type Elem = A::Elem;
    fn shape(&self) -> Vec<usize> {
        let mut s = vec![N];
        s.extend(self[0].shape());
        s
    }
    fn flatten_into(self, buf: &mut Vec<Self::Elem>) {
        for item in self {
            item.flatten_into(buf);
        }
    }
}

impl<A: IntoTensor> From<A> for DenseTensor<A::Elem> {
    fn from(arr: A) -> Self {
        let shape = arr.shape();
        let mut data = Vec::new();
        arr.flatten_into(&mut data);
        Self { shape, data }
    }
}
