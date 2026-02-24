use crate::core::tensor::DenseTensor;
use crate::core::tensor::Tensor;
use halogen_macros::einsum;

#[test]
fn basic() {
    let a: DenseTensor<i32> = [[1, 2], [3, 4]].into();
    let b: DenseTensor<i32> = [[1, 2], [3, 4]].into();
    println!("hao");
    let c: DenseTensor<i32> = einsum!(a[i,j] * b[j,k] => [i, k]);
    for i in 0..2 {
        for j in 0..2 {
            print!("{}", c.at(&[i, j]));
        }
        println!();
    }
}
