use halogen::core::ops::elementwise::*;
use halogen::core::tensor::*;

fn main() {
    let a = DenseTensor::from_vec([3], vec![1, 2, 3]).unwrap();
    let b = DenseTensor::from_vec([3], vec![3, 1, 7]).unwrap();
    let c: DenseTensor<i32, 1> = a.add(&b).unwrap();
    for i in c.as_slice() {
        println!("{}", i);
    }
}
