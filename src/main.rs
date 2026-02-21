use halogen::core::tensor::*;

fn main() {
    let a = DenseTensor::from_vec([3], vec![0.0, 2.0, 3.0]).unwrap();
    let b = DenseTensor::from_vec([3], vec![0.0, 1.0, 7.0]).unwrap();
    let c: DenseTensor<_, 1> = a / b;
    for i in c.as_slice() {
        println!("{}", i);
    }
}
