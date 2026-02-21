use halogen::core::tensor::*;
use halogen::densetensor;

fn main() {
    let a = densetensor!(2; [[1.0], [0.1]]);
    let b = densetensor!(2; [[3.0], [3.0]]);
    let c: DenseTensor<_, _> = a + b;
    for i in c.as_slice() {
        println!("{}", i);
    }
}
