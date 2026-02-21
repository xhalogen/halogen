use halogen::core::tensor::*;
use halogen::densetensor;

fn main() {
    let a = densetensor!([1, 2, 3, 4]);
    let b = densetensor!([1, 4, 5, 6]);
    let c = a + b;
    for i in c.as_slice() {
        println!("{}", i);
    }
}
