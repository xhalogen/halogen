use halogen::core::ops::elementwise::*;
use halogen::core::tensor::*;
use halogen::densetensor;

fn main() {
    let a: DenseTensor<i32> = densetensor!([1, 2, 3, 4]);
    let b: DenseTensor<i32> = densetensor!([1, 4, 5, 6]);
    let c: DenseTensor<i32> = a
        .zipwith::<_, DenseTensor<i32>, _>(&b, |x, y| x - y)
        .expect("idk")
        .fmap(|x: i32| x + 1)
        .expect("idk");
    for i in c.as_slice() {
        println!("{}", i);
    }
}
