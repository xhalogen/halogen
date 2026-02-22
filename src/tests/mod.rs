use crate::core::tensor::*;

#[test]
fn basic() {
    let a: DenseTensor<i32> = [1, 2, 3, 4].into();
    assert_eq!(a.rank(), 1);

    let b: DenseTensor<i32> = [[1, 4, 5, 6], [7, 8, 9, 10]].into();
    assert_eq!(b.rank(), 2);
}
