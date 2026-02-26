#[cfg(test)]
mod tests {
    use crate::core::tensor::{DenseTensor, Tensor};

    #[test]
    fn matmul_2x2() {
        let a: DenseTensor<i32> = [[1, 2], [3, 4]].into();
        let b: DenseTensor<i32> = [[1, 2], [3, 4]].into();
        let c = crate::einsum!(a[i, j] * b[j, k] => [i, k]);
        assert_eq!(c.shape(), &[2, 2]);
        // [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]] = [[7, 10], [15, 22]]
        assert_eq!(c.as_slice(), &[7, 10, 15, 22]);
    }

    #[test]
    fn matvec() {
        let a: DenseTensor<i32> = [[1, 2], [3, 4]].into();
        let v: DenseTensor<i32> = [1, 2].into();
        let w = crate::einsum!(a[i, j] * v[j] => [i]);
        assert_eq!(w.shape(), &[2]);
        // [1*1+2*2, 3*1+4*2] = [5, 11]
        assert_eq!(w.as_slice(), &[5, 11]);
    }

    #[test]
    fn dot_product() {
        let a: DenseTensor<i32> = [1, 2, 3].into();
        let b: DenseTensor<i32> = [4, 5, 6].into();
        let c = crate::einsum!(a[i] * b[i] => []);
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(c.as_slice(), &[32]);
    }

    #[test]
    fn outer_product() {
        let a: DenseTensor<i32> = [1, 2].into();
        let b: DenseTensor<i32> = [3, 4, 5].into();
        let c = crate::einsum!(a[i] * b[j] => [i, j]);
        assert_eq!(c.shape(), &[2, 3]);
        // [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]] = [[3, 4, 5], [6, 8, 10]]
        assert_eq!(c.as_slice(), &[3, 4, 5, 6, 8, 10]);
    }

    #[test]
    fn tr() {
        let a: DenseTensor<i32> = [[1, 2], [3, 4]].into();
        let c = crate::einsum!(a[i, i] => [i]);
        assert_eq!(c.shape(), &[2]);
        // [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]] = [[3, 4, 5], [6, 8, 10]]
        assert_eq!(c.as_slice(), &[1, 4]);
    }

    #[test]
    fn edge_case_0() {
        let a: DenseTensor<i32> = [[1, 2], [3, 4]].into();
        let b: DenseTensor<i32> = [[3, 4], [5, 6]].into();
        let c = crate::einsum!(a[i, i] + 3 + b[i, j] * 3 => [i]);
        assert_eq!(c.shape(), &[2]);
        // [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]] = [[3, 4, 5], [6, 8, 10]]
        assert_eq!(c.as_slice(), &[29, 47]);
    }

    // TODO:    파서 고치기
    // LINE:    crate::einsum!((a[i, i] + 3) + b[i, j] * 3 => [i]);
    // ERROR:   unexpected token, expected `]`
    // halogen-macros/src/einsum/parser.rs의 preprocess_einsum 문제로 추정
    // #[test]
    // fn edge_case_1() {
    //     let a: DenseTensor<i32> = [[1, 2], [3, 4]].into();
    //     let b: DenseTensor<i32> = [[3, 4], [5, 6]].into();
    //     let c = crate::einsum!((a[i, i] + 3) + b[i, j] * 3 => [i]);
    //     assert_eq!(c.shape(), &[2]);
    //     // [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]] = [[3, 4, 5], [6, 8, 10]]
    //     assert_eq!(c.as_slice(), &[29, 47]);
    // }
}
