#[macro_export]
macro_rules! densetensor {
    (0; $x:expr) => {
        {
            let data = vec![$x];
            DenseTensor::from_vec([], data).unwrap()
        }
    };
    (1; [$($x:expr),* $(,)?]) => {
        {
            let data = vec![$($x),*];
            let d0 = data.len();
            DenseTensor::from_vec([d0], data).unwrap()
        }
    };
    (2; [$([$($x:expr),* $(,)?]),* $(,)?]) => {
        {
            let data = [$(vec![$($x),*]),*];
            let d0 = data.len();
            let d1 = data[0].len();
            let data: Vec<_> = data.into_iter().flatten().collect();
            DenseTensor::from_vec([d0, d1], data).unwrap()
        }
    };
    (3; [$([$([$($x:expr),* $(,)?]),* $(,)?]),* $(,)?]) => {
        {
            let data = [$([$(vec![$($x),*]),*]),*];
            let d0 = data.len();
            let d1 = data[0].len();
            let d2 = data[0][0].len();
            let data: Vec<_> = data.into_iter()
                .flatten()
                .flatten()
                .collect();
            DenseTensor::from_vec([d0, d1, d2], data).unwrap()
        }
    };
    (4; [$([$([$([$($x:expr),* $(,)?]),* $(,)?]),* $(,)?]),* $(,)?]) => {
        {
            let data = [$([$([$(vec![$($x),*]),*]),*]),*];
            let d0 = data.len();
            let d1 = data[0].len();
            let d2 = data[0][0].len();
            let d3 = data[0][0][0].len();
            let data: Vec<_> = data.into_iter()
                .flatten()
                .flatten()
                .flatten()
                .collect();
            DenseTensor::from_vec([d0, d1, d2, d3], data).unwrap()
        }
    };
    ([$($x:expr),* $(,)?]) => {
        {
            let data = vec![$($x),*];
            let d0 = data.len();
            DenseTensor::from_vec([d0], data).unwrap()
        }
    };
    ($x:expr) => {
        {
            let data = vec![$x];
            DenseTensor::from_vec([], data).unwrap()
        }
    };
}
