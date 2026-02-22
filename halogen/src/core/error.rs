use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TensorError {
    #[error("shape mismatch: left={left:?}, right={right:?}")]
    ShapeMismatch { left: Vec<usize>, right: Vec<usize> },

    #[error("data length mismatch for shape {shape:?}: expected {expected}, provided {provided}")]
    DataLengthMismatch {
        shape: Vec<usize>,
        expected: usize,
        provided: usize,
    },

    #[error("index out of bounds: idx={idx:?}, shape={shape:?}")]
    IndexOutOfBounds { idx: Vec<usize>, shape: Vec<usize> },
}
