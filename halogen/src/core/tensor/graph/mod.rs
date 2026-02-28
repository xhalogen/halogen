use std::{cell::RefCell, rc::Rc};

pub enum DType {
    I32,
    I64,
    U32,
    U64,
    F32,
    F64,
}

pub enum ConstData {
    I32(Vec<i32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

pub enum OpKind {}

pub struct TensorNodeAttr {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

pub enum GraphNode {
    Input {
        output: usize,
    },
    Const {
        const_id: usize,
        output: usize,
    },
    Operator {
        op: OpKind,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
    },
}

pub struct Graph {
    pub nodes: Vec<GraphNode>,
    pub const_datas: Vec<ConstData>,
    pub attrs: Vec<TensorNodeAttr>,
    pub producers: Vec<(usize, usize)>,
    pub outputs: Vec<usize>,
}

pub struct GraphTensor {
    pub id: usize,
    pub graph: Rc<RefCell<Graph>>,
}
