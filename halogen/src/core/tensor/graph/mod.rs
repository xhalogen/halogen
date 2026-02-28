// use crate::core::tensor::{Tensor, TensorError};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Copy)]
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

impl ConstData {
    pub fn len(&self) -> usize {
        match self {
            ConstData::I32(v) => v.len(),
            ConstData::I64(v) => v.len(),
            ConstData::U32(v) => v.len(),
            ConstData::U64(v) => v.len(),
            ConstData::F32(v) => v.len(),
            ConstData::F64(v) => v.len(),
        }
    }
}

pub enum OpKind {
    GET,
}

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

#[derive(Clone)]
pub struct GraphTensor {
    pub id: usize,
    pub graph: Rc<RefCell<Graph>>,
}

// impl GraphTensor {
//     fn push_const(&self, data: ConstData) -> usize {
//         let mut graph = self.graph.borrow_mut();
//         let id = graph.attrs.len();
//         let node = GraphNode::Const {
//             const_id: (graph.const_datas.len()),
//             output: id,
//         };
//         graph.nodes.push(node);
//         graph.attrs.push(TensorNodeAttr {
//             dtype: DType::U64,
//             shape: vec![data.len()],
//         });
//         graph.const_datas.push(data);
//         id
//     }

//     fn push_function(&self, op: OpKind, inputs: Vec<usize>) -> usize {
//         let mut graph = self.graph.borrow_mut();
//         let id = graph.attrs.len();
//         let dtype = graph.attrs[self.id].dtype;
//         let node = GraphNode::Operator {
//             op,
//             inputs,
//             outputs: vec![id],
//         };
//         graph.nodes.push(node);
//         graph.attrs.push(TensorNodeAttr {
//             dtype,
//             shape: vec![],
//         });
//         id
//     }
// }

// impl Tensor for GraphTensor {
//     type Elem = GraphTensor;
//     type ElemRet<'a>
//         = Self::Elem
//     where
//         Self: 'a;

//     fn get<'a>(&'a self, idx: &[usize]) -> Result<Self::ElemRet<'a>, TensorError> {
//         let idx_data = ConstData::U64(idx.iter().map(|x| *x as u64).collect());
//         let idx_id = self.push_const(idx_data);
//         let elem_id = self.push_function(OpKind::GET, vec![self.id, idx_id]);
//         Ok(GraphTensor {
//             id: elem_id,
//             graph: Rc::clone(&self.graph),
//         })
//     }

//     fn as_slice(&self) -> &[Self::Elem] {}

//     fn from_vec(shape: &[usize], data: Vec<Self::Elem>) -> Result<Self, TensorError>
//     where
//         Self: Sized,
//     {
//         Ok(Self { id: (), graph: () })
//     }
//     fn rank(&self) -> usize {
//         0
//     }
//     fn reshape(&self, shape: &[usize]) -> Result<Self, TensorError>
//     where
//         Self: Sized,
//     {
//         Ok((self.clone()))
//     }
//     fn shape(&self) -> &[usize] {}
// }
