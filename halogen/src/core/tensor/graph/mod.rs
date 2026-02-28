use crate::core::tensor::TensorError;
use std::any::TypeId;
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EdgeKind {
    pub dtype: TypeId,
    pub shape: Vec<usize>,
}

pub struct Edge {
    pub kind: EdgeKind,
    pub producer: Option<NodeId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Get,
    Reshape,
    Const { data: Vec<u8> },
}

pub struct Node {
    pub kind: NodeKind,
    pub inputs: Vec<EdgeId>,
    pub outputs: Vec<EdgeId>,
}

pub struct Graph {
    pub inputs: Vec<EdgeId>,
    pub outputs: Vec<EdgeId>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn new_edge(&mut self, edge: Edge) -> EdgeId {
        let id = EdgeId(self.edges.len());
        self.edges.push(edge);
        id
    }

    pub fn new_node(
        &mut self,
        kind: NodeKind,
        inputs: Vec<EdgeId>,
        outputs: Vec<EdgeKind>,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        let mut output_ids = Vec::<EdgeId>::new();
        for output in outputs {
            let edge_id = self.new_edge(Edge {
                kind: output,
                producer: Some(id),
            });
            output_ids.push(edge_id);
        }
        let node = Node {
            kind,
            inputs,
            outputs: output_ids,
        };
        self.nodes.push(node);
        id
    }
}

pub struct GraphTensor {
    pub id: EdgeId,
    pub graph: Weak<RefCell<Graph>>,
    pub dtype: TypeId,
    pub shape: Vec<usize>,
}

#[allow(dead_code)]
impl GraphTensor {
    pub fn graph(&self) -> Result<Rc<RefCell<Graph>>, TensorError> {
        self.graph.upgrade().ok_or(TensorError::GraphDropped)
    }

    fn rank(&self) -> usize {
        self.shape().len()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn get(&self, idx: &[usize]) -> Result<Self, TensorError> {
        let graph_rc = self.graph()?;
        let mut graph = graph_rc.borrow_mut();
        let idx_bytes: Vec<u8> = idx.iter().flat_map(|i| i.to_le_bytes()).collect();
        let idx_id = graph.new_node(
            NodeKind::Const { data: idx_bytes },
            vec![],
            vec![EdgeKind {
                dtype: TypeId::of::<usize>(),
                shape: vec![idx.len()],
            }],
        );
        let idx_edge = graph.nodes[idx_id.0].outputs[0];
        let get_id = graph.new_node(
            NodeKind::Get,
            vec![self.id, idx_edge],
            vec![EdgeKind {
                dtype: self.dtype,
                shape: self.shape.clone(),
            }],
        );
        let get_edge = graph.nodes[get_id.0].outputs[0];
        Ok(GraphTensor {
            id: get_edge,
            graph: Weak::clone(&self.graph),
            dtype: self.dtype,
            shape: self.shape.clone(),
        })
    }

    fn reshape(&self, shape: &[usize]) -> Result<Self, TensorError> {
        let graph_rc = self.graph()?;
        let mut graph = graph_rc.borrow_mut();

        let shape_bytes: Vec<u8> = shape.iter().flat_map(|i| i.to_le_bytes()).collect();
        let shape_id = graph.new_node(
            NodeKind::Const { data: shape_bytes },
            vec![],
            vec![EdgeKind {
                dtype: TypeId::of::<usize>(),
                shape: vec![shape.len()],
            }],
        );
        let shape_edge = graph.nodes[shape_id.0].outputs[0];
        let reshape_id = graph.new_node(
            NodeKind::Reshape,
            vec![self.id, shape_edge],
            vec![EdgeKind {
                dtype: self.dtype,
                shape: shape.to_vec(),
            }],
        );
        let reshape_edge = graph.nodes[reshape_id.0].outputs[0];
        Ok(GraphTensor {
            id: reshape_edge,
            graph: Weak::clone(&self.graph),
            dtype: self.dtype,
            shape: shape.to_vec(),
        })
    }
}
