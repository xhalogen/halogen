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
    Get { idx: Vec<usize> },
    Reshape { shape: Vec<usize> },
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
}

#[allow(dead_code)]
impl GraphTensor {
    pub fn graph(&self) -> Result<Rc<RefCell<Graph>>, TensorError> {
        self.graph.upgrade().ok_or(TensorError::GraphDropped)
    }

    fn dtype(&self) -> TypeId {
        let graph = self.graph().unwrap();
        graph.borrow().edges[self.id.0].kind.dtype
    }

    fn rank(&self) -> usize {
        self.shape().len()
    }

    fn shape(&self) -> Vec<usize> {
        let graph = self.graph().unwrap();
        graph.borrow().edges[self.id.0].kind.shape.clone()
    }

    fn get(&self, idx: &[usize]) -> Result<Self, TensorError> {
        let graph_rc = self.graph()?;
        let mut graph = graph_rc.borrow_mut();
        let get_id = graph.new_node(
            NodeKind::Get { idx: idx.to_vec() },
            vec![self.id],
            vec![EdgeKind {
                dtype: self.dtype(),
                shape: vec![],
            }],
        );
        let get_edge = graph.nodes[get_id.0].outputs[0];
        Ok(GraphTensor {
            id: get_edge,
            graph: Weak::clone(&self.graph),
        })
    }

    fn reshape(&self, shape: &[usize]) -> Result<Self, TensorError> {
        let graph_rc = self.graph()?;
        let mut graph = graph_rc.borrow_mut();
        let reshape_id = graph.new_node(
            NodeKind::Reshape {
                shape: shape.to_vec(),
            },
            vec![self.id],
            vec![EdgeKind {
                dtype: self.dtype(),
                shape: shape.to_vec(),
            }],
        );
        let reshape_edge = graph.nodes[reshape_id.0].outputs[0];
        Ok(GraphTensor {
            id: reshape_edge,
            graph: Weak::clone(&self.graph),
        })
    }
}
