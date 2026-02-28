use std::any::TypeId;
use std::{cell::RefCell, rc::Weak};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

pub struct EdgeKind {
    pub dtype: TypeId,
    pub shape: Vec<usize>,
}

pub struct Edge {
    pub kind: EdgeKind,
    pub producer: Option<NodeId>,
}

pub enum NodeKind {}

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

pub struct GraphTensor {
    pub id: EdgeId,
    pub graph: Weak<RefCell<Graph>>,
}
