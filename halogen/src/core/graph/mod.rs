// use std::any::TypeId;
// use std::cell::RefCell;
// use std::ops::*;
// use std::rc::Rc;

// struct RxProcedure {
//     pub inputs: Vec<Rc<RxValue>>,
//     outputs: Vec<Rc<RxValue>>,
//     constants: Vec<Rc<RxValue>>,
//     instructions: Vec<RxInstruction>,
// }

// struct RxInstruction {
//     operator: RxOperator,
//     operands: Vec<String>,
//     inputs: Vec<Rc<RxValue>>,
//     outputs: Vec<Rc<RxValue>>,
// }

// enum RxOperator {
//     Add,
//     Matmul,
// }

// enum RxValue {
//     Constant(RxConstant),
//     Variable(RxVariable),
//     Dynamic(RxDynamic),
//     Tuple(RxTuple),
//     Etc(RxEtc),
// }

// struct RxConstant {
//     id: i64,
//     dtype: TypeId,
//     shape: Vec<usize>,
//     data: RxData,
// }

// struct RxVariable {
//     id: i64,
//     dtype: TypeId,
//     shape: Vec<usize>,
// }

// struct RxDynamic {
//     id: i64,
//     dtype: TypeId,
// }

// struct RxTuple {
//     id: i64,
//     elems: Vec<Rc<RxValue>>,
// }

// struct RxEtc {
//     id: i64,
//     info: Vec<String>,
// }

// struct RxData {}

// struct RxTensor {
//     owner: Rc<RefCell<RxProcedure>>,
//     data: Rc<RxValue>,
// }
