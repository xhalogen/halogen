use super::*;
use syn::{Expr, ExprIndex, visit, visit::Visit};

pub fn get_tensorvalues(expr: &Expr) -> Vec<TensorValue> {
    let mut visitor = TensorValueVisitor { items: Vec::new() };
    visitor.visit_expr(expr);
    visitor.items
}

struct TensorValueVisitor {
    items: Vec<TensorValue>,
}

impl<'ast> Visit<'ast> for TensorValueVisitor {
    fn visit_expr_index(&mut self, expridx: &'ast ExprIndex) {
        if let Some(x) = is_tensorvalue(expridx) {
            self.items.push(x);
        }
        visit::visit_expr_index(self, expridx);
    }
}
