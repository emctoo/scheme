pub mod cont;
pub mod env;
pub mod error;
pub mod json;
pub mod list;
pub mod matches;
pub mod procedure;
pub mod special;
pub mod tests;
pub mod trampoline;
pub mod value;

pub use cont::*;
pub use env::*;
pub use error::*;
pub use list::*;
pub use procedure::*;
pub use special::*;
pub use trampoline::*;
pub use value::*;

use crate::reader::parser::*;

use std::cell::RefCell;
use std::rc::Rc;

pub fn new() -> Result<Interpreter, RuntimeError> { Interpreter::new() }

#[derive(Clone)]
pub struct Interpreter {
    root: Rc<RefCell<Env>>,
}

impl Interpreter {
    pub fn new() -> Result<Interpreter, RuntimeError> { Ok(Interpreter { root: Env::new_root()? }) }
    pub fn run(&self, nodes: &[Node]) -> Result<Value, RuntimeError> { cps(List::from_nodes(nodes), self.root.clone()) }
}
