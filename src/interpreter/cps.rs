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

use crate::match_list;

pub fn new() -> Result<Interpreter, RuntimeError> { Interpreter::new() }

#[derive(Clone)]
pub struct Interpreter {
    root: Rc<RefCell<Env>>,
}

impl Interpreter {
    pub fn new() -> Result<Interpreter, RuntimeError> {
        let env = Env::new_root()?;
        Ok(Interpreter { root: env })
    }

    pub fn run(&self, nodes: &[Node]) -> Result<Value, RuntimeError> { eval_cps(List::from_nodes(nodes), self.root.clone()) }
}

#[macro_export]
macro_rules! runtime_error {
    ($($arg:tt)*) => (
        return Err(RuntimeError { message: format!($($arg)*)})
    )
}
#[macro_export]
macro_rules! shift_or_error {
    ($list:expr, $($arg:tt)*) => (
        match $list.shift() {
            Some((car, cdr)) => Ok((car, cdr)),
            None => Err(RuntimeError { message: format!($($arg)*)})
        }?
    )
}

fn apply(val: Value, args: List, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Cont(c) => Ok(Trampoline::Apply(args.into_list(), *c)),
        Value::Procedure(Procedure::Native(f)) => Ok(Trampoline::Apply(primitive(f, args)?, *k)),
        Value::Procedure(Procedure::Scheme(arg_names, body, env)) => {
            if arg_names.len() != args.len() {
                runtime_error!("Must supply exactly {} arguments to function: {:?}", arg_names.len(), args);
            }

            // Create a new, child environment for the procedure and define the arguments as local variables
            let proc_env = Env::new_child(env);
            arg_names
                .into_iter()
                .zip(args)
                .try_for_each(|(name, value)| proc_env.borrow_mut().define(name, value))?;

            // Evaluate procedure body with new environment with procedure environment as parent
            eval(body, Env::new_child(proc_env), k)
        }
        _ => runtime_error!("Don't know how to apply: {:?}", val),
    }
}

pub fn eval(expr: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match_list!(expr, head: car, tail: cdr => Trampoline::Bounce(car, env.clone(), Cont::EvalExpr(cdr, env, k)))
}
