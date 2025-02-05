use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use crate::interpreter::cps::value::Value;
use crate::interpreter::cps::{get_builtin_names, Procedure, RuntimeError};
use crate::runtime_error;

#[derive(PartialEq)]
pub struct Env {
    pub parent: Option<Rc<RefCell<Env>>>,
    pub values: HashMap<String, Value>,
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.parent {
            Some(ref parent) => write!(f, "<Env {:?}>", parent.borrow()),
            None => write!(f, "<Env>"),
        }
    }
}

impl Env {
    pub fn new_root() -> Result<Rc<RefCell<Env>>, RuntimeError> {
        let mut env = Env {
            parent: None,
            values: HashMap::new(),
        };

        for name in get_builtin_names() {
            env.define(name.into(), Value::Procedure(Procedure::Native(name)))?;
        }
        Ok(Rc::new(RefCell::new(env)))
    }

    pub fn new_child(parent: Rc<RefCell<Env>>) -> Rc<RefCell<Env>> {
        let env = Env {
            parent: Some(parent),
            values: HashMap::new(),
        };
        Rc::new(RefCell::new(env))
    }

    // Define a variable at the current level
    // If key is not defined in the current env, set it
    // If key is already defined in the current env, return runtime error
    // (So if key is defined at a higher level, still define it at the current level)
    pub fn define(&mut self, key: String, value: Value) -> Result<(), RuntimeError> {
        let values = &self.values;
        if values.contains_key(&key) {
            runtime_error!("Duplicate define: {:?}", key)
        }
        self.values.insert(key, value);
        Ok(())
    }

    // Set a variable to a value, at any level in the env, or throw a runtime error if it isn't defined at all
    pub fn set(&mut self, key: String, value: Value) -> Result<(), RuntimeError> {
        match self.values.contains_key(&key) {
            true => {
                self.values.insert(key, value);
                Ok(())
            }
            false => {
                // Recurse up the environment tree until a value is found or the end is reached
                match self.parent {
                    Some(ref parent) => parent.borrow_mut().set(key, value),
                    None => runtime_error!("Can't set! an undefined variable: {:?}", key),
                }
            }
        }
    }

    pub fn get(&self, key: &String) -> Option<Value> {
        match self.values.get(key) {
            Some(val) => Some(val.clone()),
            None => {
                // Recurse up the environment tree until a value is found or the end is reached
                match self.parent {
                    Some(ref parent) => parent.borrow().get(key),
                    None => None,
                }
            }
        }
    }

    pub fn get_root(env_ref: Rc<RefCell<Env>>) -> Rc<RefCell<Env>> {
        let env = env_ref.borrow();
        match env.parent {
            Some(ref parent) => Env::get_root(parent.clone()),
            None => env_ref.clone(),
        }
    }
}
