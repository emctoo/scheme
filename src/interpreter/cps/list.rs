use std::fmt;
use std::vec;

use crate::reader::parser::*;

use crate::interpreter::cps::value::Value;
use crate::interpreter::cps::RuntimeError;

use crate::{runtime_error, shift_or_error};

#[derive(PartialEq, Clone)]
pub enum List {
    Cell(Box<Value>, Box<List>),
    Null,
}

// null == empty list
#[macro_export]
macro_rules! null {
    () => {
        List::Null.into_list()
    };
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

impl List {
    pub fn new() -> List { List::Null }

    pub fn from_vec(src: Vec<Value>) -> List { src.iter().rfold(List::Null, |acc, val| List::Cell(Box::new(val.clone()), Box::new(acc))) }

    pub fn from_nodes(nodes: &[Node]) -> List { List::from_vec(nodes.iter().map(Value::from_node).collect()) }

    pub fn is_empty(&self) -> bool { self == &List::Null }

    /// (car cdr) -> car
    pub fn car(self) -> Result<Value, RuntimeError> {
        let (car, cdr) = shift_or_error!(self, "Expected list of length 1, but was empty");
        if !cdr.is_empty() {
            runtime_error!("Expected list of length 1, but it had more elements")
        }
        Ok(car)
    }

    /// Null => None, List => Some((car cdr)
    pub fn shift(self) -> Option<(Value, List)> {
        match self {
            List::Null => None,
            List::Cell(car, cdr) => Some((*car, *cdr)),
        }
    }

    /// car => (car self)
    pub fn unshift(self, car: Value) -> List { List::Cell(Box::new(car), Box::new(self)) }

    /// list length
    pub fn len(&self) -> usize {
        match self {
            List::Cell(_, ref cdr) => 1 + cdr.len(),
            List::Null => 0,
        }
    }

    pub fn reverse(self) -> List { self.into_iter().fold(List::Null, |acc, v| List::Cell(Box::new(v), Box::new(acc))) }

    pub fn into_list(self) -> Value { Value::List(self) }

    pub fn into_vec(self) -> Vec<Value> { self.into_iter().collect() }

    pub fn into_iter(self) -> ListIterator { ListIterator(self) }
}

impl Default for List {
    fn default() -> Self { List::new() }
}

impl IntoIterator for List {
    type Item = Value;
    type IntoIter = vec::IntoIter<Value>;

    fn into_iter(self) -> Self::IntoIter { self.into_vec().into_iter() }
}

impl fmt::Display for List {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let strs: Vec<String> = self.clone().into_iter().map(|v| format!("{}", v)).collect();
        write!(f, "({})", &strs.join(" "))
    }
}

impl fmt::Debug for List {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let strs: Vec<String> = self.clone().into_iter().map(|v| format!("{:?}", v)).collect();
        write!(f, "({})", &strs.join(" "))
    }
}

pub struct ListIterator(List);

impl Iterator for ListIterator {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        let (car, cdr) = <List as Clone>::clone(&self.0).shift()?;
        self.0 = cdr;
        Some(car)
    }
}
