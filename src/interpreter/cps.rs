pub mod cont;
pub mod json;
pub mod matches;
pub mod tests;
pub mod trampoline;
pub mod value;

pub use cont::*;
pub use trampoline::*;
pub use value::*;

use crate::reader::parser::*;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::vec;

use phf::phf_map;
use serde::{Deserialize, Serialize};

use crate::match_list;

#[derive(Debug)]
pub struct RuntimeError {
    pub message: String,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "RuntimeError: {}", self.message) }
}

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

#[derive(Clone, PartialEq)]
pub enum Procedure {
    Scheme(Vec<String>, List, Rc<RefCell<Env>>),
    Native(&'static str),
}

impl fmt::Debug for Procedure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Procedure::Scheme(_, _, _) => write!(f, "#<procedure>"),
            Procedure::Native(ref s) => write!(f, "#<procedure:{}>", s),
        }
    }
}

#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
#[serde(into = "String")]
#[serde(try_from = "String")]
pub enum SpecialForm {
    If,
    Define,
    Set,
    Lambda,
    Let,
    Quote,
    Quasiquote,
    Eval,
    Apply,
    Begin,
    And,
    Or,
    CallCC,
    DefineSyntaxRule,
}

pub static SPECIAL_FORMS: phf::Map<&'static str, SpecialForm> = phf_map! {
    "if" => SpecialForm::If,
    "define" => SpecialForm::Define,
    "set!" => SpecialForm::Set,
    "lambda" => SpecialForm::Lambda,
    "λ" => SpecialForm::Lambda,
    "let" => SpecialForm::Let,
    "quote" => SpecialForm::Quote,
    "quasiquote" => SpecialForm::Quasiquote,
    "eval" => SpecialForm::Eval,
    "apply" => SpecialForm::Apply,
    "begin" => SpecialForm::Begin,
    "and" => SpecialForm::And,
    "or" => SpecialForm::Or,
    "call/cc" => SpecialForm::CallCC,
    "define-syntax-rule" => SpecialForm::DefineSyntaxRule,
};

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

impl List {
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

#[derive(PartialEq)]
pub struct Env {
    pub parent: Option<Rc<RefCell<Env>>>,
    pub values: HashMap<String, Value>,
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.parent {
            Some(ref parent) => write!(f, "{:?}, {:?}", self.values, parent.borrow()),
            None => write!(f, "{:?} ", self.values),
        }
    }
}

impl Env {
    pub fn new_root() -> Result<Rc<RefCell<Env>>, RuntimeError> {
        let mut env = Env {
            parent: None,
            values: HashMap::new(),
        };

        let natives = vec![
            "+",
            "-",
            "*",
            "/",
            "<",
            ">",
            "=",
            "null?",
            "integer?",
            "float?",
            "list",
            "car",
            "cdr",
            "cons",
            "append",
            "error",
            "write",
            "display",
            "displayln",
            "print",
            "newline",
        ];
        for name in natives {
            env.define(name.into(), Value::Procedure(Procedure::Native(name)))?;
        }
        Ok(Rc::new(RefCell::new(env)))
    }

    fn new_child(parent: Rc<RefCell<Env>>) -> Rc<RefCell<Env>> {
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

fn primitive(f: &'static str, args: List) -> Result<Value, RuntimeError> {
    match f {
        "+" => args.into_iter().try_fold(Value::Integer(0), |acc, arg| acc + arg),
        "*" => args.into_iter().try_fold(Value::Integer(1), |acc, arg| acc * arg),
        "-" => match args.len() {
            0 => runtime_error!("`-` requires at least one argument"),
            1 => {
                let val: Value = args.car()?; // unpack1(): -> Result<Value, RuntimeError>
                -val
                // why not -args.unpack1()?
            }
            _ => {
                let mut iter = args.into_iter();
                let initial = iter.next().unwrap(); // it's okay because len > 1
                iter.try_fold(initial, |acc, arg| acc - arg)
            }
        },
        "/" => match args.len() {
            0 => runtime_error!("`/` requires at least one argument"),
            1 => {
                match args.car()? {
                    // unpack1(): -> Result<Value, RuntimeError>
                    Value::Integer(val) => Ok(Value::Float(1.0 / val as f64)),
                    Value::Float(val) => Ok(Value::Float(1.0 / val)),
                    val => runtime_error!("Expected a number, but got: {:?}", val),
                }
            }
            _ => {
                let mut iter = args.into_iter();
                let initial = iter.next().unwrap(); // it's okay because len > 1
                iter.try_fold(initial, |acc, arg| acc / arg)
            }
        },
        "<" => {
            if args.len() != 2 {
                runtime_error!("Must supply exactly two arguments to <: {:?}", args);
            }
            match_list!(args, [l, r] => Value::Boolean(l.into_integer()? < r.into_integer()?))
        }
        ">" => {
            if args.len() != 2 {
                runtime_error!("Must supply exactly two arguments to >: {:?}", args);
            }
            match_list!(args, [l, r] => Value::Boolean(l.into_integer()? > r.into_integer()?))
        }
        "=" => {
            if args.len() != 2 {
                runtime_error!("Must supply exactly two arguments to =: {:?}", args);
            }
            match_list!(args, [l, r] => Value::Boolean(l.into_integer()? == r.into_integer()?))
        }
        "null?" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to null?: {:?}", args);
            }
            match_list!(args, [value] => {
                match value {
                    Value::List(l) => Value::Boolean(l.is_empty()),
                    _ => Value::Boolean(false),
                }
            })
        }
        "integer?" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to integer?: {:?}", args);
            }
            match_list!(args, [value] => {
                match value {
                    Value::Integer(_) => Value::Boolean(true),
                    _ => Value::Boolean(false),
                }
            })
        }
        "float?" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to real?: {:?}", args);
            }
            match_list!(args, [value] => {
                match value {
                    Value::Float(_) => Value::Boolean(true),
                    _ => Value::Boolean(false),
                }
            })
        }
        "list" => Ok(args.into_list()),
        "car" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly two arguments to car: {:?}", args);
            }
            let l = args.car()?.into_list()?;
            match l.shift() {
                Some((car, _)) => Ok(car),
                None => runtime_error!("Can't run car on an empty list"),
            }
        }
        "cdr" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly two arguments to cdr: {:?}", args);
            }
            let l = args.car()?.into_list()?;
            match l.shift() {
                Some((_, cdr)) => Ok(cdr.into_list()),
                None => runtime_error!("Can't run cdr on an empty list"),
            }
        }
        "cons" => {
            if args.len() != 2 {
                runtime_error!("Must supply exactly two arguments to cons: {:?}", args);
            }
            match_list!(args, [elem, list] => list.into_list()?.unshift(elem).into_list())
        }
        "append" => {
            if args.len() != 2 {
                runtime_error!("Must supply exactly two arguments to append: {:?}", args);
            }
            match_list!(args, [list1raw, list2raw] => {
                let list1 = list1raw.into_list()?;
                let mut list2 = list2raw.into_list()?;

                for elem in list1.reverse() {
                    list2 = list2.unshift(elem)
                }
                list2.into_list()
            })
        }
        "error" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to error: {:?}", args);
            }
            let msg = args.car()?;
            runtime_error!("{:?}", msg)
        }
        "write" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to write: {:?}", args);
            }
            let val = args.car()?;
            print!("{:?}", val);
            Ok(null!())
        }
        "display" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to display: {:?}", args);
            }
            let val = args.car()?;
            print!("{}", val);
            Ok(null!())
        }
        "displayln" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to displayln: {:?}", args);
            }
            let val = args.car()?;
            println!("{}", val);
            Ok(null!())
        }
        "print" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to print: {:?}", args);
            }
            let val = args.car()?;
            match val {
                Value::Symbol(_) | Value::List(_) => print!("'{:?}", val),
                _ => print!("{:?}", val),
            }
            Ok(null!())
        }
        "newline" => {
            if !args.is_empty() {
                runtime_error!("Must supply exactly zero arguments to newline: {:?}", args);
            }
            println!();
            Ok(null!())
        }
        _ => {
            runtime_error!("Unknown primitive: {:?}", f)
        }
    }
}
