use lazy_static::lazy_static;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use crate::interpreter::cps::env::Env;
use crate::interpreter::cps::value::Value;
use crate::interpreter::cps::{List, RuntimeError};

use crate::{match_list, null, runtime_error};

type BuiltinFn = fn(List) -> Result<Value, RuntimeError>;

// 使用 lazy_static 来创建一个静态的 HashMap 存储所有内置函数
lazy_static! {
    static ref BUILTINS: HashMap<&'static str, BuiltinFn> = {
        let mut m = HashMap::new();
        m.insert("+", add as BuiltinFn);
        m.insert("*", mul as BuiltinFn);
        m.insert("-", sub as BuiltinFn);
        m.insert("/", div as BuiltinFn);
        m.insert("<", lt as BuiltinFn);
        m.insert(">", gt as BuiltinFn);
        m.insert("=", eq as BuiltinFn);
        m.insert("null?", null_pred as BuiltinFn);
        m.insert("integer?", integer_pred as BuiltinFn);
        m.insert("float?", float_pred as BuiltinFn);
        m.insert("list", list as BuiltinFn);
        m.insert("car", car as BuiltinFn);
        m.insert("cdr", cdr as BuiltinFn);
        m.insert("cons", cons as BuiltinFn);
        m.insert("append", append as BuiltinFn);
        m.insert("error", error as BuiltinFn);
        m.insert("write", write as BuiltinFn);
        m.insert("display", display as BuiltinFn);
        m.insert("displayln", displayln as BuiltinFn);
        m.insert("print", print as BuiltinFn);
        m.insert("env", env as BuiltinFn);
        m.insert("newline", newline as BuiltinFn);
        m.insert("pwd", current_working_directory as BuiltinFn);
        m.insert("ls-dir", list_directory as BuiltinFn);
        m
    };
}

// 获取所有内置函数名称的函数
pub fn get_builtin_names() -> Vec<&'static str> { BUILTINS.keys().cloned().collect() }

// 修改后的 primitive 函数
pub fn primitive(f: &'static str, args: List) -> Result<Value, RuntimeError> {
    match BUILTINS.get(f) {
        Some(func) => func(args),
        None => runtime_error!("Unknown primitive: {:?}", f),
    }
}

#[derive(Clone, PartialEq)]
pub enum Procedure {
    User(Vec<String>, List, Rc<RefCell<Env>>), // formals, body, env
    Native(&'static str),
}

impl fmt::Debug for Procedure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Procedure::User(_, _, _) => write!(f, "#<procedure>"),
            Procedure::Native(ref s) => write!(f, "#<native procedure:{}>", s),
        }
    }
}

fn list(args: List) -> Result<Value, RuntimeError> { Ok(args.into_list()) }

fn add(args: List) -> Result<Value, RuntimeError> { args.into_iter().try_fold(Value::Integer(0), |acc, arg| acc + arg) }

fn mul(args: List) -> Result<Value, RuntimeError> { args.into_iter().try_fold(Value::Integer(1), |acc, arg| acc * arg) }

fn div(args: List) -> Result<Value, RuntimeError> {
    match args.len() {
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
    }
}

fn sub(args: List) -> Result<Value, RuntimeError> {
    match args.len() {
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
    }
}

fn lt(args: List) -> Result<Value, RuntimeError> {
    if args.len() != 2 {
        runtime_error!("Must supply exactly two arguments to <: {:?}", args);
    }
    match_list!(args, [l, r] => Value::Boolean(l.into_integer()? < r.into_integer()?))
}

fn gt(args: List) -> Result<Value, RuntimeError> {
    if args.len() != 2 {
        runtime_error!("Must supply exactly two arguments to >: {:?}", args);
    }
    match_list!(args, [l, r] => Value::Boolean(l.into_integer()? > r.into_integer()?))
}

fn eq(args: List) -> Result<Value, RuntimeError> {
    if args.len() != 2 {
        runtime_error!("Must supply exactly two arguments to =: {:?}", args);
    }
    match_list!(args, [l, r] => Value::Boolean(l.into_integer()? == r.into_integer()?))
}

fn null_pred(args: List) -> Result<Value, RuntimeError> {
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

fn integer_pred(args: List) -> Result<Value, RuntimeError> {
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

fn float_pred(args: List) -> Result<Value, RuntimeError> {
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

fn car(args: List) -> Result<Value, RuntimeError> {
    if args.len() != 1 {
        runtime_error!("Must supply exactly two arguments to car: {:?}", args);
    }
    let l = args.car()?.into_list()?;
    match l.shift() {
        Some((car, _)) => Ok(car),
        None => runtime_error!("Can't run car on an empty list"),
    }
}

fn cdr(args: List) -> Result<Value, RuntimeError> {
    if args.len() != 1 {
        runtime_error!("Must supply exactly two arguments to cdr: {:?}", args);
    }
    let l = args.car()?.into_list()?;
    match l.shift() {
        Some((_, cdr)) => Ok(cdr.into_list()),
        None => runtime_error!("Can't run cdr on an empty list"),
    }
}

fn cons(args: List) -> Result<Value, RuntimeError> {
    if args.len() != 2 {
        runtime_error!("Must supply exactly two arguments to cons: {:?}", args);
    }
    match_list!(args, [elem, list] => list.into_list()?.unshift(elem).into_list())
}

fn append(args: List) -> Result<Value, RuntimeError> {
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

fn error(args: List) -> Result<Value, RuntimeError> {
    if args.len() != 1 {
        runtime_error!("Must supply exactly one argument to error: {:?}", args);
    }
    let msg = args.car()?;
    runtime_error!("{:?}", msg)
}

fn write(args: List) -> Result<Value, RuntimeError> {
    if args.len() != 1 {
        runtime_error!("Must supply exactly one argument to write: {:?}", args);
    }
    let val = args.car()?;
    print!("{:?}", val);
    Ok(null!())
}

fn display(args: List) -> Result<Value, RuntimeError> {
    if args.len() != 1 {
        runtime_error!("Must supply exactly one argument to display: {:?}", args);
    }
    let val = args.car()?;
    print!("{}", val);
    Ok(null!())
}

fn displayln(args: List) -> Result<Value, RuntimeError> {
    if args.len() != 1 {
        runtime_error!("Must supply exactly one argument to displayln: {:?}", args);
    }
    let val = args.car()?;
    println!("{}", val);
    Ok(null!())
}
fn print(args: List) -> Result<Value, RuntimeError> {
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

fn newline(args: List) -> Result<Value, RuntimeError> {
    if !args.is_empty() {
        runtime_error!("Must supply exactly zero arguments to newline: {:?}", args);
    }
    println!();
    Ok(null!())
}

fn env(args: List) -> Result<Value, RuntimeError> {
    if !args.is_empty() {
        runtime_error!("Must supply exactly zero arguments to env: {:?}", args);
    }
    Ok(Value::String("env".into()))
}

fn current_working_directory(args: List) -> Result<Value, RuntimeError> {
    if !args.is_empty() {
        runtime_error!("Must supply exactly zero arguments to pwd: {:?}", args);
    }

    let path = std::env::current_dir().map_err(|e| RuntimeError { message: e.to_string() })?;
    let path = path.to_str().ok_or_else(|| RuntimeError {
        message: "Path contains invalid UTF-8".to_string(),
    })?;
    Ok(Value::String(path.to_string()))
}

fn list_directory(args: List) -> Result<Value, RuntimeError> {
    let path = if args.is_empty() { ".".into() } else { args.car()?.to_string() };

    let full_path = std::fs::canonicalize(&path).map_err(|e| RuntimeError { message: e.to_string() })?;
    let entries = std::fs::read_dir(full_path).map_err(|e| RuntimeError { message: e.to_string() })?;

    let mut result = vec![];
    for entry in entries {
        let entry = entry.map_err(|e| RuntimeError { message: e.to_string() })?;
        let path = entry.path();
        let path = path.to_str().ok_or_else(|| RuntimeError {
            message: "Path contains invalid UTF-8".to_string(),
        })?;
        result.push(Value::String(path.to_string()));
    }
    Ok(Value::List(List::from_vec(result)))
}
