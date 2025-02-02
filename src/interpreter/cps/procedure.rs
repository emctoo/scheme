use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::interpreter::cps::env::Env;
use crate::interpreter::cps::value::Value;
use crate::interpreter::cps::{List, RuntimeError};

use crate::{match_list, null, runtime_error};

#[derive(Clone, PartialEq)]
pub enum Procedure {
    UserPr(Vec<String>, List, Rc<RefCell<Env>>),
    NativePr(&'static str),
}

impl fmt::Debug for Procedure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Procedure::UserPr(_, _, _) => write!(f, "#<procedure>"),
            Procedure::NativePr(ref s) => write!(f, "#<procedure:{}>", s),
        }
    }
}
pub fn primitive(f: &'static str, args: List) -> Result<Value, RuntimeError> {
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
