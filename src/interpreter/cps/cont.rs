use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::interpreter::cps::{primitive, Env, List, Procedure, RuntimeError, SpecialForm, Trampoline, Value};
use crate::{match_list, null, runtime_error, shift_or_error};

#[derive(PartialEq, Clone, Debug)]
pub enum Cont {
    EvalExpr(List, Rc<RefCell<Env>>, Box<Cont>),

    BeginFunc(List, Rc<RefCell<Env>>, Box<Cont>),
    EvalFunc(Value, List, List, Rc<RefCell<Env>>, Box<Cont>),

    EvalIf(Value, Value, Rc<RefCell<Env>>, Box<Cont>),
    EvalDef(String, Rc<RefCell<Env>>, Box<Cont>),
    EvalSet(String, Rc<RefCell<Env>>, Box<Cont>),
    EvalLet(String, List, List, Rc<RefCell<Env>>, Box<Cont>),

    ContinueQuasiquote(List, List, Rc<RefCell<Env>>, Box<Cont>),

    Eval(Rc<RefCell<Env>>, Box<Cont>),
    EvalApplyArgs(Value, Rc<RefCell<Env>>, Box<Cont>),
    Apply(Value, Box<Cont>),

    EvalAnd(List, Rc<RefCell<Env>>, Box<Cont>),
    EvalOr(List, Rc<RefCell<Env>>, Box<Cont>),

    ExecCallCC(Box<Cont>),
    Return,
}

fn cont_special_define_syntax_rule(rest: List, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    match_list!(rest, [defn, body] => {
        let (name, arg_names_raw) = match defn.into_list()?.shift() {
            Some((car, cdr)) => (car.into_symbol()?, cdr),
            None => runtime_error!("Must supply at least two params to first argument in define-syntax-rule"),
        };
        let arg_names = arg_names_raw
            .into_iter()
            .map(|v| v.into_symbol())
            .collect::<Result<Vec<String>, RuntimeError>>()?;

        let _ = env.borrow_mut().define(name, Value::Macro(arg_names, Box::new(body)));
        Trampoline::Apply(null!(), k)
    })
}

fn cont_special_macro(rest: List, arg_names: Vec<String>, body: Value, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    let args = rest;
    if arg_names.len() != args.len() {
        runtime_error!("Must supply exactly {} arguments to macro: {:?}", arg_names.len(), args);
    }

    // Create a lookup table for symbol substitutions
    let mut substitutions = HashMap::new();
    for (name, value) in arg_names.into_iter().zip(args.into_iter()) {
        substitutions.insert(name, value);
    }

    let expanded = expand_macro(body, &substitutions); // Expand the macro
    Ok(Trampoline::Bounce(expanded, env, k)) // Finished expanding macro, now evaluate the code manually
}

fn expand_macro(expr: Value, substitutions: &HashMap<String, Value>) -> Value {
    match expr {
        Value::Symbol(s) => substitutions.get(&s).cloned().unwrap_or(Value::Symbol(s)),
        Value::List(list) => Value::from_vec(list.into_iter().map(|val| expand_macro(val, substitutions)).collect()),
        _ => expr,
    }
}

/// define 的形式：
/// 1. (define name value)
///    (define x 42)
/// 2. (define (name arg1 arg2 ...) body), 固定数量的参数。
///
///    (define (add x y)
///      (+ x y))
///
///    (add 2 3) ; => 5
/// 3. (define (name . args) body), 可变参数函数
///
///    (define (sum . numbers)
///      (apply + numbers))
///
///    (sum 1 2 3 4) ; 返回 10
///
///    (define (sum x y . numbers)
///      (* x y (apply + numbers)))
///
///    (sum 2 3 4 5) ; 返回 54
///    (sum 1 2)     ; 返回 2
///    (sum 3 4 1 2) ; 返回 21
fn cont_special_def(rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    let (car, cdr) = shift_or_error!(rest, "Must provide at least two arguments to define");
    match car {
        Value::Symbol(name) => match_list!(cdr, [val] => Trampoline::Bounce(val, env.clone(), Cont::EvalDef(name, env, k))), // 形式1
        Value::List(list) => {
            // 形式2
            let (caar, cdar) = shift_or_error!(list, "Must provide at least two params in first argument of define");
            let fn_name = caar.into_symbol()?;

            let arg_names = cdar.into_iter().map(|v| v.into_symbol()).collect::<Result<Vec<String>, RuntimeError>>()?;
            let body = cdr;

            env.borrow_mut()
                .define(fn_name, Value::Procedure(Procedure::UserPr(arg_names, body, env.clone())))?;
            Ok(Trampoline::Apply(null!(), *k))
        }
        _ => runtime_error!("Bad argument to define: {:?}", car),
    }
}

fn cont_special_lambda(rest: List, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    let (arg_defns_raw, body) = shift_or_error!(rest, "Must provide at least two arguments to lambda");
    let arg_defns = arg_defns_raw.into_list()?;

    let arg_names = arg_defns
        .into_iter()
        .map(|v| v.into_symbol())
        .collect::<Result<Vec<String>, RuntimeError>>()?;

    let f = Procedure::UserPr(arg_names, body, env);
    Ok(Trampoline::Apply(Value::Procedure(f), k))
}

fn cont_special_let(rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    let (arg_def_raws, body) = shift_or_error!(rest, "Must provide at least two arguments to let");
    let arg_defs = arg_def_raws.into_list()?;

    let proc_env = Env::new_child(env.clone()); // 创建一个新的环境，用于存放 let 绑定的变量
    match arg_defs.is_empty() {
        true => eval(body, env, k), // 执行 body
        false => {
            let (first_def, rest_defs) = shift_or_error!(arg_defs, "Error in let definiton");
            match_list!(first_def.into_list()?, [def_key, def_val] => {
                Trampoline::Bounce(def_val, env, Cont::EvalLet(def_key.into_symbol()?, rest_defs, body, proc_env, k))
            })
        }
    }
}

fn cont_eval_let(name: String, value: Value, rest: List, body: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    env.borrow_mut().define(name, value)?; // define variable in let scope
    match rest.shift() {
        Some((next_defn, rest_defns)) => match_list!(next_defn.into_list()?, [defn_key, defn_val] => {
            Trampoline::Bounce(defn_val, env.clone(), Cont::EvalLet(defn_key.into_symbol()?, rest_defns, body, env, k))
        }),
        None => eval(body, Env::new_child(env), k),
    }
}

fn cont_special_quasiquote(rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match_list!(rest, [expr] => {
        match expr {
            Value::List(list) => match list.shift() {
                Some((car, cdr)) => Trampoline::QuasiquoteBounce(car, env.clone(), Cont::ContinueQuasiquote(cdr, List::Null, env, k)),
                None => Trampoline::Apply(null!(), *k),
            },
            // 其他类型的都用 cont 直接算
            _ => Trampoline::Apply(expr, *k),
        }
    })
}

fn cont_continue_quasiquote(val: Value, rest: List, acc: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    let acc2 = acc.unshift(val);
    match rest.shift() {
        Some((car, cdr)) => Ok(Trampoline::QuasiquoteBounce(car, env.clone(), Cont::ContinueQuasiquote(cdr, acc2, env, k))),
        None => Ok(Trampoline::Apply(acc2.reverse().into_list(), *k)),
    }
}

fn cont_special(sf: SpecialForm, rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match sf {
        SpecialForm::If => {
            match_list!(rest, [condition, if_expr, else_expr] => Trampoline::Bounce(condition, env.clone(), Cont::EvalIf(if_expr, else_expr, env, k)))
        }
        SpecialForm::Define => cont_special_def(rest, env, k),

        SpecialForm::Set => match_list!(rest, [name, val] => { // 求值，然后 set 到 env 中
            Trampoline::Bounce(val, env.clone(), Cont::EvalSet(name.into_symbol()?, env, k))
        }),

        SpecialForm::Lambda => cont_special_lambda(rest, env, *k),
        SpecialForm::Let => cont_special_let(rest, env, k),

        SpecialForm::Quote => Ok(Trampoline::Apply(rest.car()?, *k)),
        SpecialForm::Quasiquote => cont_special_quasiquote(rest, env, k),

        SpecialForm::Eval => Ok(Trampoline::Bounce(rest.car()?, env.clone(), Cont::Eval(env, k))),
        SpecialForm::Apply => match_list!(rest, [f, args] => Trampoline::Bounce(f, env.clone(), Cont::EvalApplyArgs(args, env, k))),
        SpecialForm::Begin => match_list!(rest, head: car, tail: cdr => Trampoline::Bounce(car, env.clone(), Cont::EvalExpr(cdr, env, k))),

        SpecialForm::And => match rest.shift() {
            Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalAnd(cdr, env, k))),
            None => Ok(Trampoline::Apply(Value::Boolean(true), *k)),
        },

        SpecialForm::Or => match rest.shift() {
            Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalOr(cdr, env, k))),
            None => Ok(Trampoline::Apply(Value::Boolean(false), *k)),
        },

        SpecialForm::CallCC => Ok(Trampoline::Bounce(rest.car()?, env, Cont::ExecCallCC(k))),
        SpecialForm::DefineSyntaxRule => cont_special_define_syntax_rule(rest, env, *k),
    }
}

impl Cont {
    pub fn run(self, val: Value) -> Result<Trampoline, RuntimeError> {
        match self {
            // eval list
            Cont::EvalExpr(rest, env, k) => match rest.is_empty() {
                true => Ok(Trampoline::Apply(val, *k)),
                false => eval(rest, env, k),
            },

            // bounce list
            Cont::BeginFunc(rest, env, k) => match val {
                Value::Macro(arg_names, body) => cont_special_macro(rest, arg_names, *body, env, *k),
                Value::SpecialForm(sf) => cont_special(sf, rest, env, k),
                _ => match rest.shift() {
                    Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalFunc(val, cdr, List::Null, env, k))),
                    None => apply(val, List::Null, k),
                },
            },

            // 函数参数
            Cont::EvalFunc(f, rest, acc, env, k) => {
                let acc2 = acc.unshift(val);
                match rest.shift() {
                    Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalFunc(f, cdr, acc2, env, k))),
                    None => apply(f, acc2.reverse(), k),
                }
            }

            // if/else 情况都是 bounce 回去继续计算
            Cont::EvalIf(if_expr, else_expr, env, k) => match val {
                Value::Boolean(false) => Ok(Trampoline::Bounce(else_expr, env, *k)),
                _ => Ok(Trampoline::Bounce(if_expr, env, *k)),
            },

            // (define name value), 创建一个新的绑定
            Cont::EvalDef(name, env, k) => {
                env.borrow_mut().define(name, val)?;
                Ok(Trampoline::Apply(null!(), *k))
            }

            Cont::EvalSet(name, env, k) => {
                env.borrow_mut().set(name, val)?;
                Ok(Trampoline::Apply(null!(), *k))
            }

            Cont::EvalLet(name, rest, body, env, k) => cont_eval_let(name, val, rest, body, env, k),

            Cont::ContinueQuasiquote(rest, acc, env, k) => cont_continue_quasiquote(val, rest, acc, env, k),

            Cont::Apply(f, k) => apply(f, val.into_list()?, k),
            Cont::ExecCallCC(k) => apply(val, List::Null.unshift(Value::Cont(k.clone())), k),

            Cont::EvalAnd(rest, env, k) => match val {
                Value::Boolean(false) => Ok(Trampoline::Apply(Value::Boolean(false), *k)),
                _ => match rest.shift() {
                    Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalAnd(cdr, env, k))),
                    None => Ok(Trampoline::Apply(val, *k)),
                },
            },
            Cont::EvalOr(rest, env, k) => match val {
                Value::Boolean(false) => match rest.shift() {
                    Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalOr(cdr, env, k))),
                    None => Ok(Trampoline::Apply(Value::Boolean(false), *k)),
                },
                _ => Ok(Trampoline::Apply(val, *k)),
            },

            Cont::Eval(env, k) => Ok(Trampoline::Bounce(val, Env::get_root(env), *k)),
            Cont::EvalApplyArgs(args, env, k) => Ok(Trampoline::Bounce(args, env, Cont::Apply(val, k))),

            Cont::Return => Ok(Trampoline::Land(val)),
        }
    }
}

/// apply cont/procedure
fn apply(val: Value, args: List, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Cont(c) => Ok(Trampoline::Apply(args.into_list(), *c)),

        Value::Procedure(Procedure::NativePr(f)) => Ok(Trampoline::Apply(primitive(f, args)?, *k)),
        Value::Procedure(Procedure::UserPr(formals, body, env)) => {
            if formals.len() != args.len() {
                runtime_error!("Must supply exactly {} arguments to function: {:?}", formals.len(), args);
            }
            let proc_env = Env::new_child(env); // 创建一个新的 env，用于存放函数的参数
            formals
                .into_iter()
                .zip(args)
                .try_for_each(|(name, value)| proc_env.borrow_mut().define(name, value))?;
            eval(body, Env::new_child(proc_env), k)
        }
        _ => runtime_error!("Don't know how to apply: {:?}", val),
    }
}

pub fn eval(expr: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match_list!(expr, head: car, tail: cdr => Trampoline::Bounce(car, env.clone(), Cont::EvalExpr(cdr, env, k)))
}
