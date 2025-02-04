use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use tracing::info;

use crate::interpreter::cps::{eval, Cont, Env, List, RuntimeError, Value, SPECIAL_FORMS};
use crate::{match_list, null};

#[derive(PartialEq, Clone)]
pub enum Trampoline {
    Bounce(Value, Rc<RefCell<Env>>, Cont),
    QuasiquoteBounce(Value, Rc<RefCell<Env>>, Cont),
    Apply(Value, Cont),
    Land(Value), // Land 是 不需要eavl 的，自己就是最终结果, 只对应 Cont::Return
}

impl fmt::Debug for Trampoline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Trampoline::Bounce(val, _, k) => write!(f, "Bounce({:?}, env, {:?})", val, k),
            Trampoline::QuasiquoteBounce(val, _, k) => write!(f, "QuasiquoteBounce({:?}, env, {:?})", val, k),
            Trampoline::Apply(val, k) => write!(f, "Run({:?}, {:?})", val, k),
            Trampoline::Land(val) => write!(f, "Land({:?})", val),
        }
    }
}

// (unquote x) 会直接转到 Bounce 模式，否则会继续 QuasiBounce
pub fn quasiquote_bounce(val: Value, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::List(List::Null) => k.run(null!()),
        Value::List(List::Cell(box Value::Symbol(s), cdr)) if s == "unquote" => Ok(Trampoline::Bounce(cdr.car()?, env, k)),
        Value::List(List::Cell(car, cdr)) => {
            Ok(Trampoline::QuasiquoteBounce(*car, env.clone(), Cont::ContinueQuasiquote(*cdr, List::Null, env, Box::new(k))))
        }
        _ => k.run(val),
    }
}

pub fn bounce_symbol(symbol: &String, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    // 先从 special form 找，然后是env 中的各种定义
    let val: Value = SPECIAL_FORMS
        .get(symbol.as_ref())
        .map(|sf| Value::SpecialForm(sf.clone()))
        .or_else(|| env.borrow().get(symbol))
        .ok_or_else(|| RuntimeError {
            message: format!("Identifier not found: {}", symbol),
        })?;
    let val_clone = val.clone();
    let k_clone = k.clone();
    let result = k.run(val);
    info!("bounce symbol / {}, k: {:?}, val {} => {:?}", symbol, k_clone, val_clone, result);
    result
}

fn bounce_list(list: List, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    let expr = list.clone();
    let result = match_list!(list, head: car, tail: cdr => Trampoline::Bounce(car, env.clone(), Cont::BeginFunc(cdr, env, Box::new(k))));
    info!("bounce list / {} => {:?}", expr, result);
    result
}

pub fn bounce(val: Value, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::List(list) => bounce_list(list, env, k),   // 处理列表形式
        Value::Symbol(ref s) => bounce_symbol(s, env, k), // 处理符号形式
        _ => {
            // 处理其他所有形式
            let k_clone = k.clone();
            let val_clone = val.clone();
            info!("bounce other / k: {:?}, val: {:?}", k_clone, val_clone);
            k.run(val)
        }
    }
}

pub fn cps(expr: List, env: Rc<RefCell<Env>>) -> Result<Value, RuntimeError> {
    info!("{}", expr);

    if expr.is_empty() {
        return Ok(null!());
    }
    let mut result = eval(expr, env, Box::new(Cont::Return))?; // repl 顶层的 cont 就是 Return
    info!("{:?}", result);

    loop {
        match result {
            Trampoline::Bounce(val, env, k) => result = bounce(val, env, k)?,
            Trampoline::QuasiquoteBounce(val, env, k) => result = quasiquote_bounce(val, env, k)?,
            Trampoline::Apply(val, k) => result = k.clone().run(val.clone())?, // Run 将 k 应用到 val 上
            Trampoline::Land(val) => return Ok(val),                           // Land 会返回给的值; 最早创建，也是最后的 Trampoline 求值
        }
    }
}
