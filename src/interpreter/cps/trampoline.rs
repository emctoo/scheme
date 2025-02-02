use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

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

// 定义 Trampoline trait
pub trait Trampolinable {
    /// 将自身转换为 Trampoline 形式
    fn bounce(self, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError>;

    /// 将自身转换为 QuasiquoteBounce 形式
    fn quasiquote_bounce(self, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError>;
}

// Value 的实现
impl Trampolinable for Value {
    fn bounce(self, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> { bounce_value(self, env, k) }
    fn quasiquote_bounce(self, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> { quasiquote_bounce(self, env, k) }
}

// List 的实现
impl Trampolinable for List {
    fn bounce(self, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> { bounce_list(self, env, k) }
    fn quasiquote_bounce(self, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> { quasiquote_bounce(Value::List(self), env, k) }
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

pub fn bounce_symbol(s: &String, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    // 先从 special form 找，然后是env 中的各种定义
    let val: Value = SPECIAL_FORMS
        .get(s.as_ref())
        .map(|sf| Value::SpecialForm(sf.clone()))
        .or_else(|| env.borrow().get(s))
        .ok_or_else(|| RuntimeError {
            message: format!("Identifier not found: {}", s),
        })?;
    k.run(val)
}

fn bounce_list(list: List, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    match_list!(list, head: car, tail: cdr => Trampoline::Bounce(car, env.clone(), Cont::BeginFunc(cdr, env, Box::new(k))))
}

pub fn bounce_value(val: Value, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::List(list) => bounce_list(list, env, k),   // 处理列表形式
        Value::Symbol(ref s) => bounce_symbol(s, env, k), // 处理符号形式
        _ => k.run(val),                                  // 处理其他所有形式
    }
}
pub fn eval_cps(expr: List, env: Rc<RefCell<Env>>) -> Result<Value, RuntimeError> {
    if expr.is_empty() {
        return Ok(null!());
    }
    let mut result = eval(expr, env, Box::new(Cont::Return))?; // repl 顶层的 cont 就是 Return
    loop {
        match result {
            Trampoline::Bounce(val, env, k) => result = val.bounce(env, k)?, // 常规的 Bounce
            Trampoline::QuasiquoteBounce(val, env, k) => result = val.quasiquote_bounce(env, k)?, // quasiquote Bounce 模式
            Trampoline::Apply(val, k) => result = k.run(val)?,               // Run 将 k 应用到 val 上
            Trampoline::Land(val) => return Ok(val),                         // Land 会返回给的值; 最早创建，也是最后的 Trampoline 求值
        }
    }
}
