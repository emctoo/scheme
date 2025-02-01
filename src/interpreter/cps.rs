use crate::reader::parser::*;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::vec;

use phf::phf_map;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use tracing::info;

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

    pub fn run(&self, nodes: &[Node]) -> Result<Value, RuntimeError> { process(List::from_nodes(nodes), self.root.clone()) }
}

macro_rules! runtime_error {
    ($($arg:tt)*) => (
        return Err(RuntimeError { message: format!($($arg)*)})
    )
}

macro_rules! shift_or_error {
    ($list:expr, $($arg:tt)*) => (
        match $list.shift() {
            Some((car, cdr)) => Ok((car, cdr)),
            None => Err(RuntimeError { message: format!($($arg)*)})
        }?
    )
}

#[derive(PartialEq, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Value {
    Symbol(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),

    List(List),

    #[serde(skip)]
    Procedure(Function),

    #[serde(skip)]
    SpecialForm(SpecialForm),

    Macro(Vec<String>, Box<Value>),

    #[serde(skip)]
    Continuation(Box<Cont>),
}

impl std::ops::Add for Value {
    type Output = Result<Value, RuntimeError>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a + b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a + b as f64)),
            (a, b) => runtime_error!("Cannot `+` {:?} and {:?}", a, b),
        }
    }
}

impl std::ops::Sub for Value {
    type Output = Result<Value, RuntimeError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a - b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(a as f64 - b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a - b as f64)),
            (a, b) => runtime_error!("Cannot `-` {:?} and {:?}", a, b),
        }
    }
}

impl std::ops::Mul for Value {
    type Output = Result<Value, RuntimeError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a * b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(a as f64 * b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a * b as f64)),
            (a, b) => runtime_error!("Cannot `*` {:?} and {:?}", a, b),
        }
    }
}

impl std::ops::Div for Value {
    type Output = Result<Value, RuntimeError>;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a / b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(a as f64 / b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a / b as f64)),
            (a, b) => runtime_error!("Cannot `/` {:?} and {:?}", a, b),
        }
    }
}

impl std::ops::Neg for Value {
    type Output = Result<Value, RuntimeError>;

    fn neg(self) -> Self::Output {
        match self {
            Value::Integer(a) => Ok(Value::Integer(-a)),
            Value::Float(a) => Ok(Value::Float(-a)),
            x => runtime_error!("Cannot `-` {:?}", x),
        }
    }
}

impl Value {
    fn from_vec(vec: Vec<Value>) -> Value { List::from_vec(vec).into_list() }

    fn from_node(node: &Node) -> Value {
        match *node {
            Node::Identifier(ref val) => Value::Symbol(val.clone()),
            Node::Integer(val) => Value::Integer(val),
            Node::Float(val) => Value::Float(val),
            Node::Boolean(val) => Value::Boolean(val),
            Node::String(ref val) => Value::String(val.clone()),
            Node::List(ref nodes) => Value::List(List::from_nodes(nodes)),
        }
    }

    fn into_symbol(self) -> Result<String, RuntimeError> {
        match self {
            Value::Symbol(s) => Ok(s),
            _ => runtime_error!("Expected a symbol value: {:?}", self),
        }
    }

    fn into_integer(self) -> Result<i64, RuntimeError> {
        match self {
            Value::Integer(i) => Ok(i),
            _ => runtime_error!("Expected an integer value: {:?}", self),
        }
    }

    fn into_list(self) -> Result<List, RuntimeError> {
        match self {
            Value::List(l) => Ok(l),
            _ => runtime_error!("Expected a list value: {:?}", self),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Value::Symbol(ref val) => write!(f, "{}", val),
            Value::Integer(val) => write!(f, "{}", val),
            Value::Float(val) => write!(f, "{}", val),
            Value::Boolean(val) => write!(f, "#{}", if val { "t" } else { "f" }),
            Value::String(ref val) => write!(f, "{}", val),
            Value::List(ref list) => write!(f, "{}", list),
            Value::Procedure(_) => write!(f, "#<procedure>"),
            Value::SpecialForm(_) => write!(f, "#<special_form>"),
            Value::Continuation(_) => write!(f, "#<continuation>"),
            Value::Macro(_, _) => write!(f, "#<macro>"),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Value::String(ref val) => write!(f, "\"{}\"", val),
            Value::List(ref list) => write!(f, "{:?}", list),
            _ => write!(f, "{}", self),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Function {
    Scheme(Vec<String>, List, Rc<RefCell<Env>>),
    Native(&'static str),
}

impl fmt::Debug for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Function::Scheme(_, _, _) => write!(f, "#<procedure>"),
            Function::Native(ref s) => write!(f, "#<procedure:{}>", s),
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
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

#[derive(Debug)]
pub struct RuntimeError {
    message: String,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "RuntimeError: {}", self.message) }
}

#[derive(PartialEq, Clone)]
pub enum List {
    Cell(Box<Value>, Box<List>),
    Null,
}

impl Serialize for List {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // 将 List 转换为 Vec 再序列化
        let elements: Vec<Value> = self.clone().into_iter().collect();
        let mut seq = serializer.serialize_seq(Some(elements.len()))?;
        for element in elements {
            seq.serialize_element(&element)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for List {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ListVisitor;

        impl<'de> Visitor<'de> for ListVisitor {
            type Value = List;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result { formatter.write_str("a sequence") }

            fn visit_seq<V>(self, mut seq: V) -> Result<List, V::Error>
            where
                V: SeqAccess<'de>,
            {
                // 从序列构建 List
                let mut result = List::Null;
                while let Some(value) = seq.next_element()? {
                    result = List::Cell(Box::new(value), Box::new(result));
                }
                Ok(result.reverse())
            }
        }

        deserializer.deserialize_seq(ListVisitor)
    }
}

pub fn serialize_list(list: &List) -> Result<String, serde_json::Error> { serde_json::to_string(list) }

pub fn deserialize_list(json: &str) -> Result<List, serde_json::Error> { serde_json::from_str(json) }

#[cfg(test)]
mod test_list_serialization {
    // use crate::interpreter::interpreter::parse_code;

    use super::*;

    // #[test]
    // fn test_parsing() {
    //     let list = parse_code(r#"(1 "hello" true)"#);
    //     serde_json::to_string(&list).unwrap();
    // }

    #[test]
    fn test_list_serialization() {
        // runTest (1 "hello" true)

        // 创建测试数据
        let list = List::Cell(
            Box::new(Value::Integer(1)),
            Box::new(List::Cell(
                Box::new(Value::String("hello".to_string())),
                Box::new(List::Cell(Box::new(Value::Boolean(true)), Box::new(List::Null))),
            )),
        );

        // 序列化
        let json = serialize_list(&list).unwrap();
        assert_eq!(json, r#"[{"type":"Integer","value":1},{"type":"String","value":"hello"},{"type":"Boolean","value":true}]"#,);

        // 反序列化
        let deserialized: List = deserialize_list(&json).unwrap();

        // 验证
        let original_vec: Vec<Value> = list.into_iter().collect();
        let deserialized_vec: Vec<Value> = deserialized.into_iter().collect();
        assert_eq!(original_vec, deserialized_vec);
    }

    #[test]
    fn test_empty_list() {
        let list = List::Null;
        let json = serialize_list(&list).unwrap();
        let deserialized: List = deserialize_list(&json).unwrap();
        assert_eq!(list, deserialized);
    }
}

// null == empty list
macro_rules! null {
    () => {
        List::Null.into_list()
    };
}

macro_rules! match_list {
    // 匹配空列表
    ($list:expr, [] => $empty_expr:expr) => {
        match $list {
            List::Null => Ok($empty_expr),
            _ => Err(RuntimeError {
                message: "Expected empty list".into(),
            }),
        }
    };

    // 匹配单个元素
    ($list:expr, [$x:pat] => $expr:expr) => {
        match $list {
            List::Cell(box $x, box List::Null) => Ok($expr),
            _ => Err(RuntimeError {
                message: "Expected list of length 1".into(),
            }),
        }
    };

    // 匹配两个元素
    ($list:expr, [$x:pat, $y:pat] => $expr:expr) => {
        match $list {
            List::Cell(box $x, box List::Cell(box $y, box List::Null)) => Ok($expr),
            _ => Err(RuntimeError {
                message: "Expected list of length 2".into(),
            }),
        }
    };

    // 匹配三个元素
    ($list:expr, [$x:pat, $y:pat, $z:pat] => $expr:expr) => {
        match $list {
            List::Cell(box $x, box List::Cell(box $y, box List::Cell(box $z, box List::Null))) => Ok($expr),
            _ => Err(RuntimeError {
                message: "Expected list of length 3".into(),
            }),
        }
    };

    // 匹配 head 和 tail
    ($list:expr, head: $x:pat, tail: $xs:pat => $expr:expr) => {
        match $list {
            List::Cell(box $x, box $xs) => Ok($expr),
            _ => Err(RuntimeError {
                message: "Expected non-empty list".into(),
            }),
        }
    };
}

macro_rules! match_list_multi {
    ($list:expr, {
        $( [$($pat:tt)*] => $expr:expr ),+ $(,)*
    }) => {{
        let candidate = $list.clone(); // 原列表
        let mut result: Result<_, RuntimeError> = Err(RuntimeError {
            message: "No pattern matched".into(),
        });
        $(
            // 依次用 match_list! 尝试匹配
            if result.is_err() {
                result = match_list!(candidate.clone(), [$($pat)*] => $expr);
            }
        )+
        result
    }};
}

#[cfg(test)]
mod test_match_list {
    use super::*;

    // 辅助函数,创建测试用的列表
    fn create_list(values: Vec<Value>) -> List {
        let mut list = List::Null;
        for value in values.into_iter().rev() {
            list = List::Cell(Box::new(value), Box::new(list));
        }
        list
    }

    #[test]
    fn test_empty_list_success() {
        let empty = List::Null;
        let result: Result<bool, RuntimeError> = match_list!(empty, [] => true);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_empty_list_failure() {
        // 测试非空列表应该返回错误
        let non_empty = create_list(vec![Value::Integer(1)]);
        assert!(matches!(match_list!(non_empty, [] => true), Err::<bool, RuntimeError>(_)));
    }

    #[test]
    fn test_single_element() {
        let single = create_list(vec![Value::Integer(1)]);
        let result = match_list!(single, [Value::Integer(n)] => n == 1);
        assert!(result.unwrap());

        // 测试空列表应该panic
        let empty = List::Null;
        let result = match_list!(empty, [Value::Integer(_)] => true);
        assert!(result.is_err());
    }

    #[test]
    fn test_two_elements() {
        let two = create_list(vec![Value::Integer(1), Value::Integer(2)]);
        let result = match_list!(two, [Value::Integer(x), Value::Integer(y)] => x == 1 && y == 2);
        assert!(result.unwrap());

        // 测试长度不匹配应该panic
        let one = create_list(vec![Value::Integer(1)]);
        let result = match_list!(one, [Value::Integer(_), Value::Integer(_)] => true);
        assert!(result.is_err());
    }

    #[test]
    fn test_three_elements() {
        let three = create_list(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]);
        let result = match_list!(three,
            [Value::Integer(x), Value::Integer(y), Value::Integer(z)] =>
            x == 1 && y == 2 && z == 3
        );
        assert!(result.unwrap());

        // 测试长度不匹配应该panic
        let two = create_list(vec![Value::Integer(1), Value::Integer(2)]);
        let result = match_list!(two, [Value::Integer(_), Value::Integer(_), Value::Integer(_)] => true);
        assert!(result.is_err());
    }

    #[test]
    fn test_head_tail() {
        let list = create_list(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]);
        let result = match_list!(list, head: Value::Integer(x), tail: rest => {
            x == 1 && matches!(rest, List::Cell(..))
        });
        assert!(result.unwrap());

        // 测试空列表应该panic
        let empty = List::Null;
        let result = match_list!(empty, head: Value::Integer(_), tail: _ => true);
        assert!(result.is_err());
    }

    #[test]
    fn test_match_scheme() {
        let list1 = List::from_vec(vec![Value::Integer(1), Value::Integer(2)]);
        let result1 = match_list_multi!(list1, {
            [Value::Integer(x), Value::Integer(y)] => x + y,
            [Value::Integer(x)] => x
        });
        assert_eq!(result1.unwrap(), 3);

        let list2 = List::from_vec(vec![Value::String("hello".to_string())]);
        let result2 = match_list_multi!(list2, {
            [Value::Integer(x), Value::Integer(y)] => x + y,
            [Value::String(s)] => s.len() as i64
        });
        assert_eq!(result2.unwrap(), 5);

        let list3 = List::from_vec(vec![Value::Boolean(true)]);
        let result3 = match_list_multi!(list3, {
            [Value::Integer(x)] => x,
            [Value::String(s)] => s.len() as i64
        });
        assert!(result3.is_err());
    }
}

impl List {
    fn from_vec(src: Vec<Value>) -> List { src.iter().rfold(List::Null, |acc, val| List::Cell(Box::new(val.clone()), Box::new(acc))) }

    fn from_nodes(nodes: &[Node]) -> List { List::from_vec(nodes.iter().map(Value::from_node).collect()) }

    fn is_empty(&self) -> bool { self == &List::Null }

    /// (car cdr) -> car
    fn car(self) -> Result<Value, RuntimeError> {
        let (car, cdr) = shift_or_error!(self, "Expected list of length 1, but was empty");
        if !cdr.is_empty() {
            runtime_error!("Expected list of length 1, but it had more elements")
        }
        Ok(car)
    }

    /// Null => None, List => Some((car cdr)
    fn shift(self) -> Option<(Value, List)> {
        match self {
            List::Null => None,
            List::Cell(car, cdr) => Some((*car, *cdr)),
        }
    }

    /// car => (car self)
    fn unshift(self, car: Value) -> List { List::Cell(Box::new(car), Box::new(self)) }

    /// list length
    fn len(&self) -> usize {
        match self {
            List::Cell(_, ref cdr) => 1 + cdr.len(),
            List::Null => 0,
        }
    }

    fn reverse(self) -> List { self.into_iter().fold(List::Null, |acc, v| List::Cell(Box::new(v), Box::new(acc))) }

    fn into_list(self) -> Value { Value::List(self) }

    fn into_vec(self) -> Vec<Value> { self.into_iter().collect() }

    fn into_iter(self) -> ListIterator { ListIterator(self) }
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

struct ListIterator(List);

impl Iterator for ListIterator {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        let (car, cdr) = <List as Clone>::clone(&self.0).shift()?;
        self.0 = cdr;
        Some(car)
    }
}

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

pub enum Trampoline {
    Bounce(Value, Rc<RefCell<Env>>, Cont),
    QuasiquoteBounce(Value, Rc<RefCell<Env>>, Cont),
    Run(Value, Cont),
    Land(Value),
}

impl fmt::Debug for Trampoline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Trampoline::Bounce(val, _, k) => write!(f, "Bounce({:?}, env, {:?})", val, k),
            Trampoline::QuasiquoteBounce(val, _, k) => write!(f, "QuasiquoteBounce({:?}, env, {:?})", val, k),
            Trampoline::Run(val, k) => write!(f, "Run({:?}, {:?})", val, k),
            Trampoline::Land(val) => write!(f, "Land({:?})", val),
        }
    }
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
        Trampoline::Run(null!(), k)
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

fn cont_special_def(rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    let (car, cdr) = shift_or_error!(rest, "Must provide at least two arguments to define");
    match car {
        Value::Symbol(name) => match_list!(cdr, [val] => Trampoline::Bounce(val, env.clone(), Cont::EvalDef(name, env, k))),
        Value::List(list) => {
            let (caar, cdar) = shift_or_error!(list, "Must provide at least two params in first argument of define");
            let name = caar.into_symbol()?;

            let arg_names = cdar.into_iter().map(|v| v.into_symbol()).collect::<Result<Vec<String>, RuntimeError>>()?;
            let body = cdr;
            let f = Function::Scheme(arg_names, body, env.clone());

            env.borrow_mut().define(name, Value::Procedure(f))?;
            Ok(Trampoline::Run(null!(), *k))
        }
        _ => runtime_error!("Bad argument to define: {:?}", car),
    }
}

fn cont_eval_def(name: String, val: Value, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    env.borrow_mut().define(name, val)?;
    Ok(Trampoline::Run(null!(), k))
}

fn cont_special_lambda(rest: List, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    let (arg_defns_raw, body) = shift_or_error!(rest, "Must provide at least two arguments to lambda");
    let arg_defns = arg_defns_raw.into_list()?;

    let arg_names = arg_defns
        .into_iter()
        .map(|v| v.into_symbol())
        .collect::<Result<Vec<String>, RuntimeError>>()?;

    let f = Function::Scheme(arg_names, body, env);
    Ok(Trampoline::Run(Value::Procedure(f), k))
}

fn cont_begin_fn(val: Value, rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Macro(arg_names, body) => cont_special_macro(rest, arg_names, *body, env, *k),
        Value::SpecialForm(f) => cont_special(f, rest, env, k),
        _ => match rest.shift() {
            Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalFunc(val, cdr, List::Null, env, k))),
            None => apply(val, List::Null, k),
        },
    }
}

fn cont_eval_fn(f: Value, val: Value, rest: List, acc: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    let acc2 = acc.unshift(val);
    match rest.shift() {
        Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalFunc(f, cdr, acc2, env, k))),
        None => apply(f, acc2.reverse(), k),
    }
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

fn cont_special_if(rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match_list!(rest, [condition, if_expr, else_expr] => {
        Trampoline::Bounce(condition, env.clone(), Cont::EvalIf(if_expr, else_expr, env, k))
    })
}

fn cont_eval_if(val: Value, if_expr: Value, else_expr: Value, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Boolean(false) => Ok(Trampoline::Bounce(else_expr, env, k)),
        _ => Ok(Trampoline::Bounce(if_expr, env, k)),
    }
}

fn cont_special_set(rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match_list!(rest, [name, val] => {
        Trampoline::Bounce(val, env.clone(), Cont::EvalSet(name.into_symbol()?, env, k))
    })
}

fn cont_eval_set(name: String, val: Value, env: Rc<RefCell<Env>>, k: Cont) -> Result<Trampoline, RuntimeError> {
    env.borrow_mut().set(name, val)?;
    Ok(Trampoline::Run(null!(), k))
}

fn cont_special_quasiquote(rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match_list!(rest, [expr] => {
        match expr {
            Value::List(list) => match list.shift() {
                Some((car, cdr)) => Trampoline::QuasiquoteBounce(car, env.clone(), Cont::ContinueQuasiquote(cdr, List::Null, env, k)),
                None => Trampoline::Run(null!(), *k),
            },
            // 其他类型的都用 cont 直接算
            _ => Trampoline::Run(expr, *k),
        }
    })
}

fn cont_continue_quasiquote(val: Value, rest: List, acc: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    let acc2 = acc.unshift(val);
    match rest.shift() {
        Some((car, cdr)) => Ok(Trampoline::QuasiquoteBounce(car, env.clone(), Cont::ContinueQuasiquote(cdr, acc2, env, k))),
        None => Ok(Trampoline::Run(acc2.reverse().into_list(), *k)),
    }
}

fn cont_special_apply(rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match_list!(rest, [f, args] => Trampoline::Bounce(f, env.clone(), Cont::EvalApplyArgs(args, env, k)))
}

fn cont_special_begin(rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match_list!(rest, head: car, tail: cdr => Trampoline::Bounce(car, env.clone(), Cont::EvalExpr(cdr, env, k)))
}
fn cont_special_and(rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match rest.shift() {
        Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalAnd(cdr, env, k))),
        None => Ok(Trampoline::Run(Value::Boolean(true), *k)),
    }
}

fn cont_eval_and(val: Value, rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Boolean(false) => Ok(Trampoline::Run(Value::Boolean(false), *k)),
        _ => match rest.shift() {
            Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalAnd(cdr, env, k))),
            None => Ok(Trampoline::Run(val, *k)),
        },
    }
}

fn cont_special_or(rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match rest.shift() {
        Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalOr(cdr, env, k))),
        None => Ok(Trampoline::Run(Value::Boolean(false), *k)),
    }
}

fn cont_eval_or(val: Value, rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Boolean(false) => match rest.shift() {
            Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalOr(cdr, env, k))),
            None => Ok(Trampoline::Run(Value::Boolean(false), *k)),
        },
        _ => Ok(Trampoline::Run(val, *k)),
    }
}

fn cont_special(f: SpecialForm, rest: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match f {
        SpecialForm::If => cont_special_if(rest, env, k),
        SpecialForm::Define => cont_special_def(rest, env, k),
        SpecialForm::Set => cont_special_set(rest, env, k),
        SpecialForm::Lambda => cont_special_lambda(rest, env, *k),
        SpecialForm::Let => cont_special_let(rest, env, k),
        SpecialForm::Quote => Ok(Trampoline::Run(rest.car()?, *k)),
        SpecialForm::Quasiquote => cont_special_quasiquote(rest, env, k),
        SpecialForm::Eval => Ok(Trampoline::Bounce(rest.car()?, env.clone(), Cont::Eval(env, k))),
        SpecialForm::Apply => cont_special_apply(rest, env, k),
        SpecialForm::Begin => cont_special_begin(rest, env, k),
        SpecialForm::And => cont_special_and(rest, env, k),
        SpecialForm::Or => cont_special_or(rest, env, k),
        SpecialForm::CallCC => Ok(Trampoline::Bounce(rest.car()?, env, Cont::ExecCallCC(k))),
        SpecialForm::DefineSyntaxRule => cont_special_define_syntax_rule(rest, env, *k),
    }
}

fn cont_eval_expr(val: Value, expr: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    info!("C/E eval {}", val);
    match expr.is_empty() {
        true => Ok(Trampoline::Run(val, *k)),
        false => eval(expr, env, k),
    }
}

impl Cont {
    fn run(self, val: Value) -> Result<Trampoline, RuntimeError> {
        info!("Running continuation {:?} with value {:?}", self, val);

        match self {
            Cont::EvalExpr(rest, env, k) => cont_eval_expr(val, rest, env, k),

            Cont::BeginFunc(rest, env, k) => cont_begin_fn(val, rest, env, k),
            Cont::EvalFunc(f, rest, acc, env, k) => cont_eval_fn(f, val, rest, acc, env, k),

            Cont::EvalIf(if_expr, else_expr, env, k) => cont_eval_if(val, if_expr, else_expr, env, *k),
            Cont::EvalDef(name, env, k) => cont_eval_def(name, val, env, *k),
            Cont::EvalSet(name, env, k) => cont_eval_set(name, val, env, *k),
            Cont::EvalLet(name, rest, body, env, k) => cont_eval_let(name, val, rest, body, env, k),
            Cont::ContinueQuasiquote(rest, acc, env, k) => cont_continue_quasiquote(val, rest, acc, env, k),

            Cont::Apply(f, k) => apply(f, val.into_list()?, k),
            Cont::ExecCallCC(k) => apply(val, List::Null.unshift(Value::Continuation(k.clone())), k),

            Cont::EvalAnd(rest, env, k) => cont_eval_and(val, rest, env, k),
            Cont::EvalOr(rest, env, k) => cont_eval_or(val, rest, env, k),

            Cont::Eval(env, k) => Ok(Trampoline::Bounce(val, Env::get_root(env), *k)),
            Cont::EvalApplyArgs(args, env, k) => Ok(Trampoline::Bounce(args, env, Cont::Apply(val, k))),
            Cont::Return => Ok(Trampoline::Land(val)),
        }
    }
}

fn apply(val: Value, args: List, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Continuation(c) => Ok(Trampoline::Run(args.into_list(), *c)),
        Value::Procedure(Function::Native(f)) => Ok(Trampoline::Run(primitive(f, args)?, *k)),
        Value::Procedure(Function::Scheme(arg_names, body, env)) => {
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

fn eval(expr: List, env: Rc<RefCell<Env>>, k: Box<Cont>) -> Result<Trampoline, RuntimeError> {
    match expr.shift() {
        Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Cont::EvalExpr(cdr, env, k))),
        None => runtime_error!("trying to evaluate an empty expression list"),
    }
}

fn expand_macro(expr: Value, substitutions: &HashMap<String, Value>) -> Value {
    match expr {
        Value::Symbol(s) => substitutions.get(&s).cloned().unwrap_or(Value::Symbol(s)),
        Value::List(list) => Value::from_vec(list.into_iter().map(|val| expand_macro(val, substitutions)).collect()),
        _ => expr,
    }
}

static SPECIAL_FORMS: phf::Map<&'static str, SpecialForm> = phf_map! {
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

fn process(exprs: List, env: Rc<RefCell<Env>>) -> Result<Value, RuntimeError> {
    info!("Starting process with expressions: {:?}", exprs);

    if exprs.is_empty() {
        info!("Empty expressions, returning null");
        return Ok(null!());
    }

    let mut result = eval(exprs, env, Box::new(Cont::Return))?;
    info!("Initial evaluation result: {:?}", result);

    loop {
        match result {
            // 常规的 Bounce
            Trampoline::Bounce(val, env, k) => {
                info!("Bounce: Processing value {:?}", val);

                result = match val {
                    Value::List(list) => {
                        info!("Processing list: {:?}", list);
                        match list.shift() {
                            Some((car, cdr)) => {
                                info!("List car: {:?}, cdr: {:?}", car, cdr);
                                Trampoline::Bounce(car, env.clone(), Cont::BeginFunc(cdr, env, Box::new(k)))
                            }
                            None => {
                                info!("Empty list application error");
                                runtime_error!("Can't apply an empty list as a function")
                            }
                        }
                    }
                    Value::Symbol(ref s) => {
                        info!("Processing symbol: {}", s);
                        k.run(
                            // 先从 special form 找，然后是env 中的各种定义
                            SPECIAL_FORMS
                                .get(s.as_ref())
                                .map(|sf| Value::SpecialForm(sf.clone()))
                                .or_else(|| env.borrow().get(s))
                                .ok_or_else(|| RuntimeError {
                                    message: format!("Identifier not found: {}", s),
                                })?,
                        )?
                    }
                    _ => {
                        info!("Processing other value: {:?}", val);
                        k.run(val)? // 其他的所有形式
                    }
                }
            }

            // quasiquote Bounce 模式
            // (unquote X) 会直接转到 Bounce 模式，否则会继续 QuasiBounce
            Trampoline::QuasiquoteBounce(val, env, k) => {
                result = match val {
                    Value::List(list) => match list.shift() {
                        Some((car, cdr)) => match car {
                            Value::Symbol(s) if s == "unquote" => Trampoline::Bounce(cdr.car()?, env, k),
                            _ => Trampoline::QuasiquoteBounce(car, env.clone(), Cont::ContinueQuasiquote(cdr, List::Null, env, Box::new(k))),
                        },
                        None => k.run(null!())?,
                    },
                    _ => k.run(val)?,
                }
            }

            Trampoline::Run(val, k) => result = k.run(val)?, // Run 将 k 应用到 val 上
            Trampoline::Land(val) => return Ok(val),         // Land 会返回给的值; 最早创建，也是最后的 Trampoline 求值
        }
    }
}

#[cfg(test)]
mod test_trampoline {
    use super::*;
    use crate::reader::{lexer, parser};

    // 辅助函数：将 Scheme 代码转换为 List 结构
    fn parse_list(code: &str) -> List {
        let tokens = lexer::tokenize(code).unwrap();
        let nodes = parser::parse(&tokens).unwrap();
        List::from_nodes(&nodes)
    }

    #[test]
    fn test_basic_bounce() {
        // 测试基本的 Bounce: (+ 1 2)
        // 期待执行过程:
        // 1. Bounce(+, env, BeginFunc) - 解析符号 +
        // 2. Run(proc+, BeginFunc) - 获得加法过程
        // 3. Bounce(1, env, EvalFunc) - 评估第一个参数
        // 4. Run(1, EvalFunc) - 得到参数值1
        // 5. Bounce(2, env, EvalFunc) - 评估第二个参数
        // 6. Run(2, EvalFunc) - 得到参数值2
        // 7. Return(3) - 返回最终结果
        let code = parse_list("(+ 1 2)");
        let env = Env::new_root().unwrap();
        let result = process(code, env).unwrap();
        assert_eq!(result, Value::Integer(3));
    }

    #[test]
    fn test_simple_quasiquote() {
        // 测试基本的 quasiquote，不包含 unquote
        // `(1 2 3)
        let code = parse_list("`(1 2 3)");
        let env = Env::new_root().unwrap();
        let result = process(code, env).unwrap();

        let expected = Value::from_vec(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quasiquote_bounce() {
        // 测试 QuasiquoteBounce 处理 unquote: `(2 ,(+ 1 2) 4)
        // 期待执行过程:
        // 1. QuasiquoteBounce(2, env, ContinueQuasiquote) - 处理第一个元素
        // 2. QuasiquoteBounce((unquote (+ 1 2)), env, ContinueQuasiquote) - 遇到 unquote
        // 3. Bounce((+ 1 2), env, k) - 计算 unquote 表达式
        // 4. QuasiquoteBounce(4, env, ContinueQuasiquote) - 处理最后元素
        // 5. Return((2 3 4)) - 返回完整列表
        let code = parse_list("`(2 ,(+ 1 2) 4)");
        let env = Env::new_root().unwrap();
        let result = process(code, env).unwrap();
        assert_eq!(result, Value::from_vec(vec![Value::Integer(2), Value::Integer(3), Value::Integer(4)]));
    }
    #[test]
    fn test_nested_quasiquote() {
        // 测试嵌套的 quasiquote 和 unquote
        // `(1 `(2 ,(+ 1 2) ,(+ 3 4)) 5)
        // 在外层 quasiquote 中，内层的 quasiquote 会被保留，
        // 但内层的 unquote 会被求值
        let code = parse_list("`(1 `(2 ,(+ 1 2) ,(+ 3 4)) 5)");
        let env = Env::new_root().unwrap();
        let result = process(code, env).unwrap();

        // 期待结果: (1 (quasiquote (2 3 7)) 5)
        // 注意: 在外层 quasiquote 中，(+ 1 2) 和 (+ 3 4) 会被求值
        let expected = Value::from_vec(vec![
            Value::Integer(1),
            Value::from_vec(vec![
                Value::Symbol("quasiquote".to_string()),
                Value::from_vec(vec![
                    Value::Integer(2),
                    Value::Integer(3), // (+ 1 2) 的结果
                    Value::Integer(7), // (+ 3 4) 的结果
                ]),
            ]),
            Value::Integer(5),
        ]);

        assert_eq!(result, expected);
    }

    // #[test]
    // fn test_continuation() {
    //     // 测试基本的 continuation
    //     // (call/cc (lambda (k) (k 5)))
    //     let code = parse_list("(call/cc (lambda (k) (k 5)))");
    //     let env = Env::new_root().unwrap();
    //     let result: Value = process(code, env).unwrap();
    //     assert_eq!(result, Value::Integer(5));
    // }

    // #[test]
    // fn test_complex_continuation() {
    //     // 测试更复杂的 continuation 用例
    //     // (+ 1 (call/cc (lambda (k) (k 5))))
    //     let code = parse_list("(+ 1 (call/cc (lambda (k) (k 5))))");
    //     let env = Env::new_root().unwrap();
    //     let result: Value = process(code, env).unwrap();
    //     assert_eq!(result, Value::Integer(6));
    // }

    #[test]
    fn test_return_trampoline() {
        // 测试 Return trampoline: 直接返回值
        // 这是最简单的情况,比如字面量
        let code = parse_list("42");
        let env = Env::new_root().unwrap();
        let result = process(code, env).unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    // #[test]
    // fn test_complex_trampoline_flow() {
    //     // 测试复杂的 trampoline 流程
    //     // (let ((x 1))
    //     //   (+ x (call/cc (lambda (k)
    //     //                   (k (+ x 2))))))
    //     let code = parse_list(
    //         "(let ((x 1))
    //            (+ x (call/cc (lambda (k)
    //                            (k (+ x 2))))))
    //         ",
    //     );
    //     let env = Env::new_root().unwrap();
    //     let result = process(code, env).unwrap();
    //     assert_eq!(result, Value::Integer(4));
    // }
}

#[derive(PartialEq)]
pub struct Env {
    parent: Option<Rc<RefCell<Env>>>,
    values: HashMap<String, Value>,
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
    fn new_root() -> Result<Rc<RefCell<Env>>, RuntimeError> {
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
            env.define(name.into(), Value::Procedure(Function::Native(name)))?;
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
    fn define(&mut self, key: String, value: Value) -> Result<(), RuntimeError> {
        let values = &self.values;
        if values.contains_key(&key) {
            runtime_error!("Duplicate define: {:?}", key)
        }
        self.values.insert(key, value);
        Ok(())
    }

    // Set a variable to a value, at any level in the env, or throw a runtime error if it isn't defined at all
    fn set(&mut self, key: String, value: Value) -> Result<(), RuntimeError> {
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

    fn get(&self, key: &String) -> Option<Value> {
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

    fn get_root(env_ref: Rc<RefCell<Env>>) -> Rc<RefCell<Env>> {
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

#[cfg(test)]
fn exec(list: List) -> Result<Value, RuntimeError> { process(list, Env::new_root()?) }

#[test]
fn test_add1() {
    // runTest (+ 1 2) => 3
    let i = vec![Value::from_vec(vec![
        Value::Symbol("+".to_string()),
        Value::Integer(1),
        Value::Integer(2),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(3));
}

#[test]
fn test_add2() {
    // runTest (+ (+ 1 2) (+ 3 4)) => 10
    let i = vec![Value::from_vec(vec![
        Value::Symbol("+".to_string()),
        Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(1), Value::Integer(2)]),
        Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(3), Value::Integer(4)]),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(10));
}

#[test]
fn test_add3() {
    // runTest (+ (+ 1 2) (+ (+ 3 5 6) 4)) => 21
    let i = vec![Value::from_vec(vec![
        Value::Symbol("+".to_string()),
        Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(1), Value::Integer(2)]),
        Value::from_vec(vec![
            Value::Symbol("+".to_string()),
            Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(3), Value::Integer(5), Value::Integer(6)]),
            Value::Integer(4),
        ]),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(21));
}

#[test]
fn test_subtract1() {
    // runTest (- 3 2) => 1
    let i = vec![Value::from_vec(vec![
        Value::Symbol("-".to_string()),
        Value::Integer(3),
        Value::Integer(2),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(1));
}

#[test]
fn test_if1() {
    // runTest (if (> 1 2) 3 4) => 4
    let i = vec![Value::from_vec(vec![
        Value::Symbol("if".to_string()),
        Value::from_vec(vec![Value::Symbol(">".to_string()), Value::Integer(1), Value::Integer(2)]),
        Value::Integer(3),
        Value::Integer(4),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(4));
}

#[test]
fn test_if2() {
    // runTest (if (> 2 3) (error 4) (error 5)) => null
    let i = vec![Value::from_vec(vec![
        Value::Symbol("if".to_string()),
        Value::from_vec(vec![Value::Symbol(">".to_string()), Value::Integer(2), Value::Integer(3)]),
        Value::from_vec(vec![Value::Symbol("error".to_string()), Value::Integer(4)]),
        Value::from_vec(vec![Value::Symbol("error".to_string()), Value::Integer(5)]),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap_err().to_string(), "RuntimeError: 5");
}

#[test]
fn test_if3() {
    // runTest (if ((if (> 5 4) > <) (+ 1 2) 2) (+ 5 7 8) (+ 9 10 11)) => 20
    let i = vec![Value::from_vec(vec![
        Value::Symbol("if".to_string()),
        Value::from_vec(vec![
            Value::from_vec(vec![
                Value::Symbol("if".to_string()),
                Value::from_vec(vec![Value::Symbol(">".to_string()), Value::Integer(5), Value::Integer(4)]),
                Value::Symbol(">".to_string()),
                Value::Symbol("<".to_string()),
            ]),
            Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(1), Value::Integer(2)]),
            Value::Integer(2),
        ]),
        Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(5), Value::Integer(7), Value::Integer(8)]),
        Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(9), Value::Integer(10), Value::Integer(11)]),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(20));
}

#[test]
fn test_if4() {
    // runTest (if 0 3 4) => 3
    let i = vec![Value::from_vec(vec![
        Value::Symbol("if".to_string()),
        Value::Integer(0),
        Value::Integer(3),
        Value::Integer(4),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(3));
}

#[test]
fn test_and1() {
    // runTest (and) => #t
    let i = vec![Value::from_vec(vec![Value::Symbol("and".to_string())])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Boolean(true));
}

#[test]
fn test_and2() {
    // runTest (and #f) => #f
    let i = vec![Value::from_vec(vec![Value::Symbol("and".to_string()), Value::Boolean(false)])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Boolean(false));
}

#[test]
fn test_and3() {
    // runTest (and #f #t #f) => #f
    let i = vec![Value::from_vec(vec![
        Value::Symbol("and".to_string()),
        Value::Boolean(false),
        Value::Boolean(true),
        Value::Boolean(false),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Boolean(false));
}

#[test]
fn test_and4() {
    // runTest (and 0 1) => 1
    let i = vec![Value::from_vec(vec![
        Value::Symbol("and".to_string()),
        Value::Integer(0),
        Value::Integer(1),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(1));
}

#[test]
fn test_and5() {
    // runTest (and #f (error 2)) => #f
    let i = vec![Value::from_vec(vec![
        Value::Symbol("and".to_string()),
        Value::Boolean(false),
        Value::from_vec(vec![Value::Symbol("error".to_string()), Value::Integer(2)]),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Boolean(false));
}

#[test]
fn test_or1() {
    // runTest (or) => #f
    let i = vec![Value::from_vec(vec![Value::Symbol("or".to_string())])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Boolean(false));
}

#[test]
fn test_or2() {
    // runTest (or #f) => #f
    let i = vec![Value::from_vec(vec![Value::Symbol("or".to_string()), Value::Boolean(false)])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Boolean(false));
}

#[test]
fn test_or3() {
    // runTest (or #f #t #f) => #t
    let i = vec![Value::from_vec(vec![
        Value::Symbol("or".to_string()),
        Value::Boolean(false),
        Value::Boolean(true),
        Value::Boolean(false),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Boolean(true));
}

#[test]
fn test_or4() {
    // runTest (or 0 1) => 0
    let i = vec![Value::from_vec(vec![
        Value::Symbol("or".to_string()),
        Value::Integer(0),
        Value::Integer(1),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(0));
}

#[test]
fn test_or5() {
    // runTest (or #t (error 2)) => #t
    let i = vec![Value::from_vec(vec![
        Value::Symbol("or".to_string()),
        Value::Boolean(true),
        Value::from_vec(vec![Value::Symbol("error".to_string()), Value::Integer(2)]),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Boolean(true));
}

#[test]
fn test_multiple_statements() {
    // runTest (+ 1 2) (+ 3 4) => 7
    let i = vec![
        Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(1), Value::Integer(2)]),
        Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(3), Value::Integer(4)]),
    ];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(7));
}

#[test]
fn test_list() {
    // runTest (list 1 2 3) => '(1 2 3)
    let i = vec![Value::from_vec(vec![
        Value::Symbol("list".to_string()),
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::from_vec(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]));
}

#[test]
fn test_cons() {
    // runTest (cons 1 (list 2 3)) => '(1 2 3)
    let i = vec![Value::from_vec(vec![
        Value::Symbol("cons".to_string()),
        Value::Integer(1),
        Value::from_vec(vec![Value::Symbol("list".to_string()), Value::Integer(2), Value::Integer(3)]),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::from_vec(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]));
}

#[test]
fn test_define() {
    // runTest (define x 2) (+ x x) => 4
    let i = vec![
        Value::from_vec(vec![Value::Symbol("define".to_string()), Value::Symbol("x".to_string()), Value::Integer(2)]),
        Value::from_vec(vec![
            Value::Symbol("+".to_string()),
            Value::Symbol("x".to_string()),
            Value::Symbol("x".to_string()),
        ]),
    ];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(4));
}

#[test]
fn test_set() {
    // runTest (define x 2) (set! x 3) (+ x x) => 6
    let i = vec![
        Value::from_vec(vec![Value::Symbol("define".to_string()), Value::Symbol("x".to_string()), Value::Integer(2)]),
        Value::from_vec(vec![Value::Symbol("set!".to_string()), Value::Symbol("x".to_string()), Value::Integer(3)]),
        Value::from_vec(vec![
            Value::Symbol("+".to_string()),
            Value::Symbol("x".to_string()),
            Value::Symbol("x".to_string()),
        ]),
    ];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(6));
}

#[test]
fn test_lambda() {
    // runTest ((lambda (x) (+ x 2)) 3) => 5
    let i = vec![Value::from_vec(vec![
        Value::from_vec(vec![
            Value::Symbol("lambda".to_string()),
            Value::from_vec(vec![Value::Symbol("x".to_string())]),
            Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Symbol("x".to_string()), Value::Integer(2)]),
        ]),
        Value::Integer(3),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(5));
}

#[test]
fn test_lambda_symbol() {
    // runTest ((λ (x) (+ x 2)) 3) => 5
    let i = vec![Value::from_vec(vec![
        Value::from_vec(vec![
            Value::Symbol("λ".to_string()),
            Value::from_vec(vec![Value::Symbol("x".to_string())]),
            Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Symbol("x".to_string()), Value::Integer(2)]),
        ]),
        Value::Integer(3),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(5));
}

#[test]
fn test_define_func() {
    // runTest (define (f x) (+ x 2)) (f 3) => 5
    let i = vec![
        Value::from_vec(vec![
            Value::Symbol("define".to_string()),
            Value::from_vec(vec![Value::Symbol("f".to_string()), Value::Symbol("x".to_string())]),
            Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Symbol("x".to_string()), Value::Integer(2)]),
        ]),
        Value::from_vec(vec![Value::Symbol("f".to_string()), Value::Integer(3)]),
    ];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(5));
}

#[test]
fn test_define_func2() {
    // runTest (define (noop) (+ 0 0)) (define (f x) (noop) (+ x 2)) ((lambda () (f 3))) => 5
    let i = vec![
        Value::from_vec(vec![
            Value::Symbol("define".to_string()),
            Value::from_vec(vec![Value::Symbol("noop".to_string())]),
            Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(0), Value::Integer(0)]),
        ]),
        Value::from_vec(vec![
            Value::Symbol("define".to_string()),
            Value::from_vec(vec![Value::Symbol("f".to_string()), Value::Symbol("x".to_string())]),
            Value::from_vec(vec![Value::Symbol("noop".to_string())]),
            Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Symbol("x".to_string()), Value::Integer(2)]),
        ]),
        Value::from_vec(vec![Value::from_vec(vec![
            Value::Symbol("lambda".to_string()),
            null!(),
            Value::from_vec(vec![Value::Symbol("f".to_string()), Value::Integer(3)]),
        ])]),
    ];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(5));
}

#[test]
fn test_native_fn_as_value() {
    // runTest + => #<procedure:+>
    let i = vec![Value::Symbol("+".to_string())];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Procedure(Function::Native("+")));
}

#[test]
fn test_dynamic_native_fn() {
    // runTest ((if (> 3 2) + -) 4 3) => 7
    let i = vec![Value::from_vec(vec![
        Value::from_vec(vec![
            Value::Symbol("if".to_string()),
            Value::from_vec(vec![Value::Symbol(">".to_string()), Value::Integer(3), Value::Integer(2)]),
            Value::Symbol("+".to_string()),
            Value::Symbol("-".to_string()),
        ]),
        Value::Integer(4),
        Value::Integer(3),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(7));
}

#[test]
fn test_let_bindings() {
    // runTest (let ((x 3)) (+ x 1)) => 4
    let i = vec![Value::from_vec(vec![
        Value::Symbol("let".to_string()),
        Value::from_vec(vec![Value::from_vec(vec![Value::Symbol("x".to_string()), Value::Integer(3)])]),
        Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Symbol("x".to_string()), Value::Integer(1)]),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(4));
}

#[test]
fn test_quoting() {
    // runTest (quote (1 2)) => (1 2)
    let code = vec![Value::from_vec(vec![
        Value::Symbol("quote".to_string()),
        Value::from_vec(vec![Value::Integer(1), Value::Integer(2)]),
    ])];
    assert_eq!(exec(List::from_vec(code)).unwrap(), Value::from_vec(vec![Value::Integer(1), Value::Integer(2)]));
}

#[test]
fn test_quasiquoting() {
    // runTest (quasiquote (2 (unquote (+ 1 2)) 4)) => (2 3 4)
    let code = vec![Value::from_vec(vec![
        Value::Symbol("quasiquote".to_string()),
        Value::from_vec(vec![
            Value::Integer(2),
            Value::from_vec(vec![
                Value::Symbol("unquote".to_string()),
                Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(1), Value::Integer(2)]),
            ]),
            Value::Integer(4),
        ]),
    ])];
    assert_eq!(exec(List::from_vec(code)).unwrap(), Value::from_vec(vec![Value::Integer(2), Value::Integer(3), Value::Integer(4)]));
}

// #[test]
// fn test_cps_unquote_slicing() {
//     // runTest (quasiquote (1 (unquote-slicing (list 2 3)) 4)) => (1 2 3 4)
//     let i = vec![Value::from_vec(vec![
//         Value::Symbol("quasiquote".to_string()),
//         Value::from_vec(vec![
//             Value::Integer(1),
//             Value::from_vec(vec![
//                 Value::Symbol("unquote-slicing".to_string()),
//                 Value::from_vec(vec![Value::Symbol("list".to_string()), Value::Integer(2), Value::Integer(3)]),
//             ]),
//             Value::Integer(4),
//         ]),
//     ])];
//     assert_eq!(exec(List::from_vec(i)).unwrap(), Value::from_vec(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3), Value::Integer(4)]));
// }

#[test]
fn test_eval() {
    // runTest (eval (quote (+ 1 2))) => 3
    let i = vec![Value::from_vec(vec![
        Value::Symbol("eval".to_string()),
        Value::from_vec(vec![
            Value::Symbol("quote".to_string()),
            Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(1), Value::Integer(2)]),
        ]),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(3));
}

#[test]
fn test_eval2() {
    // runTest (define (foo x) (eval (quote (+ 1 2))) x) (foo 5) => 5
    let i = vec![
        Value::from_vec(vec![
            Value::Symbol("define".to_string()),
            Value::from_vec(vec![Value::Symbol("foo".to_string()), Value::Symbol("x".to_string())]),
            Value::from_vec(vec![
                Value::Symbol("eval".to_string()),
                Value::from_vec(vec![
                    Value::Symbol("quote".to_string()),
                    Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(1), Value::Integer(2)]),
                ]),
            ]),
            Value::Symbol("x".to_string()),
        ]),
        Value::from_vec(vec![Value::Symbol("foo".to_string()), Value::Integer(5)]),
    ];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(5));
}

#[test]
fn test_apply() {
    // runTest (apply + (quote (1 2 3))) => 6
    let i = vec![Value::from_vec(vec![
        Value::Symbol("apply".to_string()),
        Value::Symbol("+".to_string()),
        Value::from_vec(vec![
            Value::Symbol("quote".to_string()),
            Value::from_vec(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]),
        ]),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(6));
}

#[test]
fn test_begin() {
    // runTest (define x 1) (begin (set! x 5) (set! x (+ x 2)) x) => 7
    let i = vec![
        Value::from_vec(vec![Value::Symbol("define".to_string()), Value::Symbol("x".to_string()), Value::Integer(1)]),
        Value::from_vec(vec![
            Value::Symbol("begin".to_string()),
            Value::from_vec(vec![Value::Symbol("set!".to_string()), Value::Symbol("x".to_string()), Value::Integer(5)]),
            Value::from_vec(vec![
                Value::Symbol("set!".to_string()),
                Value::Symbol("x".to_string()),
                Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Symbol("x".to_string()), Value::Integer(2)]),
            ]),
            Value::Symbol("x".to_string()),
        ]),
    ];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(7));
}

#[test]
fn test_callcc() {
    // runTest
    //   (define x 0)
    //   (define (+x n) (set! x (+ x n)))
    //   (define (foo k) (+x 2) (k) (+x 4))
    //   ((lambda ()
    //      (+x 1)
    //      (call/cc foo)
    //      (+x 8)))
    //   x
    // => 11
    let i = vec![
        Value::from_vec(vec![Value::Symbol("define".to_string()), Value::Symbol("x".to_string()), Value::Integer(0)]),
        Value::from_vec(vec![
            Value::Symbol("define".to_string()),
            Value::from_vec(vec![Value::Symbol("+x".to_string()), Value::Symbol("n".to_string())]),
            Value::from_vec(vec![
                Value::Symbol("set!".to_string()),
                Value::Symbol("x".to_string()),
                Value::from_vec(vec![
                    Value::Symbol("+".to_string()),
                    Value::Symbol("x".to_string()),
                    Value::Symbol("n".to_string()),
                ]),
            ]),
        ]),
        Value::from_vec(vec![
            Value::Symbol("define".to_string()),
            Value::from_vec(vec![Value::Symbol("foo".to_string()), Value::Symbol("k".to_string())]),
            Value::from_vec(vec![Value::Symbol("+x".to_string()), Value::Integer(2)]),
            Value::from_vec(vec![Value::Symbol("k".to_string())]),
            Value::from_vec(vec![Value::Symbol("+x".to_string()), Value::Integer(4)]),
        ]),
        Value::from_vec(vec![Value::from_vec(vec![
            Value::Symbol("lambda".to_string()),
            null!(),
            Value::from_vec(vec![Value::Symbol("+x".to_string()), Value::Integer(1)]),
            Value::from_vec(vec![Value::Symbol("call/cc".to_string()), Value::Symbol("foo".to_string())]),
            Value::from_vec(vec![Value::Symbol("+x".to_string()), Value::Integer(8)]),
        ])]),
        Value::Symbol("x".to_string()),
    ];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(11));
}

#[test]
fn test_macros() {
    // runTest (define-syntax-rule (incr x) (set! x (+ x 1))) (define a 1) (incr a) a => 2
    let i = vec![
        Value::from_vec(vec![
            Value::Symbol("define-syntax-rule".to_string()),
            Value::from_vec(vec![Value::Symbol("incr".to_string()), Value::Symbol("x".to_string())]),
            Value::from_vec(vec![
                Value::Symbol("set!".to_string()),
                Value::Symbol("x".to_string()),
                Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Symbol("x".to_string()), Value::Integer(1)]),
            ]),
        ]),
        Value::from_vec(vec![Value::Symbol("define".to_string()), Value::Symbol("a".to_string()), Value::Integer(1)]),
        Value::from_vec(vec![Value::Symbol("incr".to_string()), Value::Symbol("a".to_string())]),
        Value::Symbol("a".to_string()),
    ];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Integer(2));
}

#[test]
fn test_list_iter() {
    let l = List::Cell(
        Box::new(Value::Integer(1)),
        Box::new(List::Cell(Box::new(Value::Integer(2)), Box::new(List::Cell(Box::new(Value::Integer(3)), Box::new(List::Null))))),
    );
    let mut x = 0;
    for i in l {
        x += 1;
        assert_eq!(i, Value::Integer(x));
    }
    assert_eq!(x, 3);
}

#[test]
fn test_list_to_string() {
    let l = List::Cell(
        Box::new(Value::Integer(1)),
        Box::new(List::Cell(Box::new(Value::Integer(2)), Box::new(List::Cell(Box::new(Value::Integer(3)), Box::new(List::Null))))),
    );
    assert_eq!(l.to_string(), "(1 2 3)");
}
