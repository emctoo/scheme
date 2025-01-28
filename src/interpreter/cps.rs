use crate::reader::parser::*;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::vec;

pub fn new() -> Result<Interpreter, RuntimeError> {
    Interpreter::new()
}

#[derive(Clone)]
pub struct Interpreter {
    root: Rc<RefCell<Environment>>,
}

impl Interpreter {
    pub fn new() -> Result<Interpreter, RuntimeError> {
        let env = Environment::new_root()?;
        Ok(Interpreter { root: env })
    }

    pub fn run(&self, nodes: &[Node]) -> Result<Value, RuntimeError> {
        process(List::from_nodes(nodes), self.root.clone())
    }
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

#[derive(PartialEq, Clone)]
pub enum Value {
    Symbol(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    List(List),
    Procedure(Function),
    SpecialForm(SpecialForm),
    Macro(Vec<String>, Box<Value>),
    Continuation(Box<Continuation>),
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
    fn from_vec(vec: Vec<Value>) -> Value {
        List::from_vec(vec).into_list()
    }

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

    fn as_symbol(self) -> Result<String, RuntimeError> {
        match self {
            Value::Symbol(s) => Ok(s),
            _ => runtime_error!("Expected a symbol value: {:?}", self),
        }
    }

    fn as_integer(self) -> Result<i64, RuntimeError> {
        match self {
            Value::Integer(i) => Ok(i),
            _ => runtime_error!("Expected an integer value: {:?}", self),
        }
    }

    // fn as_boolean(self) -> Result<bool, RuntimeError> {
    //     match self {
    //         Value::Boolean(b) => Ok(b),
    //         _ => runtime_error!("Expected a boolean value: {:?}", self)
    //     }
    // }

    // fn as_string(self) -> Result<String, RuntimeError> {
    //     match self {
    //         Value::String(s) => Ok(s),
    //         _ => runtime_error!("Expected a string value: {:?}", self)
    //     }
    // }

    fn as_list(self) -> Result<List, RuntimeError> {
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
    Scheme(Vec<String>, List, Rc<RefCell<Environment>>),
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RuntimeError: {}", self.message)
    }
}

#[derive(PartialEq, Clone)]
pub enum List {
    Cell(Box<Value>, Box<List>),
    Null,
}

// null == empty list
macro_rules! null {
    () => {
        List::Null.into_list()
    };
}

impl List {
    fn from_vec(mut vec: Vec<Value>) -> List {
        if !vec.is_empty() {
            let mut out = List::Null;
            while let Some(v) = vec.pop() {
                out = List::Cell(Box::new(v), Box::new(out));
            }
            out
        } else {
            List::Null
        }
    }

    fn from_nodes(nodes: &[Node]) -> List {
        let vec = nodes.iter().map(Value::from_node).collect();
        List::from_vec(vec)
    }

    fn is_empty(&self) -> bool {
        self == &List::Null
    }

    /// Null => None
    /// List => Some((car cdr)
    fn shift(self) -> Option<(Value, List)> {
        match self {
            List::Cell(car, cdr) => Some((*car, *cdr)),
            List::Null => None,
        }
    }

    /// car => (car self)
    fn unshift(self, car: Value) -> List {
        List::Cell(Box::new(car), Box::new(self))
    }

    /// list length
    fn len(&self) -> usize {
        match self {
            List::Cell(_, ref cdr) => 1 + cdr.len(),
            List::Null => 0,
        }
    }

    /// (car cdr) -> car
    fn unpack1(self) -> Result<Value, RuntimeError> {
        let (car, cdr) = shift_or_error!(self, "Expected list of length 1, but was empty");
        if !cdr.is_empty() {
            runtime_error!("Expected list of length 1, but it had more elements")
        }
        Ok(car)
    }

    /// (car (cadr '())) -> (car cadr)
    fn unpack2(self) -> Result<(Value, Value), RuntimeError> {
        let (car, cdr) = shift_or_error!(self, "Expected list of length 2, but was empty");
        let (cadr, cddr) = shift_or_error!(cdr, "Expected list of length 2, but was length 1");
        if !cddr.is_empty() {
            runtime_error!("Expected list of length 2, but it had more elements")
        }
        Ok((car, cadr))
    }

    /// (car (cadr (caddr '()))) -> (car cadr caddr)
    fn unpack3(self) -> Result<(Value, Value, Value), RuntimeError> {
        let (car, cdr) = shift_or_error!(self, "Expected list of length 3, but was empty");
        let (cadr, cddr) = shift_or_error!(cdr, "Expected list of length 3, but was length 1");
        let (caddr, cdddr) = shift_or_error!(cddr, "Expected list of length 3, but was length 2");
        if !cdddr.is_empty() {
            runtime_error!("Expected list of length 3, but it had more elements")
        }
        Ok((car, cadr, caddr))
    }

    fn reverse(self) -> List {
        let mut out = List::Null;
        for val in self {
            out = out.unshift(val)
        }
        out
    }

    fn into_list(self) -> Value {
        Value::List(self)
    }

    fn into_vec(self) -> Vec<Value> {
        self.into_iter().collect()
    }

    fn into_iter(self) -> ListIterator {
        ListIterator(self)
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

impl IntoIterator for List {
    type Item = Value;
    type IntoIter = vec::IntoIter<Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
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

#[derive(PartialEq, Clone, Debug)]
pub enum Continuation {
    EvalExpr(List, Rc<RefCell<Environment>>, Box<Continuation>),
    BeginFunc(List, Rc<RefCell<Environment>>, Box<Continuation>),
    EvalIf(Value, Value, Rc<RefCell<Environment>>, Box<Continuation>),
    EvalDef(String, Rc<RefCell<Environment>>, Box<Continuation>),
    EvalSet(String, Rc<RefCell<Environment>>, Box<Continuation>),
    EvalFunc(Value, List, List, Rc<RefCell<Environment>>, Box<Continuation>),
    EvalLet(String, List, List, Rc<RefCell<Environment>>, Box<Continuation>),
    ContinueQuasiquoting(List, List, Rc<RefCell<Environment>>, Box<Continuation>),
    Eval(Rc<RefCell<Environment>>, Box<Continuation>),
    EvalApplyArgs(Value, Rc<RefCell<Environment>>, Box<Continuation>),
    Apply(Value, Box<Continuation>),
    EvalAnd(List, Rc<RefCell<Environment>>, Box<Continuation>),
    EvalOr(List, Rc<RefCell<Environment>>, Box<Continuation>),
    ExecCallCC(Box<Continuation>),
    Return,
}

pub enum Trampoline {
    Bounce(Value, Rc<RefCell<Environment>>, Continuation),
    QuasiBounce(Value, Rc<RefCell<Environment>>, Continuation),
    Run(Value, Continuation),
    Land(Value),
}

fn cont_special_def(rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    let (car, cdr) = shift_or_error!(rest, "Must provide at least two arguments to define");
    match car {
        Value::Symbol(name) => {
            let val = cdr.unpack1()?;
            Ok(Trampoline::Bounce(val, env.clone(), Continuation::EvalDef(name, env, k)))
        }
        Value::List(list) => {
            let (caar, cdar) = shift_or_error!(list, "Must provide at least two params in first argument of define");
            let name = caar.as_symbol()?;

            let arg_names = cdar.into_iter().map(|v| v.as_symbol()).collect::<Result<Vec<String>, RuntimeError>>()?;
            let body = cdr;
            let f = Function::Scheme(arg_names, body, env.clone());

            env.borrow_mut().define(name, Value::Procedure(f))?;
            Ok(Trampoline::Run(null!(), *k))
        }
        _ => runtime_error!("Bad argument to define: {:?}", car),
    }
}

fn cont_special_lambda(rest: List, env: Rc<RefCell<Environment>>, k: Continuation) -> Result<Trampoline, RuntimeError> {
    let (arg_defns_raw, body) = shift_or_error!(rest, "Must provide at least two arguments to lambda");
    let arg_defns = arg_defns_raw.as_list()?;

    let arg_names = arg_defns
        .into_iter()
        .map(|v| v.as_symbol())
        .collect::<Result<Vec<String>, RuntimeError>>()?;

    let f = Function::Scheme(arg_names, body, env);
    Ok(Trampoline::Run(Value::Procedure(f), k))
}

fn cont_special_let(rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    let (arg_defns_raw, body) = shift_or_error!(rest, "Must provide at least two arguments to let");
    let arg_defns = arg_defns_raw.as_list()?;

    // Create a new, child environment for the procedure and define the arguments as local variables
    let proc_env = Environment::new_child(env.clone());

    // Iterate through the provided arguments, defining them
    if !arg_defns.is_empty() {
        let (first_defn, rest_defns) = shift_or_error!(arg_defns, "Error in let definiton");
        let (defn_key, defn_val) = first_defn.as_list()?.unpack2()?;
        let name = defn_key.as_symbol()?;
        Ok(Trampoline::Bounce(defn_val, env, Continuation::EvalLet(name, rest_defns, body, proc_env, k)))
    } else {
        // Let bindings were empty, just execute the body directly
        eval(body, env, k)
    }
}

fn cont_special_define_syntax_rule(rest: List, env: Rc<RefCell<Environment>>, k: Continuation) -> Result<Trampoline, RuntimeError> {
    let (defn, body) = rest.unpack2()?;

    let (name, arg_names_raw) = match defn.as_list()?.shift() {
        Some((car, cdr)) => (car.as_symbol()?, cdr),
        None => runtime_error!("Must supply at least two params to first argument in define-syntax-rule"),
    };

    let arg_names = arg_names_raw
        .into_iter()
        .map(|v| v.as_symbol())
        .collect::<Result<Vec<String>, RuntimeError>>()?;

    let m = Value::Macro(arg_names, Box::new(body));
    let _ = env.borrow_mut().define(name, m);
    Ok(Trampoline::Run(null!(), k))
}
fn cont_special_macro(
    rest: List, arg_names: Vec<String>, body: Value, env: Rc<RefCell<Environment>>, k: Continuation,
) -> Result<Trampoline, RuntimeError> {
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

fn cont_special_if(rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    let (condition, if_expr, else_expr) = rest.unpack3()?;
    Ok(Trampoline::Bounce(condition, env.clone(), Continuation::EvalIf(if_expr, else_expr, env, k)))
}

fn cont_special_set(rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    let (name, val) = rest.unpack2()?;
    Ok(Trampoline::Bounce(val, env.clone(), Continuation::EvalSet(name.as_symbol()?, env, k)))
}

fn cont_special_quasiquote(rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    let expr = rest.unpack1()?;
    match expr {
        Value::List(list) => match list.shift() {
            Some((car, cdr)) => Ok(Trampoline::QuasiBounce(car, env.clone(), Continuation::ContinueQuasiquoting(cdr, List::Null, env, k))),
            None => Ok(Trampoline::Run(null!(), *k)),
        },
        _ => Ok(Trampoline::Run(expr, *k)),
    }
}

fn cont_special_apply(rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    let (f, args) = rest.unpack2()?;
    Ok(Trampoline::Bounce(f, env.clone(), Continuation::EvalApplyArgs(args, env, k)))
}
fn cont_special_begin(rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    match rest.shift() {
        Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Continuation::EvalExpr(cdr, env, k))),
        None => runtime_error!("Must provide at least one argument to a begin statement"),
    }
}
fn cont_special_and(rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    match rest.shift() {
        Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Continuation::EvalAnd(cdr, env, k))),
        None => Ok(Trampoline::Run(Value::Boolean(true), *k)),
    }
}
fn cont_special_or(rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    match rest.shift() {
        Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Continuation::EvalOr(cdr, env, k))),
        None => Ok(Trampoline::Run(Value::Boolean(false), *k)),
    }
}

fn cont_special(f: SpecialForm, rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError>  {
    match f {
        SpecialForm::If => cont_special_if(rest, env, k),
        SpecialForm::Define => cont_special_def(rest, env, k),
        SpecialForm::Set => cont_special_set(rest, env, k),
        SpecialForm::Lambda => cont_special_lambda(rest, env, *k),
        SpecialForm::Let => cont_special_let(rest, env, k),
        SpecialForm::Quote => Ok(Trampoline::Run(rest.unpack1()?, *k)),
        SpecialForm::Quasiquote => cont_special_quasiquote(rest, env, k),
        SpecialForm::Eval => Ok(Trampoline::Bounce(rest.unpack1()?, env.clone(), Continuation::Eval(env, k))),
        SpecialForm::Apply => cont_special_apply(rest, env, k),
        SpecialForm::Begin => cont_special_begin(rest, env, k),
        SpecialForm::And => cont_special_and(rest, env, k),
        SpecialForm::Or => cont_special_or(rest, env, k),
        SpecialForm::CallCC => Ok(Trampoline::Bounce(rest.unpack1()?, env, Continuation::ExecCallCC(k))),
        SpecialForm::DefineSyntaxRule => cont_special_define_syntax_rule(rest, env, *k),
    }
}

fn cont_begin_fn(val: Value, rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Macro(arg_names, body) => cont_special_macro(rest, arg_names, *body, env, *k),
        Value::SpecialForm(f) => cont_special(f, rest, env, k),
        _ => match rest.shift() {
            Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Continuation::EvalFunc(val, cdr, List::Null, env, k))),
            None => apply(val, List::Null, k),
        },
    }
}

fn cont_eval_expr(val: Value, rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    if !rest.is_empty() {
        eval(rest, env, k)
    } else {
        Ok(Trampoline::Run(val, *k))
    }
}

fn cont_eval_fn(
    f: Value, val: Value, rest: List, acc: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>,
) -> Result<Trampoline, RuntimeError> {
    let acc2 = acc.unshift(val);
    match rest.shift() {
        Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Continuation::EvalFunc(f, cdr, acc2, env, k))),
        None => apply(f, acc2.reverse(), k),
    }
}

fn cont_eval_if(if_expr: Value, else_expr: Value, val: Value, env: Rc<RefCell<Environment>>, k: Continuation) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Boolean(false) => Ok(Trampoline::Bounce(else_expr, env, k)),
        _ => Ok(Trampoline::Bounce(if_expr, env, k)),
    }
}

fn cont_eval_def(name: String, val: Value, env: Rc<RefCell<Environment>>, k: Continuation) -> Result<Trampoline, RuntimeError> {
    env.borrow_mut().define(name, val)?;
    Ok(Trampoline::Run(null!(), k))
}

fn cont_eval_set(name: String, val: Value, env: Rc<RefCell<Environment>>, k: Continuation) -> Result<Trampoline, RuntimeError> {
    env.borrow_mut().set(name, val)?;
    Ok(Trampoline::Run(null!(), k))
}

fn cont_eval_let(
    name: String, val: Value, rest: List, body: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>,
) -> Result<Trampoline, RuntimeError> {
    // Define variable in let scope
    env.borrow_mut().define(name, val)?;
    match rest.shift() {
        Some((next_defn, rest_defns)) => {
            let (defn_key, defn_val) = next_defn.as_list()?.unpack2()?;
            let name = defn_key.as_symbol()?;
            Ok(Trampoline::Bounce(defn_val, env.clone(), Continuation::EvalLet(name, rest_defns, body, env, k)))
        }
        None => {
            let inner_env = Environment::new_child(env);
            eval(body, inner_env, k)
        }
    }
}

fn cont_continue_quasiquoting(
    val: Value, rest: List, acc: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>,
) -> Result<Trampoline, RuntimeError> {
    let acc2 = acc.unshift(val);
    match rest.shift() {
        Some((car, cdr)) => Ok(Trampoline::QuasiBounce(car, env.clone(), Continuation::ContinueQuasiquoting(cdr, acc2, env, k))),
        None => Ok(Trampoline::Run(acc2.reverse().into_list(), *k)),
    }
}

fn cont_eval_and(val: Value, rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Boolean(false) => Ok(Trampoline::Run(Value::Boolean(false), *k)),
        _ => match rest.shift() {
            Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Continuation::EvalAnd(cdr, env, k))),
            None => Ok(Trampoline::Run(val, *k)),
        },
    }
}

fn cont_eval_or(val: Value, rest: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Boolean(false) => match rest.shift() {
            Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Continuation::EvalOr(cdr, env, k))),
            None => Ok(Trampoline::Run(Value::Boolean(false), *k)),
        },
        _ => Ok(Trampoline::Run(val, *k)),
    }
}
impl Continuation {
    fn run(self, val: Value) -> Result<Trampoline, RuntimeError> {
        match self {
            Continuation::EvalExpr(rest, env, k) => cont_eval_expr(val, rest, env, k),
            Continuation::BeginFunc(rest, env, k) => cont_begin_fn(val, rest, env, k),
            Continuation::EvalFunc(f, rest, acc, env, k) => cont_eval_fn(f, val, rest, acc, env, k),
            Continuation::EvalIf(if_expr, else_expr, env, k) => cont_eval_if(if_expr, else_expr, val, env, *k),
            Continuation::EvalDef(name, env, k) => cont_eval_def(name, val, env, *k),
            Continuation::EvalSet(name, env, k) => cont_eval_set(name, val, env, *k),
            Continuation::EvalLet(name, rest, body, env, k) => cont_eval_let(name, val, rest, body, env, k),
            Continuation::ContinueQuasiquoting(rest, acc, env, k) => cont_continue_quasiquoting(val, rest, acc, env, k),

            Continuation::Apply(f, k) => apply(f, val.as_list()?, k),
            Continuation::ExecCallCC(k) => apply(val, List::Null.unshift(Value::Continuation(k.clone())), k),

            Continuation::EvalAnd(rest, env, k) => cont_eval_and(val, rest, env, k),
            Continuation::EvalOr(rest, env, k) => cont_eval_or(val, rest, env, k),

            Continuation::Eval(env, k) => Ok(Trampoline::Bounce(val, Environment::get_root(env), *k)),
            Continuation::EvalApplyArgs(args, env, k) => Ok(Trampoline::Bounce(args, env, Continuation::Apply(val, k))),
            Continuation::Return => Ok(Trampoline::Land(val)),
        }
    }
}

fn apply_function(f: Function, args: List, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    match f {
        Function::Native(native_func) => Ok(Trampoline::Run(primitive(native_func, args)?, *k)),
        Function::Scheme(arg_names, body, env) => {
            if arg_names.len() != args.len() {
                runtime_error!("Must supply exactly {} arguments to function: {:?}", arg_names.len(), args);
            }

            // Create a new, child environment for the procedure and define the arguments as local variables
            let proc_env = Environment::new_child(env);
            arg_names
                .into_iter()
                .zip(args)
                .try_for_each(|(name, value)| proc_env.borrow_mut().define(name, value))?;

            // Evaluate procedure body with new environment with procedure environment as parent
            eval(body, Environment::new_child(proc_env), k)
        }
    }
}

fn apply(val: Value, args: List, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    match val {
        Value::Continuation(c) => Ok(Trampoline::Run(args.into_list(), *c)),
        Value::Procedure(func) => apply_function(func, args, k),
        _ => runtime_error!("Don't know how to apply: {:?}", val),
    }
}

fn expand_macro(value: Value, substitutions: &HashMap<String, Value>) -> Value {
    match value {
        Value::Symbol(s) => match substitutions.get(&s) {
            Some(v) => v.clone(),
            None => Value::Symbol(s),
        },
        Value::List(list) => {
            let expanded = list.into_iter().map(|v| expand_macro(v, substitutions)).collect();
            Value::from_vec(expanded)
        }
        other => other,
    }
}

fn eval(expr: List, env: Rc<RefCell<Environment>>, k: Box<Continuation>) -> Result<Trampoline, RuntimeError> {
    match expr.shift() {
        Some((car, cdr)) => Ok(Trampoline::Bounce(car, env.clone(), Continuation::EvalExpr(cdr, env, k))),
        None => runtime_error!("trying to evaluate an empty expression list"),
    }
}

fn process(exprs: List, env: Rc<RefCell<Environment>>) -> Result<Value, RuntimeError> {
    if exprs.is_empty() {
        return Ok(null!());
    }

    let mut b = eval(exprs, env, Box::new(Continuation::Return))?;
    loop {
        match b {
            // Bounce is the usual execution path. It's used for pretty much everything.
            // Special forms are caught here instead of in env so that they can't be redefined in env.
            Trampoline::Bounce(a, env, k) => {
                b = match a {
                    Value::List(list) => match list.shift() {
                        Some((car, cdr)) => Trampoline::Bounce(car, env.clone(), Continuation::BeginFunc(cdr, env, Box::new(k))),
                        None => runtime_error!("Can't apply an empty list as a function"),
                    },
                    Value::Symbol(ref s) => {
                        let val = match s.as_ref() {
                            "if" => Value::SpecialForm(SpecialForm::If),
                            "define" => Value::SpecialForm(SpecialForm::Define),
                            "set!" => Value::SpecialForm(SpecialForm::Set),
                            "lambda" => Value::SpecialForm(SpecialForm::Lambda),
                            "λ" => Value::SpecialForm(SpecialForm::Lambda),
                            "let" => Value::SpecialForm(SpecialForm::Let),
                            "quote" => Value::SpecialForm(SpecialForm::Quote),
                            "quasiquote" => Value::SpecialForm(SpecialForm::Quasiquote),
                            "eval" => Value::SpecialForm(SpecialForm::Eval),
                            "apply" => Value::SpecialForm(SpecialForm::Apply),
                            "begin" => Value::SpecialForm(SpecialForm::Begin),
                            "and" => Value::SpecialForm(SpecialForm::And),
                            "or" => Value::SpecialForm(SpecialForm::Or),
                            "call/cc" => Value::SpecialForm(SpecialForm::CallCC),
                            "define-syntax-rule" => Value::SpecialForm(SpecialForm::DefineSyntaxRule),
                            _ => match env.borrow().get(s) {
                                Some(v) => v,
                                None => runtime_error!("Identifier not found: {}", s),
                            },
                        };
                        k.run(val)?
                    }
                    _ => k.run(a)?,
                }
            }

            // QuasiBounce is for quasiquoting mode
            // it just passes the value right through, UNLESS it's of the form (unquote X), in which case it switches back to regular evaluating mode using X as the value.
            Trampoline::QuasiBounce(a, env, k) => {
                b = match a {
                    Value::List(list) => match list.shift() {
                        Some((car, cdr)) => match car {
                            Value::Symbol(ref s) if s == "unquote" => {
                                let expr = cdr.unpack1()?;
                                Trampoline::Bounce(expr, env, k)
                            }
                            _ => Trampoline::QuasiBounce(car, env.clone(), Continuation::ContinueQuasiquoting(cdr, List::Null, env, Box::new(k))),
                        },
                        None => k.run(null!())?,
                    },
                    _ => k.run(a)?,
                }
            }

            // Run doesn't evaluate the value, it just runs k with it.
            // It's similar to running inline, but bounces to avoid growing the stack.
            Trampoline::Run(a, k) => b = k.run(a)?,

            // Land just returns the value.
            // It should only ever be created at the very beginning of process, and will be the last Trampoline value called.
            Trampoline::Land(a) => return Ok(a),
        }
    }
}

#[derive(PartialEq)]
pub struct Environment {
    parent: Option<Rc<RefCell<Environment>>>,
    values: HashMap<String, Value>,
}

impl fmt::Debug for Environment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.parent {
            Some(ref parent) => write!(f, "{:?}, {:?}", self.values, parent.borrow()),
            None => write!(f, "{:?} ", self.values),
        }
    }
}

impl Environment {
    fn new_root() -> Result<Rc<RefCell<Environment>>, RuntimeError> {
        let mut env = Environment {
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

    fn new_child(parent: Rc<RefCell<Environment>>) -> Rc<RefCell<Environment>> {
        let env = Environment {
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

    fn get_root(env_ref: Rc<RefCell<Environment>>) -> Rc<RefCell<Environment>> {
        let env = env_ref.borrow();
        match env.parent {
            Some(ref parent) => Environment::get_root(parent.clone()),
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
                let val: Value = args.unpack1()?; // unpack1(): -> Result<Value, RuntimeError>
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
                match args.unpack1()? {
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
            let (l, r) = args.unpack2()?;
            Ok(Value::Boolean(l.as_integer()? < r.as_integer()?))
        }
        ">" => {
            if args.len() != 2 {
                runtime_error!("Must supply exactly two arguments to >: {:?}", args);
            }
            let (l, r) = args.unpack2()?;
            Ok(Value::Boolean(l.as_integer()? > r.as_integer()?))
        }
        "=" => {
            if args.len() != 2 {
                runtime_error!("Must supply exactly two arguments to =: {:?}", args);
            }
            let (l, r) = args.unpack2()?;
            Ok(Value::Boolean(l.as_integer()? == r.as_integer()?))
        }
        "null?" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to null?: {:?}", args);
            }
            let v = args.unpack1()?;
            match v {
                Value::List(l) => Ok(Value::Boolean(l.is_empty())),
                _ => Ok(Value::Boolean(false)),
            }
        }
        "integer?" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to integer?: {:?}", args);
            }
            let v = args.unpack1()?;
            match v {
                Value::Integer(_) => Ok(Value::Boolean(true)),
                _ => Ok(Value::Boolean(false)),
            }
        }
        "float?" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to real?: {:?}", args);
            }
            let v = args.unpack1()?;
            match v {
                Value::Float(_) => Ok(Value::Boolean(true)),
                _ => Ok(Value::Boolean(false)),
            }
        }
        "list" => Ok(args.into_list()),
        "car" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly two arguments to car: {:?}", args);
            }
            let l = args.unpack1()?.as_list()?;
            match l.shift() {
                Some((car, _)) => Ok(car),
                None => runtime_error!("Can't run car on an empty list"),
            }
        }
        "cdr" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly two arguments to cdr: {:?}", args);
            }
            let l = args.unpack1()?.as_list()?;
            match l.shift() {
                Some((_, cdr)) => Ok(cdr.into_list()),
                None => runtime_error!("Can't run cdr on an empty list"),
            }
        }
        "cons" => {
            if args.len() != 2 {
                runtime_error!("Must supply exactly two arguments to cons: {:?}", args);
            }
            let (elem, list) = args.unpack2()?;
            Ok(list.as_list()?.unshift(elem).into_list())
        }
        "append" => {
            if args.len() != 2 {
                runtime_error!("Must supply exactly two arguments to append: {:?}", args);
            }
            let (list1raw, list2raw) = args.unpack2()?;
            let list1 = list1raw.as_list()?;
            let mut list2 = list2raw.as_list()?;

            for elem in list1.reverse() {
                list2 = list2.unshift(elem)
            }
            Ok(list2.into_list())
        }
        "error" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to error: {:?}", args);
            }
            let msg = args.unpack1()?;
            runtime_error!("{:?}", msg)
        }
        "write" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to write: {:?}", args);
            }
            let val = args.unpack1()?;
            print!("{:?}", val);
            Ok(null!())
        }
        "display" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to display: {:?}", args);
            }
            let val = args.unpack1()?;
            print!("{}", val);
            Ok(null!())
        }
        "displayln" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to displayln: {:?}", args);
            }
            let val = args.unpack1()?;
            println!("{}", val);
            Ok(null!())
        }
        "print" => {
            if args.len() != 1 {
                runtime_error!("Must supply exactly one argument to print: {:?}", args);
            }
            let val = args.unpack1()?;
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
fn exec(list: List) -> Result<Value, RuntimeError> {
    process(list, Environment::new_root()?)
}

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
    let i = vec![Value::from_vec(vec![
        Value::Symbol("quote".to_string()),
        Value::from_vec(vec![Value::Integer(1), Value::Integer(2)]),
    ])];
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::from_vec(vec![Value::Integer(1), Value::Integer(2)]));
}

#[test]
fn test_quasiquoting() {
    // runTest (quasiquote (2 (unquote (+ 1 2)) 4)) => (2 3 4)
    let i = vec![Value::from_vec(vec![
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
    assert_eq!(exec(List::from_vec(i)).unwrap(), Value::from_vec(vec![Value::Integer(2), Value::Integer(3), Value::Integer(4)]));
}

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
