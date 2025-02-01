use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};

use crate::interpreter::cps::{Cont, Env, Function, List, SpecialForm, Trampoline, Value};

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
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
enum SerializedFunction {
    Scheme { args: Vec<String>, body: List, env: SerializedEnv },
    Native(String),
}
const NATIVE_FUNCTIONS: &[&'static str] = &[
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

impl Serialize for Function {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Function::Scheme(args, body, env) => {
                let serialized = SerializedFunction::Scheme {
                    args: args.clone(),
                    body: body.clone(),
                    env: to_serialized_env(&env.borrow()),
                };
                serialized.serialize(serializer)
            }
            Function::Native(name) => SerializedFunction::Native(name.to_string()).serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Function {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serialized = SerializedFunction::deserialize(deserializer)?;
        match serialized {
            SerializedFunction::Scheme { args, body, env } => {
                let env = env_from_serialized(env);
                Ok(Function::Scheme(args, body, env))
            }
            SerializedFunction::Native(name) => {
                // 通过字符串查找对应的静态字符串
                let static_name = NATIVE_FUNCTIONS
                    .iter()
                    .find(|&&func_name| func_name == name.as_str())
                    .ok_or_else(|| de::Error::custom(format!("Unknown native function: {}", name)))?;

                Ok(Function::Native(static_name))
            }
        }
    }
}
impl From<SpecialForm> for String {
    fn from(sf: SpecialForm) -> String {
        match sf {
            SpecialForm::If => "if".to_string(),
            SpecialForm::Define => "define".to_string(),
            SpecialForm::Set => "set!".to_string(),
            SpecialForm::Lambda => "lambda".to_string(),
            SpecialForm::Let => "let".to_string(),
            SpecialForm::Quote => "quote".to_string(),
            SpecialForm::Quasiquote => "quasiquote".to_string(),
            SpecialForm::Eval => "eval".to_string(),
            SpecialForm::Apply => "apply".to_string(),
            SpecialForm::Begin => "begin".to_string(),
            SpecialForm::And => "and".to_string(),
            SpecialForm::Or => "or".to_string(),
            SpecialForm::CallCC => "call/cc".to_string(),
            SpecialForm::DefineSyntaxRule => "define-syntax-rule".to_string(),
        }
    }
}

impl TryFrom<String> for SpecialForm {
    type Error = serde::de::value::Error;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.as_str() {
            "if" => Ok(SpecialForm::If),
            "define" => Ok(SpecialForm::Define),
            "set!" => Ok(SpecialForm::Set),
            "lambda" => Ok(SpecialForm::Lambda),
            "let" => Ok(SpecialForm::Let),
            "quote" => Ok(SpecialForm::Quote),
            "quasiquote" => Ok(SpecialForm::Quasiquote),
            "eval" => Ok(SpecialForm::Eval),
            "apply" => Ok(SpecialForm::Apply),
            "begin" => Ok(SpecialForm::Begin),
            "and" => Ok(SpecialForm::And),
            "or" => Ok(SpecialForm::Or),
            "call/cc" => Ok(SpecialForm::CallCC),
            "define-syntax-rule" => Ok(SpecialForm::DefineSyntaxRule),
            _ => Err(serde::de::Error::custom(format!("Invalid special form: {}", s))),
        }
    }
}

#[cfg(test)]
mod test_special_form_serialization {
    use super::*;

    #[test]
    fn test_special_form_serialization() {
        let sf = SpecialForm::Lambda;
        let serialized = serde_json::to_string(&sf).unwrap();
        assert_eq!(serialized, "\"lambda\"");

        let deserialized: SpecialForm = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, SpecialForm::Lambda);
    }

    #[test]
    fn test_special_form_invalid_deserialization() {
        let result = serde_json::from_str::<SpecialForm>("\"invalid\"");
        assert!(result.is_err());
    }

    #[test]
    fn test_all_special_forms() {
        let special_forms = vec![
            SpecialForm::If,
            SpecialForm::Define,
            SpecialForm::Set,
            SpecialForm::Lambda,
            SpecialForm::Let,
            SpecialForm::Quote,
            SpecialForm::Quasiquote,
            SpecialForm::Eval,
            SpecialForm::Apply,
            SpecialForm::Begin,
            SpecialForm::And,
            SpecialForm::Or,
            SpecialForm::CallCC,
            SpecialForm::DefineSyntaxRule,
        ];

        for sf in special_forms {
            let serialized = serde_json::to_string(&sf).unwrap();
            let deserialized: SpecialForm = serde_json::from_str(&serialized).unwrap();
            assert_eq!(sf, deserialized);
        }
    }
}
#[derive(Serialize, Deserialize)]
pub struct SerializedEnv {
    pub parent: Option<Box<SerializedEnv>>,
    pub values: HashMap<String, Value>,
}

pub fn to_serialized_env(env: &Env) -> SerializedEnv {
    let parent = env.parent.as_ref().map(|p| Box::new(to_serialized_env(&p.borrow())));

    SerializedEnv {
        values: env.values.clone(),
        parent,
    }
}

pub fn env_from_serialized(serialized: SerializedEnv) -> Rc<RefCell<Env>> {
    let parent = serialized.parent.map(|p| env_from_serialized(*p));

    let env = Env {
        values: serialized.values,
        parent,
    };

    Rc::new(RefCell::new(env))
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum SerializedCont {
    EvalExpr {
        rest: List,
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    BeginFunc {
        rest: List,
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    EvalFunc {
        f: Value,
        rest: List,
        acc: List,
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    EvalIf {
        if_expr: Value,
        else_expr: Value,
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    EvalDef {
        name: String,
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    EvalSet {
        name: String,
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    EvalLet {
        name: String,
        rest: List,
        body: List,
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    ContinueQuasiquote {
        rest: List,
        acc: List,
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    Eval {
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    EvalApplyArgs {
        args: Value,
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    Apply {
        f: Value,
        next: Box<SerializedCont>,
    },
    EvalAnd {
        rest: List,
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    EvalOr {
        rest: List,
        env: SerializedEnv,
        next: Box<SerializedCont>,
    },
    ExecCallCC {
        next: Box<SerializedCont>,
    },
    Return,
}

pub fn to_serialized_cont(cont: &Box<Cont>) -> SerializedCont {
    match &**cont {
        Cont::EvalExpr(rest, env, next) => SerializedCont::EvalExpr {
            rest: rest.clone(),
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },
        Cont::BeginFunc(rest, env, next) => SerializedCont::BeginFunc {
            rest: rest.clone(),
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },
        Cont::EvalFunc(f, rest, acc, env, next) => SerializedCont::EvalFunc {
            f: f.clone(),
            rest: rest.clone(),
            acc: acc.clone(),
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::EvalIf(if_expr, else_expr, env, next) => SerializedCont::EvalIf {
            if_expr: if_expr.clone(),
            else_expr: else_expr.clone(),
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::EvalDef(name, env, next) => SerializedCont::EvalDef {
            name: name.clone(),
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::EvalSet(name, env, next) => SerializedCont::EvalSet {
            name: name.clone(),
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::EvalLet(name, rest, body, env, next) => SerializedCont::EvalLet {
            name: name.clone(),
            rest: rest.clone(),
            body: body.clone(),
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::ContinueQuasiquote(rest, acc, env, next) => SerializedCont::ContinueQuasiquote {
            rest: rest.clone(),
            acc: acc.clone(),
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::Eval(env, next) => SerializedCont::Eval {
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::EvalApplyArgs(args, env, next) => SerializedCont::EvalApplyArgs {
            args: args.clone(),
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::Apply(f, next) => SerializedCont::Apply {
            f: f.clone(),
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::EvalAnd(rest, env, next) => SerializedCont::EvalAnd {
            rest: rest.clone(),
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::EvalOr(rest, env, next) => SerializedCont::EvalOr {
            rest: rest.clone(),
            env: to_serialized_env(&env.borrow()),
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::ExecCallCC(next) => SerializedCont::ExecCallCC {
            next: Box::new(to_serialized_cont(next)),
        },

        Cont::Return => SerializedCont::Return,
    }
}

pub fn from_serialized_cont(serialized: SerializedCont) -> Cont {
    match serialized {
        SerializedCont::EvalExpr { rest, env, next } => Cont::EvalExpr(rest, env_from_serialized(env), Box::new(from_serialized_cont(*next))),
        SerializedCont::BeginFunc { rest, env, next } => Cont::BeginFunc(rest, env_from_serialized(env), Box::new(from_serialized_cont(*next))),
        SerializedCont::EvalFunc { f, rest, acc, env, next } => {
            Cont::EvalFunc(f, rest, acc, env_from_serialized(env), Box::new(from_serialized_cont(*next)))
        }
        SerializedCont::EvalIf {
            if_expr,
            else_expr,
            env,
            next,
        } => Cont::EvalIf(if_expr, else_expr, env_from_serialized(env), Box::new(from_serialized_cont(*next))),
        SerializedCont::EvalDef { name, env, next } => Cont::EvalDef(name, env_from_serialized(env), Box::new(from_serialized_cont(*next))),
        SerializedCont::EvalSet { name, env, next } => Cont::EvalSet(name, env_from_serialized(env), Box::new(from_serialized_cont(*next))),
        SerializedCont::EvalLet { name, rest, body, env, next } => {
            Cont::EvalLet(name, rest, body, env_from_serialized(env), Box::new(from_serialized_cont(*next)))
        }
        SerializedCont::ContinueQuasiquote { rest, acc, env, next } => {
            Cont::ContinueQuasiquote(rest, acc, env_from_serialized(env), Box::new(from_serialized_cont(*next)))
        }
        SerializedCont::Eval { env, next } => Cont::Eval(env_from_serialized(env), Box::new(from_serialized_cont(*next))),
        SerializedCont::EvalApplyArgs { args, env, next } => {
            Cont::EvalApplyArgs(args, env_from_serialized(env), Box::new(from_serialized_cont(*next)))
        }
        SerializedCont::Apply { f, next } => Cont::Apply(f, Box::new(from_serialized_cont(*next))),
        SerializedCont::EvalAnd { rest, env, next } => Cont::EvalAnd(rest, env_from_serialized(env), Box::new(from_serialized_cont(*next))),
        SerializedCont::EvalOr { rest, env, next } => Cont::EvalOr(rest, env_from_serialized(env), Box::new(from_serialized_cont(*next))),
        SerializedCont::ExecCallCC { next } => Cont::ExecCallCC(Box::new(from_serialized_cont(*next))),
        SerializedCont::Return => Cont::Return,
    }
}

impl Serialize for Cont {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let serialized = match self {
            Cont::EvalExpr(rest, env, next) => SerializedCont::EvalExpr {
                rest: rest.clone(),
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::BeginFunc(rest, env, next) => SerializedCont::BeginFunc {
                rest: rest.clone(),
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::EvalFunc(f, rest, acc, env, next) => SerializedCont::EvalFunc {
                f: f.clone(),
                rest: rest.clone(),
                acc: acc.clone(),
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::EvalIf(if_expr, else_expr, env, next) => SerializedCont::EvalIf {
                if_expr: if_expr.clone(),
                else_expr: else_expr.clone(),
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::EvalDef(name, env, next) => SerializedCont::EvalDef {
                name: name.clone(),
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::EvalSet(name, env, next) => SerializedCont::EvalSet {
                name: name.clone(),
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::EvalLet(name, rest, body, env, next) => SerializedCont::EvalLet {
                name: name.clone(),
                rest: rest.clone(),
                body: body.clone(),
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::ContinueQuasiquote(rest, acc, env, next) => SerializedCont::ContinueQuasiquote {
                rest: rest.clone(),
                acc: acc.clone(),
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::Eval(env, next) => SerializedCont::Eval {
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::EvalApplyArgs(args, env, next) => SerializedCont::EvalApplyArgs {
                args: args.clone(),
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::Apply(f, next) => SerializedCont::Apply {
                f: f.clone(),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::EvalAnd(rest, env, next) => SerializedCont::EvalAnd {
                rest: rest.clone(),
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::EvalOr(rest, env, next) => SerializedCont::EvalOr {
                rest: rest.clone(),
                env: to_serialized_env(&env.borrow()),
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::ExecCallCC(next) => SerializedCont::ExecCallCC {
                next: Box::new(to_serialized_cont(next)),
            },

            Cont::Return => SerializedCont::Return,
        };
        serialized.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Cont {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serialized = SerializedCont::deserialize(deserializer)?;
        Ok(from_serialized_cont(serialized))
    }
}

#[cfg(test)]
mod test_cond_serialization {
    use super::*;

    #[cfg(test)]
    mod test_cont_serialization {
        use super::*;
        use crate::interpreter::cps::Env;

        #[test]
        fn test_serialize_complex_cont() {
            let env = Env::new_root().unwrap();

            let inner = Cont::Return;
            assert_eq!(serde_json::to_string(&inner).unwrap(), r#"{"type":"Return"}"#);

            let middle = Cont::EvalFunc(Value::Symbol("test".to_string()), List::Null, List::Null, env.clone(), Box::new(inner));
            let outer = Cont::EvalExpr(List::Null, env.clone(), Box::new(middle));

            let serialized = serde_json::to_string(&outer).unwrap();
            let deserialized: Cont = serde_json::from_str(&serialized).unwrap();

            // 验证基本结构
            match deserialized {
                Cont::EvalExpr(_, _, box_middle) => match *box_middle {
                    Cont::EvalFunc(_, _, _, _, box_inner) => match *box_inner {
                        Cont::Return => (),
                        _ => panic!("Wrong inner continuation type"),
                    },
                    _ => panic!("Wrong middle continuation type"),
                },
                _ => panic!("Wrong outer continuation type"),
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
enum SerializedTrampoline {
    Bounce { val: Value, env: SerializedEnv, k: Cont },
    QuasiquoteBounce { val: Value, env: SerializedEnv, k: Cont },
    Run { val: Value, k: Cont },
    Land { val: Value },
}

impl Serialize for Trampoline {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Trampoline::Bounce(val, env, k) => {
                let serialized = SerializedTrampoline::Bounce {
                    val: val.clone(),
                    env: to_serialized_env(&env.borrow()),
                    k: k.clone(),
                };
                serialized.serialize(serializer)
            }
            Trampoline::QuasiquoteBounce(val, env, k) => {
                let serialized = SerializedTrampoline::QuasiquoteBounce {
                    val: val.clone(),
                    env: to_serialized_env(&env.borrow()),
                    k: k.clone(),
                };
                serialized.serialize(serializer)
            }
            Trampoline::Run(val, k) => {
                let serialized = SerializedTrampoline::Run {
                    val: val.clone(),
                    k: k.clone(),
                };
                serialized.serialize(serializer)
            }
            Trampoline::Land(val) => {
                let serialized = SerializedTrampoline::Land { val: val.clone() };
                serialized.serialize(serializer)
            }
        }
    }
}

impl<'de> Deserialize<'de> for Trampoline {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serialized = SerializedTrampoline::deserialize(deserializer)?;

        Ok(match serialized {
            SerializedTrampoline::Bounce { val, env, k } => Trampoline::Bounce(val, env_from_serialized(env), k),
            SerializedTrampoline::QuasiquoteBounce { val, env, k } => Trampoline::QuasiquoteBounce(val, env_from_serialized(env), k),
            SerializedTrampoline::Run { val, k } => Trampoline::Run(val, k),
            SerializedTrampoline::Land { val } => Trampoline::Land(val),
        })
    }
}

#[cfg(test)]
mod test_trampoline_serialization {
    use super::*;

    #[test]
    fn test_serialize_land() {
        let t = Trampoline::Land(Value::Integer(42));
        let serialized = serde_json::to_string(&t).unwrap();
        let deserialized: Trampoline = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Trampoline::Land(Value::Integer(n)) => assert_eq!(n, 42),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_serialize_run() {
        // let env = Env::new_root().unwrap();
        let t = Trampoline::Run(Value::Integer(1), Cont::Return);

        let serialized = serde_json::to_string(&t).unwrap();
        let deserialized: Trampoline = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Trampoline::Run(Value::Integer(n), Cont::Return) => {
                assert_eq!(n, 1);
            }
            _ => panic!("Wrong variant"),
        }
    }
}
