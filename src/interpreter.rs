pub mod ast_walk;
pub mod cps;

// use crate::interpreter::ast_walk;
// use crate::interpreter::cps;
use crate::reader::lexer;
use crate::reader::parser;

macro_rules! try_or_return_error {
    ($inp:expr) => {
        match $inp {
            Ok(v) => v,
            Err(e) => return Err(e.to_string()),
        }
    };
}

pub fn new(t: &str) -> Interpreter { Interpreter::new(t) }

pub enum Interpreter {
    AstWalk(ast_walk::Interpreter),
    Cps(cps::Interpreter),
}

pub fn parse_code(src: &str) -> Result<Vec<parser::Node>, String> {
    let tokens = try_or_return_error!(lexer::tokenize(src));
    let ast = try_or_return_error!(parser::parse(&tokens));
    Ok(ast)
}

impl Interpreter {
    fn new(t: &str) -> Interpreter {
        match t {
            "cps" => Interpreter::Cps(cps::new().unwrap()),
            "ast-walk" => Interpreter::AstWalk(ast_walk::new()),
            _ => panic!("Interpreter type must be 'cps' or 'ast_walk'"),
        }
    }

    fn parse(&self, input: &str) -> Result<Vec<parser::Node>, String> {
        let tokens = try_or_return_error!(lexer::tokenize(input));
        let ast = try_or_return_error!(parser::parse(&tokens));
        Ok(ast)
    }

    pub fn execute(&self, input: &str) -> Result<String, String> {
        let parsed = self.parse(input)?;
        match *self {
            Interpreter::AstWalk(ref i) => Ok(format!("{:?}", try_or_return_error!(i.run(&parsed)))),
            Interpreter::Cps(ref i) => Ok(format!("{:?}", try_or_return_error!(i.run(&parsed)))),
        }
    }
}
