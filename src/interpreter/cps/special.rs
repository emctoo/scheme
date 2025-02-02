use phf::phf_map;
use serde::{Deserialize, Serialize};

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
