extern crate libc;

use clap::{Arg, Command};
use scheme::interpreter::interpreter::{self};
use std::ffi::CStr;
use std::ffi::CString;
use std::{fs::File, io::Read, path::Path};

#[link(name = "readline")]
extern "C" {
    fn readline(prompt: *const libc::c_char) -> *const libc::c_char;
    fn add_history(entry: *const libc::c_char);
}

fn prompt_for_input(prompt: &str) -> Option<String> {
    let prompt_c_str = CString::new(prompt).unwrap();

    unsafe {
        // wait for enter/CTRL-C/CTRL-D
        let raw = readline(prompt_c_str.as_ptr());
        if raw.is_null() {
            return None;
        }

        // parse into String and return
        let buf = CStr::from_ptr(raw).to_bytes();
        let cs = String::from_utf8(buf.to_vec()).unwrap();

        // add to shell history unless it's an empty string
        if !cs.is_empty() {
            add_history(raw);
        }

        Some(cs)
    }
}

fn start<F: Fn(String) -> Result<String, String>>(prompt: &str, f: F) {
    loop {
        match prompt_for_input(prompt) {
            Some(input) => {
                if !input.is_empty() {
                    let result = f(input);
                    println!("{}", result.unwrap_or_else(|e| e));
                }
            }
            None => return,
        };
    }
}

fn main() {
    let matches = Command::new("Scheme")
        .arg(
            Arg::new("type")
                .short('t')
                .long("type")
                .help("set interpreter type")
                .value_parser(["ast_walk", "cps"])
                .default_value("cps"),
        )
        .arg(Arg::new("file").help("Input file").num_args(0..=1))
        .get_matches();

    let interpreter_type = matches.get_one::<String>("type").expect("default value should exist");
    let interpreter = interpreter::new(interpreter_type);

    match matches.get_one::<String>("file") {
        Some(filename) => {
            let path = Path::new(&filename);
            let mut file = File::open(path).unwrap();
            let mut contents = String::new();
            file.read_to_string(&mut contents).unwrap();
            match interpreter.execute(&contents) {
                Ok(_) => {}
                Err(e) => println!("{}", e),
            }
        }
        None => start("> ", |code| interpreter.execute(&code)),
    }
}

macro_rules! test {
    ($name:ident, $src:expr, $res:expr) => {
        #[test]
        fn $name() {
            // assert_execute_all!($src, $res);
            assert_eq!(interpreter::new("ast_walk").execute($src).unwrap(), $res);
            assert_eq!(interpreter::new("cps").execute($src).unwrap(), $res);
        }
    }; // ($name:ident, $src:expr, $res:expr, cps) => {
       //     #[test]
       //     fn $name() {
       //         // assert_execute_cps!($src, $res);
       //         assert_eq!(interpreter::new("cps").execute($src).unwrap(), $res);
       //     }
       // };
}

macro_rules! test_fail {
    ($name:ident, $src:expr, $res:expr) => {
        #[test]
        fn $name() {
            // assert_execute_fail_all!($src, $res);
            assert_eq!(interpreter::new("ast_walk").execute($src).err().unwrap(), $res);
            assert_eq!(interpreter::new("cps").execute($src).err().unwrap(), $res);
        }
    }; // ($name:ident, $src:expr, $res:expr, cps) => {
       //     #[test]
       //     fn $name() {
       //         // assert_execute_fail_cps!($src, $res);
       //         assert_eq!(interpreter::new("cps").execute($src).err().unwrap(), $res);
       //     }
       // };
}

// macro_rules! assert_execute_all {
//     ($src:expr, $res:expr) => {
//         assert_execute_ast_walk!($src, $res);
//         assert_execute_cps!($src, $res);
//     };
// }

// macro_rules! assert_execute_fail_all {
//     ($src:expr, $res:expr) => {
//         assert_execute_fail_ast_walk!($src, $res);
//         assert_execute_fail_cps!($src, $res);
//     };
// }

// macro_rules! assert_execute_ast_walk {
//     ($src:expr, $res:expr) => {
//         assert_eq!(interpreter::new("ast_walk").execute($src).unwrap(), $res)
//     };
// }

// macro_rules! assert_execute_fail_ast_walk {
//     ($src:expr, $res:expr) => {
//         assert_eq!(interpreter::new("ast_walk").execute($src).err().unwrap(), $res)
//     };
// }

// macro_rules! assert_execute_cps {
//     ($src:expr, $res:expr) => {
//         assert_eq!(interpreter::new("cps").execute($src).unwrap(), $res)
//     };
// }

// macro_rules! assert_execute_fail_cps {
//     ($src:expr, $res:expr) => {
//         assert_eq!(interpreter::new("cps").execute($src).err().unwrap(), $res)
//     };
// }

test!(identity1, "1", "1");
test!(identity2, "#f", "#f");
test!(identity3, "\"hi\"", "\"hi\"");
test!(identity4, "(lambda (x) x)", "#<procedure>");

test!(addition1, "(+ 2 3)", "5");
test!(addition2, "(+ 2 -3)", "-1");
test!(addition3, "(+ 2 3 4 5)", "14");
test!(addition4, "(+ (+ 2 -3) (+ 4 -5) (+ 6 -7) (+ 8 -9 10 -11 12 -13))", "-6");

test!(subtraction1, "(- 3 2)", "1");
test!(subtraction2, "(- 2 -3)", "5");

test!(multiplication1, "(* 2 3)", "6");
test!(multiplication2, "(* 2 -3)", "-6");
test!(multiplication3, "(* 2 3 4 5)", "120");

test!(division1, "(/ 4 2)", "2");
test!(division2, "(/ 4 3)", "1");
test!(division3, "(/ 4 -2)", "-2");

test!(lessthan1, "(< 1 2)", "#t");
test!(lessthan2, "(< 2 2)", "#f");
test!(lessthan3, "(< 3 2)", "#f");
test!(lessthan4, "(< -1 2)", "#t");
test!(lessthan5, "(< 2 -1)", "#f");

test!(greaterthan1, "(> 3 2)", "#t");
test!(greaterthan2, "(> 2 2)", "#f");
test!(greaterthan3, "(> 1 2)", "#f");
test!(greaterthan4, "(> 2 -1)", "#t");
test!(greaterthan5, "(> -1 2)", "#f");

test!(equal1, "(= 2 2)", "#t");
test!(equal2, "(= 1 2)", "#f");
test!(equal3, "(= -1 -1)", "#t");
test!(equal4, "(= -1 2)", "#f");

test!(multiple_expression_return1, "(+ 2 3)\n(+ 1 2)", "3");

test!(nested_expressions1, "(+ 2 (- (+ 9 1) 4))", "8");

test!(list_creation1, "(list)", "()");
test!(list_creation2, "(list 1 2 3)", "(1 2 3)");
test!(list_creation3, "(list 1 (list 2 3) (list 4) (list))", "(1 (2 3) (4) ())");

test!(null1, "(null? '())", "#t");
test!(null2, "(null? '(1))", "#f");
test!(null3, "(null? '(()))", "#f");
test!(null4, "(null? 1)", "#f");
test!(null5, "(null? #t)", "#f");
test!(null6, "(null? #f)", "#f");
test!(null7, "(null? 'a)", "#f");
test!(null8, "(null? \"a\")", "#f");

test!(cons1, "(cons 1 '())", "(1)");
test!(cons2, "(cons 1 '(2))", "(1 2)");
test!(cons3, "(cons '(1) '(2))", "((1) 2)");

test!(car1, "(car '(1))", "1");
test!(car2, "(car '(1 2 3))", "1");
test!(car3, "(car '((1) (2 3)))", "(1)");
test_fail!(car4, "(car '())", "RuntimeError: Can't run car on an empty list");

test!(cdr1, "(cdr '(1 2))", "(2)");
test!(cdr2, "(cdr '(1 2 3))", "(2 3)");
test!(cdr3, "(cdr '(1))", "()");
test!(cdr4, "(cdr '((1) (2 3)))", "((2 3))");
test_fail!(cdr5, "(cdr '())", "RuntimeError: Can't run cdr on an empty list");

test!(append1, "(append '(1) '(2))", "(1 2)");
test!(append2, "(append '(1) '())", "(1)");
test!(append3, "(append '() '(2))", "(2)");
test!(append4, "(append '() '())", "()");
test!(append5, "(append '(1) '((2)))", "(1 (2))");

test!(variable_definition1, "(define x 2) (+ x x x)", "6");
test!(variable_definition2, "(define x 2) ((lambda (x) x) 3)", "3");
test!(variable_definition3, "(define x 2) (let ((x 3)) x)", "3");
test!(variable_definition4, "(define x 2) ((lambda (x) (define x 4) x) 3)", "4");
test!(variable_definition5, "(define x 2) (let ((x 3)) (define x 4) x)", "4");

test_fail!(duplicate_variable_definition1, "(define x 2) (define x 3)", "RuntimeError: Duplicate define: \"x\"");
test_fail!(duplicate_variable_definition2, "((lambda () (define x 2) (define x 3)))", "RuntimeError: Duplicate define: \"x\"");
test_fail!(duplicate_variable_definition3, "(let ((y 2)) (define x 2) (define x 3))", "RuntimeError: Duplicate define: \"x\"");

test!(variable_modification1, "(define x 2) (set! x 3) (+ x x x)", "9");
test!(variable_modification2, "(define x 2) ((lambda () (set! x 3))) x", "3");
test!(variable_modification3, "(define x 2) (let ((y 2)) (set! x 3)) x", "3");

test_fail!(unknown_variable_modification1, "(set! x 3)", "RuntimeError: Can't set! an undefined variable: \"x\"");

test!(procedure_definition1, "(define double (lambda (x) (+ x x))) (double 8)", "16");
test!(procedure_definition2, "(define twice (lambda (f v) (f (f v)))) (twice (lambda (x) (+ x x)) 8)", "32");
test!(procedure_definition3, "(define twice (λ (f v) (f (f v)))) (twice (λ (x) (+ x x)) 8)", "32");
test!(procedure_definition4, "((λ (x) (+ x x)) 8)", "16");
test!(procedure_definition5, "(define foo (λ (x) (λ (y) (+ x y)))) (define add2 (foo 2)) (add2 5)", "7");
test!(procedure_definition6, "(define foo (λ (x) (λ (y) (+ x y)))) (define add2 (foo 2)) ((λ (x) (add2 (+ x 1))) 1)", "4");
test!(procedure_definition7, "(define (twice f v) (f (f v))) (twice (lambda (x) (+ x x)) 8)", "32");

test!(begin_statement1, "(define x 1) (begin (set! x 5) (set! x (+ x 2)) x)", "7");

test!(let_statement1, "(let ((x 2)) (+ x x))", "4");
test!(let_statement2, "(let ((x 2) (y 3)) (+ x y))", "5");
test!(let_statement3, "(let ((x 2) (y 3)) (set! y (+ y 1)) (+ x y))", "6");

test!(conditional_execution1, "(if #t 1 2)", "1");
test!(conditional_execution2, "(if #f 1 2)", "2");
test!(conditional_execution3, "(if 0 1 2)", "1");
test!(conditional_execution4, "(if \"\" 1 2)", "1");

test!(conditional_execution_doesnt_run_other_case1, "(if #t 1 (error \"bad\"))", "1");
test!(conditional_execution_doesnt_run_other_case2, "(if #f (error \"bad\") 2)", "2");

test!(boolean_operators1, "(and)", "#t");
test!(boolean_operators2, "(and #t)", "#t");
test!(boolean_operators3, "(and 1)", "1");
test!(boolean_operators4, "(and 1 2 3)", "3");
test!(boolean_operators5, "(and 1 #f 3)", "#f");
test!(boolean_operators6, "(and 1 #f (error \"bad\"))", "#f");
test!(boolean_operators7, "(or)", "#f");
test!(boolean_operators8, "(or #f)", "#f");
test!(boolean_operators9, "(or 1)", "1");
test!(boolean_operators10, "(or 1 2)", "1");
test!(boolean_operators11, "(or 1 #f)", "1");
test!(boolean_operators12, "(or #f 3)", "3");
test!(boolean_operators13, "(or #f #f)", "#f");
test!(boolean_operators14, "(or 1 (error \"bad\"))", "1");

test!(quoting1, "(quote #t)", "#t");
test!(quoting2, "(quote 1)", "1");
test!(quoting3, "(quote sym)", "sym");
test!(quoting4, "(quote \"hi\")", "\"hi\"");
test!(quoting5, "(quote (1 2))", "(1 2)");
test!(quoting6, "(quote (a b))", "(a b)");
test!(quoting7, "(quote (a b (c (d) e ())))", "(a b (c (d) e ()))");
test!(quoting8, "(quote (a (quote b)))", "(a (quote b))");
test!(quoting9, "'(1 2)", "(1 2)");
test!(quoting10, "'(a b (c (d) e ()))", "(a b (c (d) e ()))");
test!(quoting11, "'(1 '2)", "(1 (quote 2))");

test!(quasiquoting1, "(quasiquote (1 2))", "(1 2)");
test!(quasiquoting2, "(quasiquote (2 (unquote (+ 1 2)) 4))", "(2 3 4)");
test!(quasiquoting3, "`(2 ,(+ 1 2) 4)", "(2 3 4)");
test!(quasiquoting4, "(define formula '(+ x y)) `((lambda (x y) ,formula) 2 3)", "((lambda (x y) (+ x y)) 2 3)");

test!(apply1, "(apply + '(1 2 3))", "6");
test!(apply2, "(define foo (lambda (f) (lambda (x y) (f (f x y) y)))) (apply (apply foo (list +)) '(5 3))", "11");

test!(eval1, "(eval '(+ 1 2 3))", "6");
test!(eval2, "(define eval-formula (lambda (formula) (eval `((lambda (x y) ,formula) 2 3)))) (eval-formula '(+ (- y x) y))", "4");
test_fail!(
    eval3,
    "(define bad-eval-formula (lambda (formula) ((lambda (x y) (eval formula)) 2 3))) (bad-eval-formula '(+ x y))",
    "RuntimeError: Identifier not found: x"
);

test_fail!(bad_syntax1, "(22+)", "SyntaxError: Unexpected character when looking for a delimiter: + (line: 1, column: 4)");
test_fail!(bad_syntax2, "(+ 2 3)\n(+ 1 2-)", "SyntaxError: Unexpected character when looking for a delimiter: - (line: 2, column: 7)");

test_fail!(generated_runtime_error1, "(error \"fail, please\")", "RuntimeError: \"fail, please\"");
test_fail!(generated_runtime_error2, "(error (+ 2 3))", "RuntimeError: 5");

test_fail!(errors_halt_execution1, "(error \"fail, please\") 5", "RuntimeError: \"fail, please\"");

test!(unicode_identifiers1, "(define ★ 3) (define ♫ 4) (+ ★ ♫)", "7");

test!(macros1, "(define-syntax-rule (incr x) (set! x (+ x 1))) (define a 1) (incr a) a", "2");
test!(macros2, "(define-syntax-rule (incr x) (set! x (+ x 1))) (define x 1) (incr x) x", "2");
test!(
    macros3,
    r"
    (define-syntax-rule (incr x) (set! x (+ x 1))) 
    (define-syntax-rule (foo x y z) (if x (incr y) (incr z)))
    (define a #t)
    (define b 10)
    (define c 20)
    (foo a b c)
    (set! a #f)
    (foo a b c)
    (list b c)",
    "(11 21)"
);
test!(macros4, "(define-syntax-rule (foo x) (if x (+ (foo #f) 3) 10)) (foo #t)", "13");
test!(macros5, "(define-syntax-rule (testy a b c) (if a b c)) (testy #t 1 (error \"test\")) (testy #f (error \"test\") 2)", "2");

test!(multiline1, "(define x 3)\n(define y 4)\n(+ x y)", "7");

test!(comment1, "(define x 3)\n(define y 4)\n;(set! y 5)\n(+ x y); (+ x y)", "7");

#[cfg(test)]
mod tests {
    use crate::interpreter;

    #[test]
    fn test_tail_call() {
        // test!(tail_call_optimization1, "(define (f i) (if (= i 1000) '() (f (+ i 1)))) (f 1)", "()", cps);
        let src = "(define (f i) (if (= i 1000) '() (f (+ i 1)))) (f 1)";
        let res = "()";
        assert_eq!(interpreter::new("cps").execute(src).unwrap(), res);
    }
}
