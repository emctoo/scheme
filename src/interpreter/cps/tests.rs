#![allow(unused_imports)]

use crate::interpreter::cps::env::Env;
use crate::interpreter::cps::error::*;
use crate::interpreter::cps::trampoline::*;
use crate::interpreter::cps::value::*;
use crate::interpreter::cps::*;
use crate::null;

#[cfg(test)]
mod test_trampoline {
    use super::*;
    use crate::reader::{lexer, parser};
    use std::cell::RefCell;
    use std::rc::Rc;

    // 辅助函数：将 Scheme 代码转换为 List 结构
    fn parse_list(code: &str) -> List {
        let tokens = lexer::tokenize(code).unwrap();
        let nodes = parser::parse(&tokens).unwrap();
        List::from_nodes(&nodes)
    }

    fn setup_env() -> Rc<RefCell<Env>> {
        let env = Env::new_root().unwrap();
        env.borrow_mut().define("x".into(), Value::Integer(42)).unwrap();
        env.borrow_mut().define("y".into(), Value::Integer(10)).unwrap();
        env
    }

    #[test]
    fn test_handle_simple_value() {
        // 测试简单值（非列表）
        let env = setup_env();
        let result = quasiquote_bounce(Value::Integer(42), env, Cont::Return).unwrap();

        // 对于简单值，应该直接返回 Trampoline::Land
        match result {
            Trampoline::Land(Value::Integer(42)) => (),
            _ => panic!("Expected Land(42), got {:?}", result),
        }
    }

    #[test]
    fn test_handle_unquote() {
        // 测试 unquote 形式: (unquote expr)
        let env = setup_env();
        let list = List::from_vec(vec![Value::Symbol("unquote".to_string()), Value::Integer(42)]);

        let result = quasiquote_bounce(Value::List(list), env.clone(), Cont::Return).unwrap();

        // unquote 应该触发 Bounce
        match result {
            Trampoline::Bounce(Value::Integer(42), env2, Cont::Return) => {
                assert_eq!(env2, env);
            }
            _ => panic!("Expected Bounce with unquote expression, got {:?}", result),
        }
    }

    #[test]
    fn test_handle_nested_list() {
        // 测试嵌套列表: (a b c)
        let env = setup_env();
        let list = List::from_vec(vec![
            Value::Symbol("a".to_string()),
            Value::Symbol("b".to_string()),
            Value::Symbol("c".to_string()),
        ]);

        let result = quasiquote_bounce(Value::List(list), env.clone(), Cont::Return).unwrap();

        // 应该继续 QuasiquoteBounce 处理
        match result {
            Trampoline::QuasiquoteBounce(Value::Symbol(s), env2, Cont::ContinueQuasiquote(rest, acc, env3, k)) => {
                assert_eq!(s, "a");
                assert_eq!(env2, env);
                assert_eq!(env3, env);
                assert_eq!(acc, List::Null);
                // rest 应该包含 (b c)
                assert_eq!(rest, List::from_vec(vec![Value::Symbol("b".to_string()), Value::Symbol("c".to_string()),]));
                match *k {
                    Cont::Return => (),
                    _ => panic!("Expected Return continuation"),
                }
            }
            _ => panic!("Expected QuasiquoteBounce, got {:?}", result),
        }
    }

    #[test]
    fn test_handle_unquote_with_complex_expr() {
        // 测试带有复杂表达式的 unquote: (unquote (+ 1 2))
        let env = setup_env();
        let list = List::from_vec(vec![
            Value::Symbol("unquote".to_string()),
            Value::List(List::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(1), Value::Integer(2)])),
        ]);

        let result = quasiquote_bounce(Value::List(list), env.clone(), Cont::Return).unwrap();

        // 应该返回 Bounce 来计算 (+ 1 2)
        match result {
            Trampoline::Bounce(Value::List(expr), env2, Cont::Return) => {
                assert_eq!(env2, env);
                assert_eq!(expr, List::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(1), Value::Integer(2),]));
            }
            _ => panic!("Expected Bounce with complex unquote expression, got {:?}", result),
        }
    }

    #[test]
    fn test_handle_symbol() {
        // 测试单个符号
        let env = setup_env();
        let result = quasiquote_bounce(Value::Symbol("x".to_string()), env, Cont::Return).unwrap();

        // 符号应该直接返回
        match result {
            Trampoline::Land(Value::Symbol(s)) => assert_eq!(s, "x"),
            _ => panic!("Expected Land with symbol, got {:?}", result),
        }
    }

    #[test]
    fn test_handle_mixed_list() {
        // 测试混合列表: (a ,(+ 1 2) c)
        let env = setup_env();
        let list = List::from_vec(vec![
            Value::Symbol("a".to_string()),
            Value::List(List::from_vec(vec![
                Value::Symbol("unquote".to_string()),
                Value::List(List::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(1), Value::Integer(2)])),
            ])),
            Value::Symbol("c".to_string()),
        ]);

        let result = quasiquote_bounce(Value::List(list), env.clone(), Cont::Return).unwrap();

        // 应该首先处理第一个元素 'a'
        match result {
            Trampoline::QuasiquoteBounce(Value::Symbol(s), env2, Cont::ContinueQuasiquote(_rest, acc, env3, k)) => {
                assert_eq!(s, "a");
                assert_eq!(env2, env);
                assert_eq!(env3, env);
                assert_eq!(acc, List::Null);
                match *k {
                    Cont::Return => (),
                    _ => panic!("Expected Return continuation"),
                }
            }
            _ => panic!("Expected QuasiquoteBounce with mixed list, got {:?}", result),
        }
    }

    #[test]
    fn test_handle_empty_list() {
        let env = setup_env();
        let k = Cont::Return;
        let val = Value::List(List::Null);

        let result = bounce(val, env, k);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().message, "Expected non-empty list");
    }

    #[test]
    fn test_handle_non_empty_list() {
        let env = setup_env();
        let k = Cont::Return;

        // 构造列表 (+ 1 2)
        let val = Value::from_vec(vec![Value::Symbol("+".to_string()), Value::Integer(1), Value::Integer(2)]);

        let result = bounce(val, env.clone(), k).unwrap();
        match result {
            Trampoline::Bounce(car, env2, Cont::BeginFunc(cdr, env3, _)) => {
                assert_eq!(car, Value::Symbol("+".to_string()));
                assert_eq!(env2, env);
                assert_eq!(env3, env);
                assert_eq!(cdr, List::from_vec(vec![Value::Integer(1), Value::Integer(2)]));
            }
            _ => panic!("Expected Bounce with BeginFunc continuation"),
        }
    }

    #[test]
    fn test_handle_special_form() {
        let env = setup_env();
        let k = Cont::Return;
        let val = Value::Symbol("if".to_string());

        let result = bounce(val, env, k).unwrap();
        match result {
            // 当使用 Return continuation 时，会直接得到 Land
            Trampoline::Land(Value::SpecialForm(SpecialForm::If)) => (),
            _ => panic!("Expected Land with SpecialForm::If"),
        }
    }

    #[test]
    fn test_handle_special_form_non_return() {
        let env = setup_env();

        // (if #t 1 2)
        let args = List::from_vec(vec![Value::Boolean(true), Value::Integer(1), Value::Integer(2)]);

        let k = Cont::BeginFunc(args, env.clone(), Box::new(Cont::Return));
        // let val = Value::Symbol("if".to_string());

        let result = bounce_symbol(&"if".to_string(), env.clone(), k).unwrap();
        match &result {
            Trampoline::Bounce(val, env1, Cont::EvalIf(if_expr, else_expr, env2, k)) => {
                // 验证各个部分是否正确
                assert_eq!(*val, Value::Boolean(true)); // 条件部分
                assert_eq!(*if_expr, Value::Integer(1)); // if 分支
                assert_eq!(*else_expr, Value::Integer(2)); // else 分支
                assert_eq!(env1, &env); // 环境相同
                assert_eq!(env2, &env); // 环境相同
                match **k {
                    Cont::Return => (), // 验证最内层的 continuation 是 Return
                    _ => panic!("Expected Return continuation"),
                }
            }
            _ => panic!("Expected Run with SpecialForm::If and BeginFunc continuation, got {:?}", result),
        }
    }

    #[test]
    fn test_handle_special_form_complete_if() {
        let env = setup_env();

        // 构造完整的 if 表达式: (if #t 1 2)
        let if_expr = Value::List(List::from_vec(vec![
            Value::Symbol("if".to_string()),
            Value::Boolean(true),
            Value::Integer(1),
            Value::Integer(2),
        ]));

        let result = bounce(if_expr, env.clone(), Cont::Return).unwrap();

        // 验证第一步的结果是否正确
        match result {
            Trampoline::Bounce(val, _bounce_env, k) => {
                assert_eq!(val, Value::Symbol("if".to_string()));

                // 验证 continuation 中的参数列表
                match k {
                    Cont::BeginFunc(args, _, _) => {
                        assert_eq!(args, List::from_vec(vec![Value::Boolean(true), Value::Integer(1), Value::Integer(2)]));
                    }
                    _ => panic!("Expected BeginFunc continuation"),
                }
            }
            _ => panic!("Expected Bounce with if symbol"),
        }
    }

    // 添加一个测试验证特殊形式的基本识别
    #[test]
    fn test_special_form_identification() {
        let env = setup_env();
        let special_forms = vec!["if", "define", "lambda", "begin"];

        for form in special_forms {
            let val = Value::Symbol(form.to_string());
            let result = bounce(val.clone(), env.clone(), Cont::Return).unwrap();

            match result {
                Trampoline::Land(Value::SpecialForm(_)) => (),
                _ => panic!("Expected Land with SpecialForm for {}", form),
            }
        }
    }

    #[test]
    fn test_handle_defined_symbol() {
        let env = setup_env();
        let k = Cont::Return;
        let val = Value::Symbol("x".to_string());

        let result = bounce(val, env, k).unwrap();
        // Return continuation 会直接得到 Land
        match result {
            Trampoline::Land(Value::Integer(42)) => (),
            _ => panic!("Expected Land with Value::Integer(42)"),
        }
    }

    #[test]
    fn test_handle_undefined_symbol() {
        let env = setup_env();
        let k = Cont::Return;
        let val = Value::Symbol("undefined".to_string());

        let result = bounce(val, env, k);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().message, "Identifier not found: undefined");
    }

    #[test]
    fn test_handle_literal() {
        let env = setup_env();
        let k = Cont::Return;
        let val = Value::Integer(5);

        let result = bounce(val, env, k).unwrap();
        // Return continuation 会直接得到 Land
        match result {
            Trampoline::Land(Value::Integer(5)) => (),
            _ => panic!("Expected Land with Value::Integer(5)"),
        }
    }

    #[test]
    fn test_return() {
        // test the return trampoline
        let _env = Env::new_root().unwrap();
        let k = Cont::Return;
        let result = k.run(Value::Integer(42)).unwrap();
        assert_eq!(result, Trampoline::Land(Value::Integer(42)));
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
        let result = cps(code, env).unwrap();
        assert_eq!(result, Value::Integer(3));
    }

    #[test]
    fn test_simple_quasiquote() {
        // 测试基本的 quasiquote，不包含 unquote
        // `(1 2 3)
        let code = parse_list("`(1 2 3)");
        let env = Env::new_root().unwrap();
        let result = cps(code, env).unwrap();

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
        let result = cps(code, env).unwrap();
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
        let result = cps(code, env).unwrap();

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
        let result = cps(code, env).unwrap();
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

#[cfg(test)]
mod test_cps {
    use super::*;

    fn exec(list: List) -> Result<Value, RuntimeError> { cps(list, Env::new_root()?) }

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
        assert_eq!(exec(List::from_vec(i)).unwrap(), Value::Procedure(Procedure::Native("+")));
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
}
