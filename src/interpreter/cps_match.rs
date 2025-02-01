use crate::interpreter::cps::{List, RuntimeError, Value};

#[macro_export]
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
