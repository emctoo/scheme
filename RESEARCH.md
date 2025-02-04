
## Actual todo

** DONE See if I can get the tokenizer to return lifetime strs instead of Strings -> NOPE http://www.reddit.com/r/rust/comments/28waq4/how_do_i_create_and_return_a_str_dynamically/
** DONE Parse to AST!
** DONE Add functions
** DONE Add nil
** DONE Print result to string
** DONE Implement set!
** DONE Conditionals
** DONE Boolean logic/operations
** DONE Symbols
** DONE Add quote operator
** DONE Add quote syntax sugar
** DONE Add quasiquote/unquote operators
** DONE Add quasiquote/unquote syntax sugar
** DONE Lists as data
** DONE Apply
** DONE Eval
** DONE REPL
** DONE Add up/down arrows to REPL
** TODO Unify AST and Intepreter enums? -> hard, have to include procedures somehow :(
** DONE Print statement
** DONE Run files from file system
** DONE Let expressions
** DONE Shortcut syntax for defining functions (define (myfunc x) (+ x x))
** DONE Try to replace most for loops with iterators
** TODO See if I can internalize the RefCell contract and expose something simpler for Envirnoment (probably not)
** TODO Tab completion in REPL (based on defined functions and constants, and maybe even local vars?)
** DONE Add macros
** TODO Hygenic macros
** TODO call/cc (implement with workers? (probably not possible) or manual stack/instruction pointer?)
** TODO Bytecode VM (stack, or register based? -> stack is probably easier)
** TODO JIT

## Unimplemented/maybe TODO

** TODO Floats
** TODO Ecaping doubles quotes and backslashes in strings
** TODO Restricting non-global defines? (seems like there's mixed implementations on this, but should at least be conistent)
** TODO Tail call optimization
** TODO Nested quasiquotes
** TODO unquote-splicing in quasiquote
** TODO quote-syntax

* Interpreters: Existing languages
** Ruby 1.8: normal interpreter, no precompilation, no VM.
** Ruby 1.9: no precompilation, compiles to bytecode, runs on VM (YARV), VM is interpreted.
** JVM: precompiles to bytecode, runs on VM, VM is interpreted with a tracing JIT (or static JIT? depends in the VM?)
** V8: no precomplation, no VM, no interpreter, static JIT ("full compiler") compiles JS to machine code when it's run for the first time, tracing JIT ("optimizing compiler") watches for hot functions and re-compiles with assumptions & guards baked in, and backs out to static JIT if it breaks. Both JITs are stack machines.
** Firefox: no precompilation, compiles to bytecode, VM interpreter runs, then first tracing "baseline" JIT, then second optimizing tracing JIT ("Ion") kicks in. VM and JITs are all stack machines, VM interpreter stack and JITs native C stacks.
** Safari: no precompilation, VM interpreter, first tracing JIT, second optimizing JIT. Both are register machines, not stack machines. Or actually, maybe most platforms ship with interpreter turned off, so it's just a baseline JIT and an optimizing JIT, like V8 but operating on intermediate bytecode.
** Rust: precompiled to machine code (obviously, I guess).
** Python (CPython): no precompilation, compiles to bytecode on first run, VM & VM interpreter.
** PyPy: JIT'ed interpreter written in RPython.

## Interpreters: My options

** Static compilation (generate machine code statically)
** Vanilla interpreter
** Static JIT (generate machine code on first run)
** Vanilla interpreter + tracing JIT (profile & generate machine code for hot loops/functions)
** Bytecode + VM interpreter
** Bytecode + VM w/ static JIT (generate machine code on first execution of each operation)
** Bytecode + VM w/ interpreter & tracing JIT (profile & generate machine code for hot loops)
** JVM bytecode compiler (or other VM to target)
** LLVM backend compiler
** Plan: do the non-machine code ones first (vanilla interpreter, bytecode + VM interpreter), then try static compilation, then static JITs, then tracing JITs? Or if it's too hard to do a full static compile, just do VM interpreter + tracing JIT, as that's probably the least amount of machine code.

## Resources

https://web.archive.org/web/20200212080133/http://home.pipeline.com/~hbaker1/
http://blog.reverberate.org/2012/12/hello-jit-world-joy-of-simple-jits.html?m=1

https://www.cs.cornell.edu/~asampson/blog/flattening.html

## tests

https://www.reddit.com/r/scheme/comments/j0ws45/any_existing_set_of_small_tests_for_scheme/

https://github.com/jcubic/lips/tree/master/tests
https://github.com/ashinn/chibi-scheme/blob/master/tests/r5rs-tests.scm
https://github.com/ashinn/chibi-scheme/blob/master/tests/r7rs-tests.scm
https://git.savannah.gnu.org/cgit/guile.git/tree/test-suite/tests
https://github.com/alaricsp/chicken-scheme/tree/master/tests
https://github.com/edeproject/edelib/blob/master/test/scheme/r5rs.ss
https://github.com/edeproject/edelib/blob/master/test/scheme/tiny-clos.ss
http://t3x.org/s9fes/test.scm.html

## various links

https://github.com/ytakano/blisp
https://github.com/udem-dlteam/ribbit
https://github.com/raviqqe/stak
https://spritely.institute/hoot/
https://scheme.fail/

https://github.com/schemedoc/cookbook
https://www.gnu.org/software/guile/manual/html_node/Continuation_002dPassing-Style.html

https://ecraven.github.io/r7rs-benchmarks/

(Write your own tiny programming system(s)!)[https://d3s.mff.cuni.cz/teaching/nprg077/]


## scheme lang

### and

- and 表达式从左到右依次求值其参数。
- 如果任何一个参数的值为 #f（false），and 表达式立即返回 #f，并停止求值后续参数（短路求值）。
- 如果所有参数都不为 #f，and 表达式返回最后一个参数的值。
- 如果没有参数，and 返回 #t（true）。

```scheme
(and #t 1 2) ; 返回 2，因为所有参数都为真，返回最后一个参数的值
(and #t #f 2) ; 返回 #f，因为第二个参数为假，短路求值
(and) ; 返回 #t，空 and 表达式返回真
```

### or
- or 表达式从左到右依次求值其参数。
- 如果任何一个参数的值不为 #f，or 表达式立即返回该参数的值，并停止求值后续参数（短路求值）。
- 如果所有参数都为 #f，or 表达式返回 #f。
- 如果没有参数，or 返回 #f。

```scheme
(or #f 1 2) ; 返回 1，因为第一个参数为假，继续求值第二个参数，返回第一个真值
(or #f #f 2) ; 返回 2，因为第二个参数为假，继续求值第三个参数，返回第一个真值
(or) ; 返回 #f，空 or 表达式返回假
```

### and/or 的实现

`and` 和 `or` 通常被实现为特殊形式(special form)而不是原生函数(native function)，主要有以下几个原因：

1. 短路求值 (Short-circuit evaluation):
- `and` 在遇到第一个 false 值时就应该停止求值后续表达式
- `or` 在遇到第一个非 false 值时就应该停止求值后续表达式

如果实现为原生函数,所有参数都会被预先求值,无法实现短路效果。这可能导致:
- 效率问题 - 不必要的计算
- 正确性问题 - 可能执行本不该执行的副作用
- 错误处理问题 - 可能抛出本可避免的错误

例如:
```scheme
(and #f (error "不该执行"))  ; 应该返回 #f,不该抛错
(or #t (error "不该执行"))   ; 应该返回 #t,不该抛错
```

2. 返回值处理:
- `and` 要返回最后一个求值的表达式的值
- `or` 要返回第一个非 false 值

如果作为原生函数,无法获取到原始表达式,只能拿到求值后的结果,会失去这种语义。

## typed impl

```rust
#[derive(Clone)]
pub struct TypedValue<S> {
    value: Value,
    _marker: PhantomData<S>
}
```

`_marker: PhantomData<S>` 的作用是为了在类型系统中引入类型参数 S，而不实际使用这个参数存储任何数据。

### 类型参数必须被使用

在 Rust 中，如果你声明了一个类型参数但没有在类型定义中使用它，编译器会报错：

```rust
// 这样写会编译错误
struct TypedValue<S> {
    value: Value,  // S 没有被使用
}
```

编译器会提示 "unused type parameter"，因为 S 虽然在类型名中声明了，但实际上并没有在结构体的字段中使用。

### PhantomData 的作用

PhantomData 是一个零大小的类型(zero-sized type)，它告诉编译器"我们在逻辑上使用了这个类型参数"：

```rust
struct TypedValue<S> {
    value: Value,
    _marker: PhantomData<S>  // 标记我们在逻辑上使用了 S
}
```

### 为什么需要这样做？

```rust
// 不同的状态类型
struct Evaluating;
struct Applying;

// TypedValue 用来标记值处于哪个状态
let eval_value: TypedValue<Evaluating> = TypedValue::new(some_value);
let apply_value: TypedValue<Applying> = TypedValue::new(other_value);

// 编译器会阻止错误的状态转换
fn process_evaluating(v: TypedValue<Evaluating>) { /* ... */ }

process_evaluating(eval_value);   // OK
process_evaluating(apply_value);  // 编译错误！
```

如果没有 `PhantomData`：

- `TypedValue<Evaluating>` 和 `TypedValue<Applying>` 在运行时会是完全相同的类型
- 编译器无法区分不同状态的值
- 类型系统无法帮助我们捕获状态转换错误

实际例子

```rust
// 定义状态类型
struct Evaluating;
struct Applying;

struct TypedValue<S> {
    value: Value,
    _marker: PhantomData<S>
}

impl<S> TypedValue<S> {
    // 通用方法，适用于所有状态
    fn new(value: Value) -> Self {
        Self {
            value,
            _marker: PhantomData
        }
    }
}

impl TypedValue<Evaluating> {
    // 只有处于 Evaluating 状态的值才能调用这个方法
    fn start_apply(self) -> TypedValue<Applying> {
        TypedValue {
            value: self.value,
            _marker: PhantomData
        }
    }
}

// 使用示例
let eval_value = TypedValue::<Evaluating>::new(some_value);
let apply_value = eval_value.start_apply();  // OK
// let wrong = eval_value.end_apply();       // 编译错误！
```

### 其他用途

PhantomData 还可以用来：

- 标记所有权关系
- 表达变量的生命周期
- 实现标记trait
- 处理类型的变型(variance)

```rust
// 例如标记所有权
struct Container<T> {
    data: *mut T,
    _owner: PhantomData<T>,  // 表明 Container 拥有 T 类型数据
}
```

### 总结：

- PhantomData 是一个在类型系统层面必要的工具
- 它使得我们可以在不增加运行时开销的情况下使用类型参数
- 在你的状态机设计中，它是实现类型级别状态检查的关键
