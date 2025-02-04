_default:
  just --list

watch-test:
  watchexec -w . -e rs -r -c -- cargo test

repl mode="cps":
  cargo run -- --mode {{mode}}

eval expr="(+ 1 2)":
  cargo run -- --mode cps --eval '{{expr}}' --log-file /tmp/scheme.log

eval-ast expr="(+ 1 2)":
  cargo run -- --mode ast --eval '{{expr}}' --log-file /tmp/scheme.log

test-scm:
  cargo run -- examples/printing.scm
  cargo run -- examples/tail_call.scm
  cargo run -- examples/threads.scm
