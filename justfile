_default:
  just --list

watch:
  watchexec -w . -e rs -r -c -- cargo test

repl mode="cps":
  cargo run -- --mode {{mode}}

test-scm:
  cargo run -- examples/printing.scm
  cargo run -- examples/tail_call.scm
  cargo run -- examples/threads.scm
