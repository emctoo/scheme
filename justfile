_default:
  just --list

watch:
  watchexec -w . -e rs -r -c -- cargo test

repl mode="cps":
  cargo run -- -t {{mode}}

