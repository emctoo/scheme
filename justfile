_default:
  just --list

watch:
  watchexec -w . -e rs -r -c -- cargo test
