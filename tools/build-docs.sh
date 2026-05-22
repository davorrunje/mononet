#!/usr/bin/env bash
set -e
set -x

# During Phase 2 the source dir is still docs/docs (Task 7 moves content up).
if [[ -d docs/docs ]]; then
  uv run sphinx-build -c docs -W docs/docs docs/_build/html
else
  uv run sphinx-build -W docs docs/_build/html
fi
