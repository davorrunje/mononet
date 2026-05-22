#!/usr/bin/env bash
set -e
set -x

if [[ -d docs/docs ]]; then
  uv run sphinx-autobuild -c docs docs/docs docs/_build/html \
    --port 8008 --host 0.0.0.0 \
    --watch mononet
else
  uv run sphinx-autobuild docs docs/_build/html \
    --port 8008 --host 0.0.0.0 \
    --watch mononet
fi
