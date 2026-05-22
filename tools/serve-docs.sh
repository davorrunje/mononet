#!/usr/bin/env bash
set -e
set -x

uv run sphinx-autobuild docs docs/_build/html \
  --port 8008 --host 0.0.0.0 \
  --watch mononet
