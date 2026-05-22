#!/usr/bin/env bash
set -e
set -x

uv run sphinx-build -W docs docs/_build/html
