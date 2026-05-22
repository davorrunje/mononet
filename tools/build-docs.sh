#!/usr/bin/env bash
set -e
set -x

uv run sphinx-build docs docs/_build/html
