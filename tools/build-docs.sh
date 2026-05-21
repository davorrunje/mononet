#!/usr/bin/env bash

set -e
set -x

cd docs; uv run python docs.py build
