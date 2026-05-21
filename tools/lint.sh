#!/bin/bash
set -e

echo "Running ruff linter (isort, flake, pyupgrade, etc. replacement)..."
uv run ruff check --exit-non-zero-on-fix

echo "Running ruff formatter (black replacement)..."
uv run ruff format
