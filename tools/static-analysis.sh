#!/bin/bash
set -e

echo "Running mypy..."
# --group bench: mypy type-checks benchmarks/, which imports typer/optuna/etc.
# `uv run` syncs its own venv to [tool.uv] default-groups (no bench), so without
# this the Typer decorator resolves to Any and disallow_untyped_decorators fails.
uv run --group bench mypy

echo "Running bandit..."
uv run bandit -c pyproject.toml -r mononet

echo "Running semgrep..."
uv run semgrep scan --config auto --error
