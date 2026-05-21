#!/bin/bash
set -e

echo "Running mypy..."
uv run mypy

echo "Running bandit..."
uv run bandit -c pyproject.toml -r mononet

echo "Running semgrep..."
uv run semgrep scan --config auto --error
