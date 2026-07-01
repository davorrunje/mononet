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
# Pinned to the curated `p/python` pack instead of `--config auto`. `auto` pulls the
# latest community rules from the registry at run time, so newly-added rules (e.g.
# github-actions mutable-tag / dependabot / uv cooldown) silently break unrelated PRs
# and make CI non-reproducible. `p/python` is stable and scoped to this Python
# codebase. Supply-chain hardening flagged by `auto` on the workflow/config files
# (pin action SHAs, avoid curl|sh, add dependency cooldowns) is tracked separately.
uv run semgrep scan --config p/python --error
