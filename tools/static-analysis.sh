#!/bin/bash
set -e

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
