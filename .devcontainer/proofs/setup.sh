#!/usr/bin/env bash
# Proofs (Lean) devcontainer: install Lean toolchain + mathlib cache,
# plus the lint/docs uv groups so pre-commit hooks run on commits made
# from this container.
#
# Deliberately NOT installing any of mononet's ML extras (torch, jax,
# keras) — the proofs flavor is for Lean review only.
set -euo pipefail

cd /workspaces/mononet

bash .devcontainer/shared/install_common_tools.sh
bash .devcontainer/shared/install_lean.sh

# No --extra flag: skip torch/jax/keras. Only the dev-time groups
# pre-commit needs to run cleanly.
echo ">>> uv sync --only-group lint --only-group docs"
uv sync --only-group lint --only-group docs
