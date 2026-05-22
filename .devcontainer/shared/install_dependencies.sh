#!/usr/bin/env bash
# Backend-agnostic dependency install. Per-flavor setup.sh exports
# MONONET_EXTRAS before calling this (defaults to 'all' for CPU flavor).
set -euo pipefail

cd /workspaces/mononet

bash .devcontainer/shared/install_common_tools.sh

# Dev/docs/lint groups are pulled in automatically via
# [tool.uv] default-groups in pyproject.toml.
EXTRAS="${MONONET_EXTRAS:-all}"
echo ">>> uv sync --extra ${EXTRAS}"
uv sync --extra "${EXTRAS}"
