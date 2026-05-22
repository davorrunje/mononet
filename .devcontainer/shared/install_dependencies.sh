#!/usr/bin/env bash
# Backend-agnostic dependency install. Per-flavor setup.sh exports
# MONONET_EXTRAS before calling this (defaults to 'all' for CPU flavor).
set -euo pipefail

cd /workspaces/mononet

EXTRAS="${MONONET_EXTRAS:-all}"
echo ">>> installing mononet[${EXTRAS}] with dev + docs + lint groups"
uv pip install --system -e ".[${EXTRAS}]" --group=dev --group=docs --group=lint
