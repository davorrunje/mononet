#!/usr/bin/env bash
# Run once after the container is created. Installs the project (uv sync)
# and pre-commit hooks. The per-flavor setup.sh is responsible for
# choosing which extras to install.
set -euo pipefail

cd /workspaces/mononet

# Sync the project's lockfile, including any flavor-specific extras the
# per-flavor setup.sh placed in $MONONET_EXTRAS.
EXTRAS="${MONONET_EXTRAS:-all}"
echo ">>> uv sync --extra ${EXTRAS}"
uv sync --extra "${EXTRAS}"

echo ">>> installing pre-commit hooks"
uv run pre-commit install --install-hooks

echo ">>> done"
