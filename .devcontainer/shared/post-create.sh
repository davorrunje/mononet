#!/usr/bin/env bash
# Run once after the container is created. Common tools and project
# dependencies are installed earlier via updateContentCommand
# (setup.sh -> install_dependencies.sh). This hook only installs
# pre-commit hooks, which write to the .git directory and so must
# wait until the workspace is mounted.
set -euo pipefail

cd /workspaces/mononet

echo ">>> installing pre-commit hooks"
uv run pre-commit install --install-hooks

echo ">>> done"
