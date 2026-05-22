#!/usr/bin/env bash
# Default (CPU) devcontainer: install all backends + dev dependencies.
set -euo pipefail

export MONONET_EXTRAS="all"
bash .devcontainer/shared/install_dependencies.sh
