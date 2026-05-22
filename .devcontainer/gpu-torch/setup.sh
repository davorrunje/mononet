#!/usr/bin/env bash
# GPU (PyTorch) devcontainer: install torch-gpu extra + dev/docs/lint.
set -euo pipefail

export MONONET_EXTRAS="torch-gpu"
bash .devcontainer/shared/install_dependencies.sh
