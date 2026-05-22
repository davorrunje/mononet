#!/usr/bin/env bash
# GPU (Keras) devcontainer: keras + jax-cuda12 by default.
# To use the PyTorch backend instead:
#   1. Re-run with MONONET_KERAS_BACKEND=torch
#   2. Set the container env var KERAS_BACKEND=torch
set -euo pipefail

export MONONET_EXTRAS="keras-gpu"
bash .devcontainer/shared/install_dependencies.sh
