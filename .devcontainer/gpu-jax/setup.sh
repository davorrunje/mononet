#!/usr/bin/env bash
# GPU (JAX) devcontainer: install jax-gpu extra + dev/docs/lint.
set -euo pipefail

export MONONET_EXTRAS="jax-gpu"
bash .devcontainer/shared/install_dependencies.sh
