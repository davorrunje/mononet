#!/usr/bin/env bash
# Default (CPU) devcontainer: all backends, CPU-only torch (via the cpu wheel
# index) + dev dependencies. No CUDA/nvidia wheels.
set -euo pipefail

export MONONET_EXTRAS="all-cpu"
bash .devcontainer/shared/install_dependencies.sh
