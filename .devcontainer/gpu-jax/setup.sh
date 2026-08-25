#!/usr/bin/env bash
# GPU (JAX) devcontainer: install jax-gpu extra + dev/docs/lint.
set -euo pipefail

export MONONET_EXTRAS="jax-gpu"
bash .devcontainer/shared/install_dependencies.sh

# --- cuSPARSE / CUDA loader fix -------------------------------------------
# jaxlib's cuda12 plugin (CUDA 12.9) must load the pip-installed nvidia-*-cu12
# libraries, but the base image ships a system CUDA 12.8 toolkit on the default
# loader path (ldconfig), whose older libcusparse.so.12 shadows the pip one —
# jax then fails with "Unable to load cuSPARSE" and silently falls back to CPU.
# Fix: prepend the venv's pip nvidia lib dirs to LD_LIBRARY_PATH for every shell.
# Globbed at shell-init so it tracks the actual python/site-packages layout.
sudo tee /etc/profile.d/10-mononet-jax-cuda.sh >/dev/null <<'PROFILE'
# Put pip CUDA 12.9 libs ahead of the system CUDA 12.8 toolkit for jaxlib.
_nvlibs=$(echo /workspaces/mononet/.venv/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr ' ' ':')
if [ -n "$_nvlibs" ] && [ "$_nvlibs" != "/workspaces/mononet/.venv/lib/python*/site-packages/nvidia/*/lib" ]; then
  export LD_LIBRARY_PATH="${_nvlibs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _nvlibs
PROFILE
