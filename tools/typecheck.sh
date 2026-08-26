#!/bin/bash
# Run mypy, either in the ambient project environment or isolated for one
# supported Python version.
#
#   tools/typecheck.sh          # ambient env; what the pre-commit hook runs
#   tools/typecheck.sh 3.11     # isolated env resolved for 3.11
#
# The isolated form exists because mypy applies --python-version to the grammar
# it parses every file with, including dependency stubs, and the stubs installed
# differ per version: uv.lock resolves numpy 2.4.6 below 3.12 and 2.5.2 at or
# above it, and 2.5.2's stubs use PEP 695 `type` aliases that a 3.11 target
# cannot parse. A --python-version flag alone, in whatever env happens to be
# present, therefore cannot check the other versions.
set -e

cd "$(dirname "$0")"/..

version="${1:-}"

# A present-but-invalid argument must fail loudly: silently falling through to
# the ambient run would report success for a version that was never checked.
if [ "$#" -gt 0 ] && ! printf '%s' "${version}" | grep -qE '^3\.[0-9]+$'; then
  echo "usage: $(basename "$0") [3.X]   (got: '${version}')" >&2
  exit 2
fi

if [ -z "${version}" ]; then
  echo "Running mypy (ambient environment)..."
  exec uv run mypy
fi

# Never the project's own .venv: `uv run --python 3.11` would rebuild .venv on
# 3.11 and destroy the working environment. Kept out of the repo because the
# workspace is a bind mount in every devcontainer flavor, and kept next to the
# uv cache so wheels hardlink instead of copying.
env_root="${MONONET_TYPECHECK_ENV_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/mononet-typecheck}"

echo "Running mypy for Python ${version}..."
# --extra all-cpu: without a framework installed, ignore_missing_imports makes
# every torch/jax/keras symbol resolve to Any, so mono.torch/jax/keras layers
# and benchmarks/_common/model_builder.py are never really type-checked. Only
# the isolated env gets this: the ambient .venv may carry torch-gpu, which
# pyproject.toml's [tool.uv] conflicts table declares mutually exclusive with
# all-cpu, so adding it there would fail to resolve or blow away the ambient env.
UV_PROJECT_ENVIRONMENT="${env_root}/mypy-${version}" \
  exec uv run --extra all-cpu --python "${version}" mypy --python-version "${version}"
