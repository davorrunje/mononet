#!/usr/bin/env bash
# Re-execute all benchmark notebooks in place.
# Run from a gpu-* devcontainer so the timings reflect real hardware.
# Intended to be run manually before tagging a release; see
# CONTRIBUTING.md "Release process".
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Ensure the bench group is installed.
uv sync --group bench

# The paper's original PyTorch implementation is installed without its
# own dependency metadata, because its old typing-extensions pin
# conflicts with modern Python+TF. It is only imported for direct
# numerical comparison inside the notebooks; its own deps are not
# needed at runtime.
echo ">>> installing airtai/monotonic-nn (no-deps) for paper-baseline comparison"
uv pip install --no-deps "git+https://github.com/airtai/monotonic-nn"

echo ">>> executing rendered benchmark notebooks under docs/benchmarks/"
# NOTE: the Phase-2a SEARCH notebooks (benchmarks/notebooks/) are NOT executed here —
# run them via the CLI: tools/mononet-benchmark-search (see benchmarks/README.md).
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=14400 \
  docs/benchmarks/*.ipynb docs/benchmarks/paper-reproduction/*.ipynb

echo
echo ">>> done. Review diffs:"
echo "    git diff docs/benchmarks/"
echo
echo ">>> then commit and tag a release."
