#!/usr/bin/env bash
# Build the Lake project, fetching the mathlib4 cache first.
set -euo pipefail
cd "$(dirname "$0")/.."
lake exe cache get
lake build
