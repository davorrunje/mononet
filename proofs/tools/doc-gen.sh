#!/usr/bin/env bash
# Build the doc-gen4 HTML render. Requires `lake -Kenv=dev` config.
set -euo pipefail
cd "$(dirname "$0")/.."
lake -Kenv=dev update doc-gen4
lake -Kenv=dev build Mononet:docs
echo "Output: $(pwd)/.lake/build/doc"
