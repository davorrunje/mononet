#!/usr/bin/env bash

# A script for running mypy in the ambient environment,
# with all its dependencies installed.
#
# The pre-commit hook checks only this interpreter. CI sweeps every supported
# version (tools/typecheck-all.sh, and the `typecheck` matrix job).

set -o errexit

# Change directory to the project root directory.
cd "$(dirname "$0")"/..

./tools/typecheck.sh
