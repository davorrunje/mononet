#!/usr/bin/env bash
# Guard: the committed equivalence vectors are generated from mononet/core/reference.py
# by tools/regenerate-cases.py, which stamps tests/equivalence/cases/REFERENCE_HASH with
# `git hash-object mononet/core/reference.py`. If reference.py changes without regenerating
# the cases, the committed vectors go stale and the equivalence suite silently validates
# every backend against an outdated ground truth. This hook fails when the stamp no longer
# matches the current reference, so the drift is caught at commit time instead of never.
set -euo pipefail

hash_file="tests/equivalence/cases/REFERENCE_HASH"
expected=$(git hash-object mononet/core/reference.py)
actual=$(tr -d '[:space:]' < "$hash_file")

if [ "$expected" != "$actual" ]; then
    echo "REFERENCE_HASH is stale: mononet/core/reference.py changed without regenerating"
    echo "the equivalence cases."
    echo "  current reference.py: $expected"
    echo "  committed hash:       $actual"
    echo "Run:  uv run python tools/regenerate-cases.py"
    echo "then commit the refreshed tests/equivalence/cases/ (including REFERENCE_HASH)."
    exit 1
fi
