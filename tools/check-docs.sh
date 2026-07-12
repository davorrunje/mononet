#!/usr/bin/env bash
# On-demand documentation health checks: broken links (internal + external)
# and dangling cross-references (nitpicky). NOT wired into the CI gate — see
# docs/superpowers/specs/2026-07-12-docs-audit-design.md.
set -uo pipefail

echo "=== linkcheck (internal + external URLs) ==="
uv run sphinx-build -b linkcheck docs docs/_build/linkcheck
link_status=$?
echo "linkcheck exit: ${link_status}"
echo "(broken/redirected details: docs/_build/linkcheck/output.txt)"

echo
echo "=== nitpicky cross-reference check (warnings only, non-fatal) ==="
uv run sphinx-build -n -b html docs docs/_build/nitpick 2>&1 \
  | grep -iE "WARNING|ERROR" || echo "no nitpicky warnings"
