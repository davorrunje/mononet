#!/bin/bash
# Type check every Python version in [project] requires-python.
#
# Runs all versions even after one fails: a sweep that stops at the first
# failure hides how much is broken. Exits non-zero naming the failures.
set -uo pipefail

cd "$(dirname "$0")"/.. || exit 1

mapfile -t versions < <(uv run --script tools/supported_pythons.py)

if [ ${#versions[@]} -eq 0 ]; then
  echo "no supported versions resolved; is [project] requires-python set?" >&2
  exit 1
fi

echo "Type checking ${#versions[@]} versions: ${versions[*]}"

failed=()
for version in "${versions[@]}"; do
  if ! ./tools/typecheck.sh "${version}"; then
    failed+=("${version}")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "" >&2
  echo "mypy FAILED for: ${failed[*]}" >&2
  exit 1
fi

echo ""
echo "mypy passed for: ${versions[*]}"
