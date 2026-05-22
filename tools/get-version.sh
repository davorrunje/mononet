#!/bin/bash
# Extract version from pyproject.toml
# Usage: ./get-version.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYPROJECT_FILE="$PROJECT_ROOT/pyproject.toml"

if [ ! -f "$PYPROJECT_FILE" ]; then
    echo "ERROR: pyproject.toml not found at $PYPROJECT_FILE" >&2
    exit 1
fi

VERSION=$(grep -oP '(?<=^version = ")[^"]*' "$PYPROJECT_FILE")

if [ -z "$VERSION" ]; then
    echo "ERROR: Could not extract version from pyproject.toml" >&2
    exit 1
fi

echo "$VERSION"
