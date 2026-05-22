#!/usr/bin/env bash
# Install uv on a base image that does not already have it.
# Used by GPU flavors built from nvidia/cuda images.
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  echo "uv already installed: $(uv --version)"
  exit 0
fi

curl -LsSf https://astral.sh/uv/install.sh | sh

# Add ~/.local/bin (the default uv install location) to the shell's PATH
# for interactive sessions launched after the install.
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.zshrc 2>/dev/null || true

echo "uv installed: $($HOME/.local/bin/uv --version)"
