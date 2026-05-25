#!/usr/bin/env bash

# Common installation script for devcontainer setup
# This script contains all the common tool installations shared across different devcontainers

set -euo pipefail

cd /workspaces/mononet

echo -e "\033[36m=== Installing Common Tools ===\033[0m"

# Conditionally install git-lfs if .gitattributes has LFS entries
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -f "$REPO_ROOT/.gitattributes" ] && grep -qE '^[[:space:]]*[^#[:space:]].*filter=lfs' "$REPO_ROOT/.gitattributes"; then
  echo "[setup.sh] Installing git-lfs..."
  git lfs install --force
  git lfs pull || echo -e "\033[1;33mWARNING: git lfs pull failed (repo may not be committed yet).\033[0m"
  echo -e "\033[32mGit LFS has been installed and artifacts pulled.\033[0m"
fi

# Install uv only when missing.
if command -v uv >/dev/null 2>&1; then
  echo "uv already installed: $(uv --version)"
else
  echo -e "\033[32mInstalling uv...\033[0m"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Add ~/.local/bin (the default uv install location) to interactive shells
  # launched after the install. Only on first install to avoid duplicates.
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.bashrc
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.zshrc 2>/dev/null || true
fi

export PATH="$HOME/.local/bin:$PATH"
echo "uv installed: $(uv --version)"

# Authenticate GitHub CLI. Two token sources, in order:
#   1. $GITHUB_TOKEN (Codespaces auto-injects this).
#   2. /var/run/devcontainer-host-secrets/gh-token (forwarded from the
#      host by .devcontainer/shared/host-init.sh via initializeCommand).
echo -e "\033[32mAuthenticating GitHub CLI (if a token is available)...\033[0m"

HOST_TOKEN_FILE="/var/run/devcontainer-host-secrets/gh-token"
gh_token=""

if [ -n "${GITHUB_TOKEN:-}" ]; then
  gh_token="${GITHUB_TOKEN}"
elif [ -s "${HOST_TOKEN_FILE}" ] && [ -r "${HOST_TOKEN_FILE}" ]; then
  gh_token="$(cat "${HOST_TOKEN_FILE}")"
fi

if [ -n "${gh_token}" ]; then
  if printf '%s' "${gh_token}" | gh auth login --with-token; then
    echo -e "\033[32mGitHub CLI authenticated.\033[0m"
  else
    echo -e "\033[1;33mWARNING: GitHub CLI authentication failed.\033[0m"
  fi
else
  echo -e "\033[1;33mWARNING: No GitHub token available (neither \$GITHUB_TOKEN nor host-forwarded token); gh CLI is unauthenticated.\033[0m"
fi
unset gh_token HOST_TOKEN_FILE

# Install Claude Code via official installer only when missing.
if command -v claude >/dev/null 2>&1; then
  echo "claude already installed: $(claude --version || echo 'version unavailable')"
else
  echo -e "\033[32mInstalling Claude Code...\033[0m"
  # nosemgrep
  curl -fsSL https://claude.ai/install.sh | bash
fi

echo -e "\033[32m✓ Common tools installation completed\033[0m"
