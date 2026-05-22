#!/usr/bin/env bash

# Common installation script for devcontainer setup
# This script contains all the common tool installations shared across different devcontainers

set -e  # Exit on error

echo -e "\033[36m=== Installing Common Tools ===\033[0m"

# Conditionally install git-lfs if .gitattributes has LFS entries
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -f "$REPO_ROOT/.gitattributes" ] && grep -q 'filter=lfs' "$REPO_ROOT/.gitattributes"; then
  echo "[setup.sh] Installing git-lfs..."
  git lfs install --force
  git lfs pull || echo -e "\033[1;33mWARNING: git lfs pull failed (repo may not be committed yet).\033[0m"
  echo -e "\033[32mGit LFS has been installed and artifacts pulled.\033[0m"
fi

# Install UV
echo -e "\033[32mInstalling UV...\033[0m"
curl -LsSf https://astral.sh/uv/install.sh | sh

# Authenticate GitHub CLI (installed via devcontainer feature)
echo -e "\033[32mAuthenticating GitHub CLI...\033[0m"
if [ -n "$GITHUB_TOKEN" ]; then
  echo "$GITHUB_TOKEN" | gh auth login --with-token && echo -e "\033[32mGitHub CLI authenticated.\033[0m" || echo -e "\033[1;33mWARNING: GitHub CLI authentication failed.\033[0m"
else
  echo -e "\033[1;33mWARNING: GITHUB_TOKEN not set; gh CLI installed but not authenticated.\033[0m"
fi

# Install Claude Code via official installer
# nosemgrep
curl -fsSL https://claude.ai/install.sh | bash

echo -e "\033[32m✓ Common tools installation completed\033[0m"
