#!/usr/bin/env bash

# Common installation script for devcontainer setup
# This script contains all the common tool installations shared across different devcontainers

set -euo pipefail

cd /workspaces/mononet

echo -e "\033[36m=== Installing Common Tools ===\033[0m"

# Named Docker volumes are initialised root-owned, so the non-root user
# cannot write to them until we take ownership. Every flavor mounts two:
#   - CLAUDE_CONFIG_DIR (~/.claude): needed by the Claude installer
#     (~/.claude/downloads) and plugin provisioning.
#   - /workspaces/mononet/.venv: container-private virtualenv volume,
#     needed by `uv sync`; also isolates it from any host-side .venv.
# Claim both before anything writes to them. Idempotent — only acts on an
# existing, non-writable dir; passwordless sudo is available in all flavors.
for _vol in "${CLAUDE_CONFIG_DIR:-}" /workspaces/mononet/.venv; do
  if [ -n "${_vol}" ] && [ -d "${_vol}" ] && [ ! -w "${_vol}" ]; then
    echo "[setup.sh] Claiming ownership of ${_vol} (root-owned named volume)..."
    sudo chown "$(id -u):$(id -g)" "${_vol}"
  fi
done
unset _vol

# Interactive-shell tooling. The base image is minimized and ships none of
# this:
#   bash-completion  the completion *scripts* under
#                    /usr/share/bash-completion/completions are present but the
#                    loader that sources them is not, so `git sta<TAB>` does
#                    nothing and __git_ps1 is undefined
#   vim              no editor at all -- /usr/bin/editor does not exist, so
#                    `git commit` without -m, `git rebase -i` and interactive
#                    `gh` all fail
#   less             no pager either; git/gh/man dump unpaged to stdout
#   jq tree          routine CLI work (`gh ... --json` piping, listings)
#   fzf htop btop    fuzzy history/file search (Ctrl-R, Ctrl-T), process views
#   nvtop            GPU process view; useful only in the gpu-* flavors, but
#                    installed everywhere to keep one shared tool list
# Only the missing ones are installed, in a single apt-get, so re-runs are
# cheap. Shell wiring for these lives in shared/shell-prompt.sh.
_shell_pkgs=()
[ -r /usr/share/bash-completion/bash_completion ] || _shell_pkgs+=(bash-completion)
for _entry in vim less jq tree fzf htop btop nvtop; do
  command -v "${_entry}" >/dev/null 2>&1 || _shell_pkgs+=("${_entry}")
done
if [ ${#_shell_pkgs[@]} -gt 0 ]; then
  echo -e "\033[32mInstalling shell tooling: ${_shell_pkgs[*]}\033[0m"
  sudo apt-get update &&
    sudo apt-get install -y --no-install-recommends "${_shell_pkgs[@]}"
fi
unset _shell_pkgs _entry

# /etc/dpkg/dpkg.cfg.d/excludes drops /usr/share/doc/* on this image, and
# Debian ships fzf's shell key bindings (Ctrl-R, Ctrl-T, Alt-C) there rather
# than in a sourced location. Pull just those files back in.
if command -v fzf >/dev/null 2>&1 &&
  [ ! -r /usr/share/doc/fzf/examples/key-bindings.bash ]; then
  echo -e "\033[32mRestoring fzf shell key bindings...\033[0m"
  sudo apt-get install -y --reinstall \
    -o DPkg::Options::="--path-include=/usr/share/doc/fzf/examples/*" fzf
fi

# git-lfs — required to pull committed benchmark datasets under benchmarks/data/
if ! command -v git-lfs >/dev/null 2>&1; then
  echo -e "\033[32mInstalling git-lfs...\033[0m"
  sudo apt-get update && sudo apt-get install -y git-lfs
fi
git lfs install --skip-repo
git lfs pull || echo -e "\033[1;33mWARNING: git lfs pull failed (repo may not be committed yet).\033[0m"

# Install uv only when missing.
if command -v uv >/dev/null 2>&1; then
  echo "uv already installed: $(uv --version)"
else
  echo -e "\033[32mInstalling uv...\033[0m"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Add ~/.local/bin (the default uv install location) to interactive shells
  # launched after the install. Only on first install to avoid duplicates.
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc" 2>/dev/null || true
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
