#!/usr/bin/env bash
# Install elan (Lean version manager), let it install the toolchain
# pinned in proofs/lean-toolchain, fetch the mathlib4 build cache, and
# run a smoke build to catch toolchain breakage early.
#
# Idempotent: each step checks before reinstalling. Designed to be
# called from any devcontainer flavor's setup.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo -e "\033[36m=== Installing Lean toolchain ===\033[0m"

# Make ~/.elan/bin reachable for the rest of this script regardless of
# whether elan was installed already or is being installed below.
export PATH="$HOME/.elan/bin:$PATH"

# 1. Install elan only when missing.
if command -v elan >/dev/null 2>&1; then
  echo "elan already installed: $(elan --version 2>&1 | head -1)"
else
  echo -e "\033[32mInstalling elan...\033[0m"
  # --default-toolchain none keeps elan from installing a global toolchain;
  # elan reads proofs/lean-toolchain when invoked from that directory.
  curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none

  # Add ~/.elan/bin to interactive shells. Only on first install to
  # avoid duplicates in the rc files.
  echo 'export PATH="$HOME/.elan/bin:$PATH"' >> /root/.bashrc
  echo 'export PATH="$HOME/.elan/bin:$PATH"' >> /root/.zshrc 2>/dev/null || true
fi
echo "elan: $(elan --version)"

# 2. The toolchain install + smoke build only make sense if the Lean
#    project files are actually on disk (testing the presence of the
#    proofs/lean-toolchain file is more robust than testing the
#    directory itself — `proofs/.lake/` artifact dirs can be left
#    behind by prior work without the source files being there).
if [ ! -f "$REPO_ROOT/proofs/lean-toolchain" ]; then
  echo -e "\033[1;33mproofs/lean-toolchain not present; skipping toolchain install and smoke build.\033[0m"
  echo -e "\033[1;33mThis is expected when this script runs before the Lean proofs PR has merged.\033[0m"
  echo -e "\033[32m✓ elan installed; Lean toolchain install deferred\033[0m"
  exit 0
fi

# 3. Trigger elan to install the toolchain pinned in proofs/lean-toolchain.
#    Any lake command from proofs/ does this lazily; we force it here
#    so the create-time error is clear if the pinned version is gone.
cd "$REPO_ROOT/proofs"
echo "lean: $(lean --version)"

# 4. Fetch the mathlib4 build cache (skip if .lake/build/lib is populated).
if [ -d ".lake/build/lib/Mathlib" ] && [ -n "$(find .lake/build/lib/Mathlib -name '*.olean' -print -quit)" ]; then
  echo "mathlib cache already populated; skipping lake exe cache get"
else
  echo -e "\033[32mFetching mathlib4 build cache...\033[0m"
  lake exe cache get
fi

# 5. Smoke build: confirm everything wired up. Mononet.Basic has no
#    dependencies on the more substantive proof modules, so it builds
#    quickly and validates elan / mathlib / our own files all line up.
echo -e "\033[32mSmoke build: lake build Mononet.Basic\033[0m"
lake build Mononet.Basic

echo -e "\033[32m✓ Lean toolchain ready\033[0m"
