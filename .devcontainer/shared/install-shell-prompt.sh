#!/usr/bin/env bash
# Wire shared/shell-prompt.sh into the container user's shell rc files.
#
# Idempotent: safe to re-run. The home directory is not a persisted volume,
# so this runs from post-create.sh on every container (re)creation.
set -euo pipefail

PROMPT_FILE="/workspaces/mononet/.devcontainer/shared/shell-prompt.sh"
MARKER="# >>> mononet devcontainer prompt >>>"

for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    [ -e "$rc" ] || touch "$rc"
    if grep -qF "$MARKER" "$rc"; then
        continue
    fi
    cat >> "$rc" <<EOF

$MARKER
[ -r "$PROMPT_FILE" ] && . "$PROMPT_FILE"
# <<< mononet devcontainer prompt <<<
EOF
    echo ">>> installed git-aware prompt in $rc"
done
