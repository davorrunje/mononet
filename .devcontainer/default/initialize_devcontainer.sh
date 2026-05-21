#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval $(op signin)

# Optional secrets that may not exist in every user's vault.
# Check each one individually and blank out only the missing ones before running op inject.
# Add any optional secret variable names here (e.g. "LINEAR_API_KEY" "GITHUB_TOKEN").
OPTIONAL_SECRETS=("LINEAR_API_KEY" "GITHUB_TOKEN")

cp "$SCRIPT_DIR/devcontainer.env" "$SCRIPT_DIR/devcontainer.env.bak"

for var_name in "${OPTIONAL_SECRETS[@]}"; do
  # Extract the op:// reference for this variable from devcontainer.env
  op_ref=$(grep "^${var_name}=" "$SCRIPT_DIR/devcontainer.env" | head -1 | cut -d'=' -f2-)
  if [[ "$op_ref" == op://* ]]; then
    if ! op read "$op_ref" > /dev/null 2>&1; then
      echo -e "\033[1;33mWARNING: ${var_name} not found in 1Password vault. Setting to empty.\033[0m"
      sed -i '' "s|^${var_name}=.*$|${var_name}=|" "$SCRIPT_DIR/devcontainer.env"
    fi
  fi
done

# Read env vars mentioned in devcontainer.env from 1Password
op inject --force --in-file "$SCRIPT_DIR/devcontainer.env" --out-file "$SCRIPT_DIR/devcontainer.env.tmp"

# Restore the original devcontainer.env
mv "$SCRIPT_DIR/devcontainer.env.bak" "$SCRIPT_DIR/devcontainer.env"
