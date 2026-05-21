#!/usr/bin/env bash

# Auto-detect the devcontainer directory
DEVCONTAINER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHARED_DIR="$(dirname "$DEVCONTAINER_DIR")/shared"
JSON_VARS_DIR="$DEVCONTAINER_DIR/json_variables"

# Use the shared script to setup JSON environment variables
source "$SHARED_DIR/setup_json_env_vars.sh" "$JSON_VARS_DIR"
