#!/usr/bin/env bash

# Auto-detect the devcontainer directory
DEVCONTAINER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Source the environment variables script instead of running in a subshell
source "$DEVCONTAINER_DIR/setup_env_vars.sh"

rm ${DEVCONTAINER_DIR}/devcontainer.env.tmp
