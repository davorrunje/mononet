#!/usr/bin/env bash

DEVCONTAINER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHARED_DIR="$(dirname "$DEVCONTAINER_DIR")/shared"

bash "$SHARED_DIR/install_common_tools.sh"

export PATH="$HOME/.local/bin:$PATH"

bash "$SHARED_DIR/install_dependencies.sh"
bash "$SHARED_DIR/setup_path.sh"
