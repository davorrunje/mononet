#!/usr/bin/env bash

# Install Python dependencies using UV
# This script handles dependency installation and pre-commit hooks setup

set -e  # Exit on error

echo -e "\033[36m=== Installing Python Dependencies ===\033[0m"

if [ -n "${UV_INDEX_SYNTHPOP_PKGS_PASSWORD}" ]; then
    uv pip install --system -e "." --group={dev,devdocs,lint}
    echo -e "\033[32mDependencies have been successfully installed using UV.\033[0m"

    # install pre-commit hooks
    pre-commit install
    echo -e "\033[32m✓ Pre-commit hooks installed\033[0m"

else
    echo -e "\033[33mWARNING: UV_INDEX_SYNTHPOP_PKGS_PASSWORD is not set. Dependencies cannot be installed automatically.\033[0m"
    echo -e "\033[33mPlease set UV_INDEX_SYNTHPOP_PKGS_PASSWORD and then reopen the devcontainer.\033[0m"

    # Add warning to .zshrc
    echo "echo -e \"\033[33mWARNING: UV_INDEX_SYNTHPOP_PKGS_PASSWORD is not set. Dependencies cannot be installed automatically.\033[0m\"" >> ~/.zshrc
    echo "echo -e \"\033[33mPlease set UV_INDEX_SYNTHPOP_PKGS_PASSWORD and reopen the devcontainer.\033[0m\"" >> ~/.zshrc
fi
