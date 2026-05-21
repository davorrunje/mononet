#!/usr/bin/env bash

# Setup PATH modifications
# Prepend /usr/local/bin to PATH to prioritize UV-installed tools over py-utils tools
# This ensures 'mypy' and other tools use the versions specified in pyproject.toml

echo -e "\033[32mConfiguring PATH...\033[0m"
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
echo -e "\033[32m✓ PATH configured\033[0m"
