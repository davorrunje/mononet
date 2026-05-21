#!/usr/bin/env bash

# Generic JSON environment variables setup
# This script sets up environment variables from JSON files
# Usage: setup_json_env_vars.sh <json_variables_directory>

set -e  # Exit on error

JSON_VARS_DIR="$1"

if [ -z "$JSON_VARS_DIR" ]; then
    echo -e "\033[31mERROR: JSON variables directory not specified\033[0m"
    exit 1
fi

if [ ! -d "$JSON_VARS_DIR" ]; then
    echo -e "\033[33mWARNING: JSON variables directory does not exist: $JSON_VARS_DIR\033[0m"
    exit 0
fi

# Read all json files in the specified directory and add them to the .zshrc file
for file in "$JSON_VARS_DIR"/*.json; do
    if [ -f "$file" ]; then
        # Extract the variable name from the filename (without extension)
        var_name=$(basename "$file" .json)

        # Create a file reference instead of trying to store the entire content in the environment
        # This avoids issues with large JSON files being truncated
        echo "export $var_name=\"\$(cat '$file')\"" >> ~/.zshrc
        echo "export $var_name=\"\$(cat '$file')\"" >> ~/.bashrc
        echo -e "\033[32mAdded $var_name reference to .zshrc and .bashrc.\033[0m"

        # For the current shell, use a temporary file to handle large JSON content
        json_file_path=$(realpath "$file")
        # Create a temporary script to export the variable and then source it
        tmp_script=$(mktemp)
        echo "export $var_name=\"\$(cat '$json_file_path')\"" > "$tmp_script"
        source "$tmp_script"
        rm "$tmp_script"

        # Verify the variable is set
        if [ -n "${!var_name}" ]; then
            echo -e "\033[32mExported $var_name as an environment variable for the current script.\033[0m"
        else
            echo -e "\033[33mWARNING: Failed to export $var_name for the current script.\033[0m"
        fi
    else
        echo -e "\033[33mWARNING: $file does not exist or is not a file.\033[0m"
    fi
done
