#!/bin/bash

# Exit immediately if any command fails
set -e

# Set the directory to the benchmark folder
benchmark_dir="basic_benchmark"

# Check if the directory exists
if [ ! -d "$benchmark_dir" ]; then
    echo "Directory $benchmark_dir does not exist."
    exit 1
fi

# Find and delete all .json files except config_params.json
find "$benchmark_dir" -type f -name "*.json" ! -name "config_params.json" -exec rm -f {} \;

# Find and delete all .png files
find "$benchmark_dir" -type f -name "*.png" -exec rm -f {} \;

# Output message
echo "Deleted all JSON files (except config_params.json) and all PNG files from $benchmark_dir."