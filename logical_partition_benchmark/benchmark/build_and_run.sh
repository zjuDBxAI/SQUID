#!/bin/bash

# Build and run pointer benchmark
# This script compiles and executes the pointer-based HNSW benchmark

set -e  # Exit on error

echo "=== Pointer HNSW Benchmark Build & Run Script ==="
echo ""

# Hint CMake where to look for OpenMP/BLAS pieces on macOS.
if [ -d "/opt/homebrew/opt/libomp" ]; then
    export OpenMP_ROOT="/opt/homebrew/opt/libomp"
elif [ -d "/usr/local/opt/libomp" ]; then
    export OpenMP_ROOT="/usr/local/opt/libomp"
fi
export BLA_VENDOR="Apple"

echo "Using in-tree Faiss (CPU-only build)"
echo ""

# Create build directory
echo "Creating build directory..."
mkdir -p build2
cd build2

# Run CMake
echo "Running CMake..."
cmake ..

# Compile
echo ""
echo "Compiling..."
make -j8

# Check if compilation succeeded
if [ ! -f "pointer_benchmark" ]; then
    echo ""
    echo "❌ Compilation failed - executable not found"
    exit 1
fi

echo ""
echo "✓ Compilation successful"
echo ""

# Run the benchmark
echo "========================================="
echo "Running Pointer HNSW Benchmark"
echo "========================================="
echo ""

./pointer_benchmark

echo ""
echo "========================================="
echo "✓ Benchmark completed"
echo "========================================="
