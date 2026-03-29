#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Updating system package list..."
sudo apt update

echo "Installing essential build tools..."
sudo apt install -y build-essential cmake git

echo "Installing PostgreSQL development libraries (libpq and libpqxx)..."
sudo apt install -y libpq-dev libpqxx-dev

echo "Installing nlohmann/json library..."
sudo apt install -y nlohmann-json-dev

echo "Cloning and installing ACORN..."
# Clone the ACORN_lib repository if not already present
if [ ! -d "ACORN" ]; then
    git clone https://github.com/stanford-futuredata/ACORN.git
fi

cd ACORN_lib

# Build and install ACORN_lib
echo "Building ACORN..."
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build
make -C build -j faiss
cd ..

echo "All dependencies installed successfully!"

echo "Verifying installed versions:"
echo -n "GCC version: "
gcc --version | head -n 1

echo -n "CMake version: "
cmake --version | head -n 1

echo "PostgreSQL client libraries installed."
echo "ACORN installed to /usr/local/lib."