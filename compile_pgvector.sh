#!/bin/bash
# Script to compile and install pgvector in debug mode
# Run this after making changes to pgvector source code

echo "Compiling pgvector in debug mode..."
cd pgvector
make CFLAGS="-g -O0 -fno-omit-frame-pointer"

if [ $? -eq 0 ]; then
    echo ""
    echo "Installing pgvector..."
    sudo make install

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ pgvector compiled and installed successfully!"
        echo ""
        echo "To apply changes, restart PostgreSQL:"
        echo "  sudo service postgresql restart"
        echo ""
        echo "Or reload the extension in your database:"
        echo "  psql -U xx -d rbacdatabase_treebase -c 'DROP EXTENSION IF EXISTS vector CASCADE; CREATE EXTENSION vector;'"
    else
        echo ""
        echo "✗ Installation failed."
    fi
else
    echo ""
    echo "✗ Compilation failed. Please check the error messages above."
fi
