#!/bin/bash

# Script to generate Doxygen documentation for Psyne

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Generating Doxygen documentation for Psyne..."
echo "============================================"

cd "$PROJECT_ROOT"

# Create output directory if it doesn't exist
mkdir -p docs/doxygen

# Check if doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "Error: Doxygen is not installed!"
    echo "Please install it with: sudo apt-get install doxygen graphviz"
    exit 1
fi

# Generate documentation
echo "Running Doxygen..."
doxygen Doxyfile

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Documentation generated successfully!"
    echo "Open docs/doxygen/html/index.html in your browser to view."
    
    # Optional: Open in default browser
    if command -v xdg-open &> /dev/null; then
        read -p "Open documentation in browser? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            xdg-open docs/doxygen/html/index.html
        fi
    fi
else
    echo "Error: Documentation generation failed!"
    exit 1
fi