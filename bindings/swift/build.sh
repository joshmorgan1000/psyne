#!/bin/bash

# Build script for Psyne Swift bindings

set -e

echo "Building Psyne Swift Bindings..."

# Check if we're in the correct directory
if [ ! -f "Package.swift" ]; then
    echo "Error: Package.swift not found. Please run this script from the swift bindings directory."
    exit 1
fi

# Build the main Psyne library first (if not already built)
echo "Checking for Psyne C library..."
if [ ! -f "../../build/lib/libpsyne.a" ] && [ ! -f "../../build/lib/libpsyne.so" ]; then
    echo "Psyne C library not found. Building..."
    cd ../..
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd ../../bindings/swift
    echo "Psyne C library built successfully."
else
    echo "Psyne C library found."
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf .build

# Build Swift package
echo "Building Swift package..."
swift build -c release

echo "Running tests..."
swift test

echo "Build completed successfully!"

# Show package structure
echo ""
echo "Package structure:"
find . -name "*.swift" -type f | head -20