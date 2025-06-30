#!/bin/bash

# Psyne Language Bindings Build Script
# Builds all available language bindings for Psyne

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

echo -e "${BLUE}Psyne Language Bindings Build Script${NC}"
echo -e "${BLUE}====================================${NC}\n"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "SKIP" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
    fi
}

# Function to build binding with error handling
build_binding() {
    local lang=$1
    local build_cmd=$2
    local check_cmd=$3
    
    echo -e "\n${BLUE}Building $lang bindings...${NC}"
    
    if [ -n "$check_cmd" ] && ! eval "$check_cmd"; then
        print_status "SKIP" "$lang - Required dependencies not found"
        return 0
    fi
    
    cd "$SCRIPT_DIR/$lang"
    
    if eval "$build_cmd"; then
        print_status "OK" "$lang bindings built successfully"
        return 0
    else
        print_status "FAIL" "$lang bindings failed to build"
        return 1
    fi
}

# Track build results
SUCCESSFUL_BUILDS=()
FAILED_BUILDS=()
SKIPPED_BUILDS=()

# Ensure Psyne C++ library is built first
echo -e "${BLUE}Building Psyne C++ library...${NC}"
cd "$PROJECT_ROOT"
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
if cmake .. && cmake --build . --target psyne; then
    print_status "OK" "Psyne C++ library built"
else
    print_status "FAIL" "Failed to build Psyne C++ library - bindings may fail"
fi

cd "$SCRIPT_DIR"

# Python bindings
if command_exists python3 && command_exists pip3; then
    if build_binding "python" "pip3 install -e ." "python3 -c 'import pybind11'"; then
        SUCCESSFUL_BUILDS+=("Python")
    else
        FAILED_BUILDS+=("Python")
    fi
else
    SKIPPED_BUILDS+=("Python - python3/pip3 not found")
fi

# Rust bindings
if command_exists cargo; then
    if build_binding "rust" "cargo build --release" ""; then
        SUCCESSFUL_BUILDS+=("Rust")
    else
        FAILED_BUILDS+=("Rust")
    fi
else
    SKIPPED_BUILDS+=("Rust - cargo not found")
fi

# Go bindings
if command_exists go; then
    if build_binding "go" "go build -o psyne.so -buildmode=c-shared ." ""; then
        SUCCESSFUL_BUILDS+=("Go")
    else
        FAILED_BUILDS+=("Go")
    fi
else
    SKIPPED_BUILDS+=("Go - go not found")
fi

# JavaScript/Node.js bindings
if command_exists npm; then
    if build_binding "javascript" "npm install && npm run build" "command_exists node"; then
        SUCCESSFUL_BUILDS+=("JavaScript")
    else
        FAILED_BUILDS+=("JavaScript")
    fi
else
    SKIPPED_BUILDS+=("JavaScript - npm not found")
fi

# Java bindings
if command_exists gradle; then
    if build_binding "java" "gradle build" "command_exists javac"; then
        SUCCESSFUL_BUILDS+=("Java")
    else
        FAILED_BUILDS+=("Java")
    fi
else
    SKIPPED_BUILDS+=("Java - gradle not found")
fi

# C# bindings
if command_exists dotnet; then
    if build_binding "csharp" "dotnet build src/Psyne/Psyne.csproj" ""; then
        SUCCESSFUL_BUILDS+=("C#")
    else
        FAILED_BUILDS+=("C#")
    fi
else
    SKIPPED_BUILDS+=("C# - dotnet not found")
fi

# Swift bindings
if command_exists swift; then
    if build_binding "swift" "swift build" ""; then
        SUCCESSFUL_BUILDS+=("Swift")
    else
        FAILED_BUILDS+=("Swift")
    fi
else
    SKIPPED_BUILDS+=("Swift - swift not found")
fi

# Julia bindings
if command_exists julia; then
    if build_binding "julia" "julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'" ""; then
        SUCCESSFUL_BUILDS+=("Julia")
    else
        FAILED_BUILDS+=("Julia")
    fi
else
    SKIPPED_BUILDS+=("Julia - julia not found")
fi

# Summary
echo -e "\n${BLUE}Build Summary${NC}"
echo -e "${BLUE}=============${NC}"

if [ ${#SUCCESSFUL_BUILDS[@]} -gt 0 ]; then
    echo -e "\n${GREEN}Successfully built (${#SUCCESSFUL_BUILDS[@]}):${NC}"
    for lang in "${SUCCESSFUL_BUILDS[@]}"; do
        echo -e "  ${GREEN}✓${NC} $lang"
    done
fi

if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    echo -e "\n${RED}Failed to build (${#FAILED_BUILDS[@]}):${NC}"
    for lang in "${FAILED_BUILDS[@]}"; do
        echo -e "  ${RED}✗${NC} $lang"
    done
fi

if [ ${#SKIPPED_BUILDS[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}Skipped (${#SKIPPED_BUILDS[@]}):${NC}"
    for reason in "${SKIPPED_BUILDS[@]}"; do
        echo -e "  ${YELLOW}⚠${NC} $reason"
    done
fi

# Exit with appropriate code
if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    echo -e "\n${RED}Some bindings failed to build. Check the output above for details.${NC}"
    exit 1
else
    echo -e "\n${GREEN}All available bindings built successfully!${NC}"
    exit 0
fi