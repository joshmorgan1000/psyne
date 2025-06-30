#!/bin/bash

# TCP Examples Build Script
# Builds the new TCP channel examples separately from the main CMake build
# This allows testing the TCP implementation independently

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo -e "${BLUE}🚀 Psyne TCP Channel Examples Builder${NC}"
echo -e "${BLUE}=====================================${NC}"
echo

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
BUILD_DIR="${PROJECT_ROOT}/build/examples/tcp"

print_status "Script directory: $SCRIPT_DIR"
print_status "Project root: $PROJECT_ROOT"
print_status "Build directory: $BUILD_DIR"

# Check if we're in the right directory
if [ ! -f "${SCRIPT_DIR}/tcp_basic_demo.cpp" ]; then
    print_error "TCP example files not found in ${SCRIPT_DIR}"
    print_error "Please run this script from the examples/networking directory"
    exit 1
fi

# Check if psyne header exists
PSYNE_HEADER="${PROJECT_ROOT}/include/psyne/psyne.hpp"
if [ ! -f "$PSYNE_HEADER" ]; then
    print_error "Psyne header not found at $PSYNE_HEADER"
    exit 1
fi

print_success "Found Psyne header at $PSYNE_HEADER"

# Create build directory
print_status "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Compiler configuration
CXX=${CXX:-g++}
CXXFLAGS="-std=c++20 -Wall -Wextra -O2 -pthread"
INCLUDES="-I${PROJECT_ROOT}/include"
LIBS="-lboost_system -lboost_thread"

print_status "Compiler: $CXX"
print_status "Flags: $CXXFLAGS"
print_status "Includes: $INCLUDES"
print_status "Libraries: $LIBS"
echo

# Build function
build_example() {
    local name=$1
    local source="${SCRIPT_DIR}/${name}.cpp"
    local target="${BUILD_DIR}/${name}"
    
    print_status "Building ${name}..."
    
    if [ ! -f "$source" ]; then
        print_error "Source file not found: $source"
        return 1
    fi
    
    if $CXX $CXXFLAGS $INCLUDES "$source" -o "$target" $LIBS 2>&1; then
        print_success "${name} built successfully"
        return 0
    else
        print_error "Failed to build ${name}"
        return 1
    fi
}

# Build all examples
EXAMPLES=("tcp_basic_demo" "tcp_multimode_demo" "tcp_performance_demo")
BUILD_SUCCESS=0
BUILD_TOTAL=${#EXAMPLES[@]}

for example in "${EXAMPLES[@]}"; do
    if build_example "$example"; then
        ((BUILD_SUCCESS++))
    fi
    echo
done

# Report results
echo -e "${BLUE}📊 Build Summary${NC}"
echo "================"
echo "Successfully built: $BUILD_SUCCESS/$BUILD_TOTAL examples"

if [ $BUILD_SUCCESS -eq $BUILD_TOTAL ]; then
    print_success "All TCP examples built successfully!"
    echo
    echo -e "${GREEN}🏃 Ready to run examples:${NC}"
    echo "  Basic Demo:        ./${EXAMPLES[0]} [server|client]"
    echo "  Multi-mode Demo:   ./${EXAMPLES[1]} [server|client] [spsc|mpsc|spmc|mpmc]"
    echo "  Performance Demo:  ./${EXAMPLES[2]} [server|client]"
    echo
else
    print_warning "Some examples failed to build"
fi

# Check for quick test option
if [ "$1" = "--test" ] || [ "$1" = "-t" ]; then
    if [ $BUILD_SUCCESS -eq 0 ]; then
        print_error "No examples built successfully, cannot run tests"
        exit 1
    fi
    
    echo -e "${BLUE}🧪 Running Quick Tests${NC}"
    echo "====================="
    echo
    
    # Test basic TCP functionality
    if [ -f "./tcp_basic_demo" ]; then
        print_status "Testing basic TCP demo (30 second test)..."
        
        # Start server in background
        ./tcp_basic_demo server &
        SERVER_PID=$!
        
        # Give server time to start
        sleep 3
        
        # Check if server is running
        if kill -0 $SERVER_PID 2>/dev/null; then
            print_success "TCP server started (PID: $SERVER_PID)"
            
            # Run client with timeout
            if timeout 30 ./tcp_basic_demo client; then
                print_success "Basic TCP test completed successfully"
            else
                print_warning "Basic TCP test timed out or failed"
            fi
            
            # Clean up server
            kill $SERVER_PID 2>/dev/null || true
            wait $SERVER_PID 2>/dev/null || true
        else
            print_error "Failed to start TCP server for testing"
        fi
        echo
    fi
    
    # Test SPSC mode if multimode demo built
    if [ -f "./tcp_multimode_demo" ]; then
        print_status "Testing SPSC mode (60 second test)..."
        
        ./tcp_multimode_demo server spsc &
        SERVER_PID=$!
        
        sleep 3
        
        if kill -0 $SERVER_PID 2>/dev/null; then
            print_success "SPSC server started (PID: $SERVER_PID)"
            
            if timeout 60 ./tcp_multimode_demo client spsc; then
                print_success "SPSC mode test completed successfully"
            else
                print_warning "SPSC mode test timed out or failed"
            fi
            
            kill $SERVER_PID 2>/dev/null || true
            wait $SERVER_PID 2>/dev/null || true
        else
            print_error "Failed to start SPSC server for testing"
        fi
        echo
    fi
    
    print_success "Quick tests completed!"
    echo
fi

# Show usage information
if [ "$1" != "--test" ] && [ "$1" != "-t" ]; then
    echo -e "${YELLOW}💡 Usage Tips:${NC}"
    echo "============="
    echo
    echo "🔧 Build only:    $0"
    echo "🧪 Build + test:  $0 --test"
    echo
    echo "📖 Manual testing:"
    echo "  Run examples in separate terminals:"
    echo
    echo "  Example 1 - Basic Demo:"
    echo "    Terminal 1: cd $BUILD_DIR && ./tcp_basic_demo server"
    echo "    Terminal 2: cd $BUILD_DIR && ./tcp_basic_demo client"
    echo
    echo "  Example 2 - SPSC Mode:"
    echo "    Terminal 1: cd $BUILD_DIR && ./tcp_multimode_demo server spsc"
    echo "    Terminal 2: cd $BUILD_DIR && ./tcp_multimode_demo client spsc"
    echo
    echo "  Example 3 - Performance Test:"
    echo "    Terminal 1: cd $BUILD_DIR && ./tcp_performance_demo server"
    echo "    Terminal 2: cd $BUILD_DIR && ./tcp_performance_demo client"
    echo
fi

if [ $BUILD_SUCCESS -lt $BUILD_TOTAL ]; then
    exit 1
fi