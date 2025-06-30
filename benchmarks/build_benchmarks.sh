#!/bin/bash

# Build script for Psyne benchmarks
# This script builds all benchmarks in the benchmarks directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR/build"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo -e "${BLUE}=== Psyne Benchmarks Build Script ===${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"
echo "Build type: $CMAKE_BUILD_TYPE"
echo "Parallel jobs: $NUM_CORES"
echo

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists cmake; then
    print_error "CMake is required but not installed"
    exit 1
fi

if ! command_exists g++ && ! command_exists clang++; then
    print_error "C++ compiler (g++ or clang++) is required"
    exit 1
fi

# Detect compiler
if command_exists clang++; then
    COMPILER="clang++"
    print_status "Using clang++ compiler"
elif command_exists g++; then
    COMPILER="g++"
    print_status "Using g++ compiler"
fi

# Check if main psyne library is built, and build it if not
MAIN_BUILD_DIR="$PROJECT_ROOT/build"
PSYNE_LIB="$MAIN_BUILD_DIR/libpsyne.so"
if [[ "$OSTYPE" == "darwin"* ]]; then
    PSYNE_LIB="$MAIN_BUILD_DIR/libpsyne.dylib"
fi

if [ ! -f "$PSYNE_LIB" ]; then
    print_status "Main psyne library not found. Building it first..."
    
    # Create main build directory if it doesn't exist
    mkdir -p "$MAIN_BUILD_DIR"
    cd "$MAIN_BUILD_DIR"
    
    # Configure and build main library
    print_status "Configuring main psyne library..."
    cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" "$PROJECT_ROOT"
    
    print_status "Building main psyne library..."
    if cmake --build . --parallel "$NUM_CORES"; then
        # Verify the library was built
        if [ -f "$PSYNE_LIB" ]; then
            print_status "Main psyne library built successfully"
        else
            print_warning "Main psyne library build completed but library file not found"
            print_warning "Will use header-only mode for benchmarks"
        fi
    else
        print_warning "Main psyne library build failed"
        print_warning "Will use header-only mode for benchmarks"
    fi
else
    print_status "Main psyne library already exists"
fi

# Create build directory
print_status "Creating benchmark build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Generate CMakeLists.txt for benchmarks
print_status "Generating CMakeLists.txt..."
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(PsyneBenchmarks CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for benchmarks
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
endif()

# Find required packages
find_package(Threads REQUIRED)

# Find the main psyne library
set(PSYNE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(PSYNE_BUILD_DIR ${PSYNE_ROOT_DIR}/build)

# Include directories
include_directories(${PSYNE_ROOT_DIR}/include)
include_directories(${PSYNE_ROOT_DIR}/src)

# Try to find psyne library - if not found, we'll build header-only
find_library(PSYNE_LIBRARY 
    NAMES psyne
    PATHS ${PSYNE_BUILD_DIR}
    NO_DEFAULT_PATH
)

if(PSYNE_LIBRARY)
    message(STATUS "Found psyne library: ${PSYNE_LIBRARY}")
    set(PSYNE_LINK_LIBRARIES ${PSYNE_LIBRARY})
else()
    message(STATUS "Psyne library not found - using header-only mode")
    # Add core implementation directly to benchmarks
    set(PSYNE_CORE_SOURCES
        ${PSYNE_ROOT_DIR}/src/psyne.cpp
        ${PSYNE_ROOT_DIR}/src/compression/compression.cpp
        ${PSYNE_ROOT_DIR}/src/memory/dynamic_slab_allocator.cpp
        ${PSYNE_ROOT_DIR}/src/memory/custom_allocator.cpp
        ${PSYNE_ROOT_DIR}/src/simd/simd_ops.cpp
    )
    set(PSYNE_LINK_LIBRARIES)
endif()

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-fconcepts-diagnostics-depth=2)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-Wno-unused-parameter)
endif()

# Platform-specific settings
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# Helper function to create benchmark targets
function(add_benchmark_executable name source_file)
    if(PSYNE_LIBRARY)
        add_executable(${name} ${source_file})
        target_link_libraries(${name} 
            ${PSYNE_LINK_LIBRARIES}
            Threads::Threads
            ${CMAKE_DL_LIBS}
        )
    else()
        add_executable(${name} ${source_file} ${PSYNE_CORE_SOURCES})
        target_link_libraries(${name} 
            Threads::Threads
            ${CMAKE_DL_LIBS}
        )
    endif()
    
    # Add atomic library on some platforms
    if(NOT APPLE)
        target_link_libraries(${name} atomic)
    endif()
    
    # Set output directory
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

# Create bin directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add in-process benchmarks
add_subdirectory(../in-process in-process)

# Print summary
message(STATUS "=== Psyne Benchmarks Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
EOF

# Generate CMakeLists.txt for in-process benchmarks
print_status "Generating in-process benchmarks CMakeLists.txt..."
mkdir -p "$SCRIPT_DIR/in-process"
cat > "$SCRIPT_DIR/in-process/CMakeLists.txt" << 'EOF'
# In-process benchmarks

# SPSC (Single Producer Single Consumer) benchmark
add_benchmark_executable(spsc_benchmark spsc_benchmark.cpp)

# MPSC (Multi Producer Single Consumer) benchmark  
add_benchmark_executable(mpsc_benchmark mpsc_benchmark.cpp)

# SPMC (Single Producer Multi Consumer) benchmark
add_benchmark_executable(spmc_benchmark spmc_benchmark.cpp)

# MPMC (Multi Producer Multi Consumer) benchmark
add_benchmark_executable(mpmc_benchmark mpmc_benchmark.cpp)

# Throughput comparison benchmark
add_benchmark_executable(throughput_comparison throughput_comparison.cpp)

# Latency benchmark
add_benchmark_executable(latency_benchmark latency_benchmark.cpp)

# Memory efficiency benchmark
add_benchmark_executable(memory_benchmark memory_benchmark.cpp)

# Scalability benchmark
add_benchmark_executable(scalability_benchmark scalability_benchmark.cpp)
EOF

# Configure with CMake
print_status "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
      -DCMAKE_CXX_COMPILER="$COMPILER" \
      .

# Build benchmarks
print_status "Building benchmarks with $NUM_CORES parallel jobs..."
make -j"$NUM_CORES"

# Check if build was successful
if [ $? -eq 0 ]; then
    print_status "Build completed successfully!"
    echo
    print_status "Benchmark executables are located in: $BUILD_DIR/bin/"
    echo
    echo -e "${BLUE}Available benchmarks:${NC}"
    if [ -d "$BUILD_DIR/bin" ]; then
        ls -la "$BUILD_DIR/bin/" | grep -v "^total" | sed 's/^/  /'
    fi
    echo
    echo -e "${BLUE}To run all benchmarks:${NC}"
    echo "  cd $BUILD_DIR/bin && ./throughput_comparison"
    echo
    echo -e "${BLUE}To run individual benchmarks:${NC}"
    echo "  cd $BUILD_DIR/bin"
    echo "  ./spsc_benchmark"
    echo "  ./mpsc_benchmark" 
    echo "  ./spmc_benchmark"
    echo "  ./mpmc_benchmark"
    echo "  ./latency_benchmark"
    echo "  ./memory_benchmark"
    echo "  ./scalability_benchmark"
else
    print_error "Build failed!"
    exit 1
fi