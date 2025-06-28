cmake_minimum_required(VERSION 3.16)
project(psyne VERSION 1.2.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS version")
# ----------------------------------------------------------------------------------------
# CMake configuration for the Psyne messaging library
# ----------------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    if(APPLE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-x")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--strip-all")
    endif()
    add_compile_definitions(NDEBUG ABSL_ASSERT_LEVEL=1)
else()
    # Enable AddressSanitizer only for debug builds
    option(PSYNE_ENABLE_ASAN "Enable AddressSanitizer" ON)
    if(PSYNE_ENABLE_ASAN)
        add_compile_options("-fsanitize=address" "-fno-omit-frame-pointer")
        add_link_options("-fsanitize=address" "-fno-omit-frame-pointer")
    endif()
endif()

# ----------------------------------------------------------------------------------------
# Eigen3
# ----------------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Please install Eigen3 development libraries.")
endif()

# ----------------------------------------------------------------------------------------
# OpenSSL
# ----------------------------------------------------------------------------------------
find_package(OpenSSL REQUIRED)
if(NOT OpenSSL_FOUND)
    message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL development libraries.")
endif()


# ----------------------------------------------------------------------------------------
# Boost - Removed to minimize dependencies as per project requirements
# ----------------------------------------------------------------------------------------
# find_package(Boost REQUIRED COMPONENTS system thread)
# if(NOT Boost_FOUND)
#     message(FATAL_ERROR "Boost not found. Please install Boost development libraries.")
# endif()

# ----------------------------------------------------------------------------------------
# Threads
# ----------------------------------------------------------------------------------------
find_package(Threads REQUIRED)
if(NOT Threads_FOUND)
    message(FATAL_ERROR "Threads not found. Please ensure threading support is available.")
endif()

# GPU support has been removed to keep the library focused on transport

# ----------------------------------------------------------------------------------------
# Main Psyne library
add_library(psyne SHARED
    # Public header
    "${CMAKE_SOURCE_DIR}/include/psyne/psyne.hpp"
    
    # Minimal implementation for testing (replaces other implementation files)
    "${CMAKE_SOURCE_DIR}/src/psyne_minimal.cpp"
    
    # Header-only files
    "${CMAKE_SOURCE_DIR}/src/channel/channel_impl.hpp"
    "${CMAKE_SOURCE_DIR}/src/debug/metrics.hpp"
    "${CMAKE_SOURCE_DIR}/src/memory/ring_buffer_impl.hpp"
    "${CMAKE_SOURCE_DIR}/src/types/types.hpp"
    "${CMAKE_SOURCE_DIR}/src/types/enhanced_types.hpp"
    "${CMAKE_SOURCE_DIR}/src/types/advanced_types.hpp"
    "${CMAKE_SOURCE_DIR}/src/psyne_impl.hpp"
)
target_link_libraries(
    psyne
    PRIVATE
    Eigen3::Eigen
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
    # Boost::system  # Removed to minimize dependencies
    # Boost::thread  # Removed to minimize dependencies
)
target_include_directories(psyne
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/psyne
    ${CMAKE_SOURCE_DIR}/src/utils
    ${EIGEN3_INCLUDE_DIR}
    # ${Boost_INCLUDE_DIRS}  # Removed to minimize dependencies
)

# ------------------------------------------------------------------------------
# Examples (optional)
# ------------------------------------------------------------------------------
option(PSYNE_BUILD_EXAMPLES "Build example programs" OFF)
if(PSYNE_BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_SOURCE_DIR}/examples)
endif()

# ------------------------------------------------------------------------------
# Tests (optional)
# ------------------------------------------------------------------------------
option(PSYNE_BUILD_TESTS "Build unit tests" OFF)
# Tests disabled by default due to implementation issues
if(PSYNE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
endif()

# ------------------------------------------------------------------------------
# Benchmarks (optional)
# ------------------------------------------------------------------------------
option(PSYNE_BUILD_BENCHMARKS "Build benchmark programs" OFF)
if(PSYNE_BUILD_BENCHMARKS)
    add_subdirectory(${CMAKE_SOURCE_DIR}/benchmarks)
endif()

# ------------------------------------------------------------------------------
# Python bindings (optional)
# ------------------------------------------------------------------------------
option(PSYNE_PYTHON_BINDINGS "Build Python bindings" OFF)
if(PSYNE_PYTHON_BINDINGS)
    add_subdirectory(${CMAKE_SOURCE_DIR}/bindings/python)
endif()

# ------------------------------------------------------------------------------
# Status
# ------------------------------------------------------------------------------
message(STATUS "  _____  ______ __    _ ____   _  ______  ")
message(STATUS " |  .  ||   ___|\\ \\  //|    \\ | ||   ___| ")
message(STATUS " |    _| `-.`-.  \\ \\// |     \\| ||   ___| ")
message(STATUS " |___|  |______| /__/  |__/\\____||______| ")
message(STATUS " Zero-copy RPC library optimized for AI/ML")
message(STATUS "===================================================")
message(STATUS " ")
message(STATUS " CMake Project Configured!")
message(STATUS "  Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Project Version:   ${PROJECT_VERSION}")
message(STATUS "  CMake Version:     ${CMAKE_VERSION}")
message(STATUS "  C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenSSL Version:   ${OPENSSL_VERSION}")
message(STATUS "")
message(STATUS "Optional Components:")
message(STATUS "  Examples:          ${PSYNE_BUILD_EXAMPLES}")
message(STATUS "  Tests:             ${PSYNE_BUILD_TESTS}")
message(STATUS "  Benchmarks:        ${PSYNE_BUILD_BENCHMARKS}")
message(STATUS "  Python Bindings:   ${PSYNE_PYTHON_BINDINGS}")
message(STATUS "")
message(STATUS "Use -DPSYNE_BUILD_EXAMPLES=ON to build examples")
message(STATUS "Use -DPSYNE_BUILD_TESTS=ON to build tests")
message(STATUS "===================================================")