cmake_minimum_required(VERSION 3.16)
project(psyne VERSION 0.1.1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
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
    add_compile_options("-fsanitize=address" "-fno-omit-frame-pointer")
    add_link_options("-fsanitize=address" "-fno-omit-frame-pointer")
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
# Vulkan
# ----------------------------------------------------------------------------------------
find_package(Vulkan REQUIRED)
if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan not found. Please install Vulkan development libraries.")
else()
    if(NOT Vulkan_INCLUDE_DIRS)
        message(FATAL_ERROR "Vulkan include directories not found. Please ensure Vulkan development libraries are installed.")
    endif()
    if(NOT Vulkan_LIBRARY)
        message(FATAL_ERROR "Vulkan library not found. Please ensure Vulkan development libraries are installed.")
    endif()
    message(STATUS "Vulkan include directories: ${Vulkan_INCLUDE_DIRS}")
    message(STATUS "Vulkan library: ${Vulkan_LIBRARY}")
endif()

# ----------------------------------------------------------------------------------------
# Threads
# ----------------------------------------------------------------------------------------
find_package(Threads REQUIRED)
if(NOT Threads_FOUND)
    message(FATAL_ERROR "Threads not found. Please ensure threading support is available.")
endif()

# ----------------------------------------------------------------------------------------
# BLAKE3 - a fast cryptographic hash function
# ----------------------------------------------------------------------------------------
add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/BLAKE3/c")

# ----------------------------------------------------------------------------------------
# Zstandard (Zstd) - a fast lossless compression algorithm
# ----------------------------------------------------------------------------------------
find_package(zstd REQUIRED)

# ----------------------------------------------------------------------------------------
# Main Psyne library
add_library(psyne STATIC
    src/arena/vectorslab.hpp
    src/arena/slab.hpp
    src/arena/variant.hpp
    src/arena/variantbuilder.hpp
    src/arena/byteslab.hpp
    src/utils/utils.hpp
    src/utils/vthread.hpp
    src/utils/nlohmann/json.hpp
    src/encryption/encryption.hpp
    src/encryption/obfuscate.hpp
    src/encryption/openssl_raii.hpp
    src/cortex/psyne.cpp
)
target_link_libraries(
    psyne
    PRIVATE
    Eigen3::Eigen
    OpenSSL::SSL
    OpenSSL::Crypto
    Vulkan::Vulkan
    Threads::Threads
    zstd::libzstd_static
)
target_include_directories(psyne
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/psyne
    ${CMAKE_SOURCE_DIR}/src/arena
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/gpu
    ${CMAKE_SOURCE_DIR}/src/cortex
    ${CMAKE_SOURCE_DIR}/src/net
    ${CMAKE_SOURCE_DIR}/src/dispatch
    ${CMAKE_SOURCE_DIR}/src/encryption
    ${Vulkan_INCLUDE_DIRS}
    /Users/joshmorgan/boost
)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
add_subdirectory(${CMAKE_SOURCE_DIR}/tests)

# ------------------------------------------------------------------------------
# Status
# ------------------------------------------------------------------------------
message(STATUS "  _____  ______ __    _ ____   _  ______  ")
message(STATUS " |  .  ||   ___|\\ \\  //|    \\ | ||   ___| ")
message(STATUS " |    _| `-.`-.  \\ \\// |     \\| ||   ___| ")
message(STATUS " |___|  |______| /__/  |__/\\____||______| ")
message(STATUS " Ahigh-performance C++ RPC library for AI")
message(STATUS "===================================================")
message(STATUS "CMake Project Configured!")
message(STATUS "Build Type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "Project Version:     ${PROJECT_VERSION}")
message(STATUS "CMake Version:       ${CMAKE_VERSION}")
message(STATUS "C++ Standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenSSL Version:     ${OPENSSL_VERSION}")
message(STATUS "===================================================")