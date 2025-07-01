cmake_minimum_required(VERSION 3.20)
project(psyne VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for performance
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address -fsanitize=undefined")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
endif()

# ----------------------------------------------------------------------------------------
# CMake configuration for the Psyne messaging library
# ----------------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif()
    
    if(APPLE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-x")
    elseif(NOT WIN32)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--strip-all")
    endif()
    add_compile_definitions(NDEBUG ABSL_ASSERT_LEVEL=1)
    
    # Enable SIMD optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        if(NOT MSVC)
            add_compile_options("-march=native" "-mavx2" "-mfma")
            # Check if CPU actually supports AVX-512 at build time
            # Only enable AVX-512 if both compiler and CPU support it
            include(CheckCXXCompilerFlag)
            CHECK_CXX_COMPILER_FLAG("-mavx512f" COMPILER_SUPPORTS_AVX512)
            
            if(COMPILER_SUPPORTS_AVX512)
                # Check CPU features at build time (Linux/macOS)
                if(EXISTS "/proc/cpuinfo")
                    execute_process(COMMAND grep -q "avx512f" /proc/cpuinfo
                                  RESULT_VARIABLE CPU_HAS_AVX512
                                  OUTPUT_QUIET ERROR_QUIET)
                elseif(APPLE)
                    execute_process(COMMAND sysctl -n machdep.cpu.features
                                  COMMAND grep -q "AVX512F"
                                  RESULT_VARIABLE CPU_HAS_AVX512
                                  OUTPUT_QUIET ERROR_QUIET)
                else()
                    set(CPU_HAS_AVX512 1) # Unknown platform, skip AVX-512
                endif()
                
                if(CPU_HAS_AVX512 EQUAL 0)
                    add_compile_options("-mavx512f" "-mavx512dq" "-mavx512bw" "-mavx512vl")
                    message(STATUS "AVX-512 enabled (compiler and CPU support detected)")
                else()
                    message(STATUS "AVX-512 disabled (CPU does not support it)")
                endif()
            else()
                message(STATUS "AVX-512 disabled (compiler does not support it)")
            endif()
        else()
            add_compile_options("/arch:AVX2")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        # ARM NEON is enabled by default on AArch64
        add_compile_options("-march=native")
    endif()
else()
    # Enable AddressSanitizer only for debug builds on non-Windows platforms
    if(NOT WIN32)
        option(PSYNE_ENABLE_ASAN "Enable AddressSanitizer" ON)
        if(PSYNE_ENABLE_ASAN)
            add_compile_options("-fsanitize=address" "-fno-omit-frame-pointer")
            add_link_options("-fsanitize=address" "-fno-omit-frame-pointer")
        endif()
    endif()
endif()

# ----------------------------------------------------------------------------------------
# Eigen3
# ----------------------------------------------------------------------------------------
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found - matrix operations enabled")
    add_compile_definitions(PSYNE_HAS_EIGEN3)
else()
    message(STATUS "Eigen3 not found - matrix operations disabled")
endif()

# ----------------------------------------------------------------------------------------
# OpenSSL
# ----------------------------------------------------------------------------------------
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    message(STATUS "OpenSSL ${OPENSSL_VERSION} found - encryption enabled")
    add_compile_definitions(PSYNE_HAS_OPENSSL)
else()
    message(STATUS "OpenSSL not found - encryption disabled")
endif()


# ----------------------------------------------------------------------------------------
# Boost - Required for networking (Boost.Asio)
# ----------------------------------------------------------------------------------------
find_package(Boost QUIET COMPONENTS system thread)
if(Boost_FOUND)
    message(STATUS "Boost found - network substrates enabled")
    add_compile_definitions(PSYNE_HAS_BOOST)
else()
    message(STATUS "Boost not found - network substrates disabled")
endif()

# ----------------------------------------------------------------------------------------
# Threads
# ----------------------------------------------------------------------------------------
find_package(Threads REQUIRED)
if(NOT Threads_FOUND)
    message(FATAL_ERROR "Threads not found. Please ensure threading support is available.")
endif()

# ----------------------------------------------------------------------------------------
# GPU Support (CUDA and Metal)
# ----------------------------------------------------------------------------------------
option(PSYNE_ENABLE_GPU "Enable GPU support" ON)

if(PSYNE_ENABLE_GPU)
    # Check for CUDA
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(PSYNE_CUDA_ENABLED ON)
        message(STATUS "CUDA found: Version ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
        message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARY_DIR}")
    else()
        message(STATUS "CUDA toolkit not found. Install nvidia-cuda-toolkit for CUDA support.")
        set(PSYNE_CUDA_ENABLED OFF)
    endif()
    
    # Check for Metal (macOS only)
    if(APPLE)
        find_library(METAL_FRAMEWORK Metal)
        find_library(METALKIT_FRAMEWORK MetalKit)
        if(METAL_FRAMEWORK AND METALKIT_FRAMEWORK)
            set(PSYNE_METAL_ENABLED ON)
            message(STATUS "Metal support enabled")
        else()
            set(PSYNE_METAL_ENABLED OFF)
            message(STATUS "Metal support disabled (Metal framework not found)")
        endif()
    else()
        set(PSYNE_METAL_ENABLED OFF)
    endif()
    
    # Check for Vulkan
    find_package(Vulkan)
    if(Vulkan_FOUND)
        set(PSYNE_VULKAN_ENABLED ON)
        message(STATUS "Vulkan support enabled")
        message(STATUS "Vulkan include dir: ${Vulkan_INCLUDE_DIRS}")
        message(STATUS "Vulkan library: ${Vulkan_LIBRARIES}")
    else()
        set(PSYNE_VULKAN_ENABLED OFF)
        message(STATUS "Vulkan support disabled (Vulkan SDK not found)")
    endif()
    
else()
    set(PSYNE_CUDA_ENABLED OFF)
    set(PSYNE_METAL_ENABLED OFF)
endif()

# Header-only library - no source files needed for GPU support

# Add GPU libraries if enabled
if(PSYNE_ENABLE_GPU)
    if(PSYNE_METAL_ENABLED)
        find_library(METAL_LIBRARY Metal REQUIRED)
        find_library(METALKIT_LIBRARY MetalKit REQUIRED)
        find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
        list(APPEND PSYNE_LIBRARIES
            ${METAL_LIBRARY}
            ${METALKIT_LIBRARY}
            ${FOUNDATION_LIBRARY}
        )
    endif()
    
    if(PSYNE_CUDA_ENABLED)
        list(APPEND PSYNE_LIBRARIES
            CUDA::cudart
            CUDA::cuda_driver
        )
    endif()
    
    if(PSYNE_VULKAN_ENABLED)
        list(APPEND PSYNE_LIBRARIES
            ${Vulkan_LIBRARIES}
        )
    endif()
endif()

# Add Windows-specific libraries
if(WIN32)
    list(APPEND PSYNE_LIBRARIES ws2_32 iphlpapi)
endif()

# Add Linux-specific libraries
if(UNIX AND NOT APPLE)
    list(APPEND PSYNE_LIBRARIES numa)
endif()

# Add GPU include directories if enabled
if(PSYNE_ENABLE_GPU)
    if(PSYNE_CUDA_ENABLED)
        list(APPEND PSYNE_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    
    if(PSYNE_VULKAN_ENABLED)
        list(APPEND PSYNE_INCLUDE_DIRS ${Vulkan_INCLUDE_DIRS})
    endif()
endif()

# ----------------------------------------------------------------------------------------
# Generate Configuration Header
# ----------------------------------------------------------------------------------------
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/psyne/config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/psyne/config.hpp
    @ONLY
)

# ----------------------------------------------------------------------------------------
# Source Files
# ----------------------------------------------------------------------------------------
set(PSYNE_CORE_SOURCES
    # Header-only library for now - add source files as they are created
)

set(PSYNE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/psyne/global
)

# ----------------------------------------------------------------------------------------
# Create Psyne Library
# ----------------------------------------------------------------------------------------
option(PSYNE_BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(PSYNE_BUILD_TESTS "Build test suite" OFF)
option(PSYNE_BUILD_EXAMPLES "Build examples" OFF)
option(PSYNE_BUILD_BENCHMARKS "Build benchmarks" ON)

# Create header-only interface library
add_library(psyne INTERFACE)

# Interface library doesn't need most properties

# Include directories
target_include_directories(psyne INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/psyne/global>
    $<INSTALL_INTERFACE:include>
)

# Link libraries
target_link_libraries(psyne INTERFACE
    Threads::Threads
    ${PSYNE_LIBRARIES}
)

# Optional dependencies
if(Eigen3_FOUND)
    target_link_libraries(psyne INTERFACE Eigen3::Eigen)
endif()

if(OpenSSL_FOUND)
    target_link_libraries(psyne INTERFACE OpenSSL::SSL OpenSSL::Crypto)
endif()

if(Boost_FOUND)
    target_link_libraries(psyne INTERFACE Boost::system Boost::thread)
endif()

# Compile definitions
target_compile_definitions(psyne INTERFACE
    PSYNE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    PSYNE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    PSYNE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

if(PSYNE_ENABLE_GPU)
    target_compile_definitions(psyne INTERFACE PSYNE_ENABLE_GPU)
    if(PSYNE_CUDA_ENABLED)
        target_compile_definitions(psyne INTERFACE PSYNE_CUDA_ENABLED)
    endif()
    if(PSYNE_METAL_ENABLED)
        target_compile_definitions(psyne INTERFACE PSYNE_METAL_ENABLED)
    endif()
    if(PSYNE_VULKAN_ENABLED)
        target_compile_definitions(psyne INTERFACE PSYNE_VULKAN_ENABLED)
    endif()
endif()

# Alias for easier usage
add_library(Psyne::psyne ALIAS psyne)

# ----------------------------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------------------------
if(PSYNE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ----------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------
if(PSYNE_BUILD_TESTS)
    enable_testing()
    find_package(GTest)
    if(NOT GTest_FOUND)
        message(WARNING "Google Test not found. Tests will be disabled.")
    else()
        add_subdirectory(tests)
    endif()
endif()

# ----------------------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------------------
if(PSYNE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ----------------------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------------------
include(GNUInstallDirs)

# Install library
install(TARGETS psyne
    EXPORT PsyneTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Global headers are already part of the include directory
# No separate installation needed as they're in include/psyne/global/

# Create and install CMake config files
install(EXPORT PsyneTargets
    FILE PsyneTargets.cmake
    NAMESPACE Psyne::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Psyne
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    PsyneConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PsyneConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PsyneConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Psyne
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/PsyneConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/PsyneConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Psyne
)

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
message(STATUS "  GPU Support:       ${PSYNE_ENABLE_GPU}")
if(PSYNE_ENABLE_GPU)
    message(STATUS "    CUDA:            ${PSYNE_CUDA_ENABLED}")
    message(STATUS "    Metal:           ${PSYNE_METAL_ENABLED}")
    message(STATUS "    Vulkan:          ${PSYNE_VULKAN_ENABLED}")
endif()
message(STATUS "")
message(STATUS "Use -DPSYNE_BUILD_EXAMPLES=OFF to disable examples")
message(STATUS "Use -DPSYNE_BUILD_TESTS=OFF to disable tests")
message(STATUS "Use 'make install' to install library and headers")
message(STATUS "===================================================")