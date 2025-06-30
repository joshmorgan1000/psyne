cmake_minimum_required(VERSION 3.16)
project(psyne VERSION 1.3.0 LANGUAGES C CXX)
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
# Boost - Required for networking (Boost.Asio)
# ----------------------------------------------------------------------------------------
find_package(Boost REQUIRED COMPONENTS system thread)
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost not found. Please install Boost development libraries.")
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
        # Temporarily disable Metal until metal-cpp headers are added
        set(PSYNE_METAL_ENABLED OFF)
        message(STATUS "Metal support temporarily disabled (needs metal-cpp headers)")
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

# ----------------------------------------------------------------------------------------
# Main Psyne library
set(PSYNE_SOURCES
    # Public header
    "${CMAKE_SOURCE_DIR}/include/psyne/psyne.hpp"
    
    # Core implementation - minimal set
    "${CMAKE_SOURCE_DIR}/src/psyne_minimal.cpp"
    "${CMAKE_SOURCE_DIR}/src/channel/webrtc_channel.cpp"
    "${CMAKE_SOURCE_DIR}/src/channel/webrtc/ice_agent.cpp"
    # "${CMAKE_SOURCE_DIR}/src/channel/lockfree_ipc_channel.cpp" # TODO: Fix for new API
    "${CMAKE_SOURCE_DIR}/src/compression/compression.cpp"
    
    # Memory management
    "${CMAKE_SOURCE_DIR}/src/memory/dynamic_slab_allocator.cpp"
    "${CMAKE_SOURCE_DIR}/src/memory/custom_allocator.cpp"
    
    # Header-only files
    "${CMAKE_SOURCE_DIR}/src/channel/channel_impl.hpp"
    "${CMAKE_SOURCE_DIR}/src/debug/metrics.hpp"
    "${CMAKE_SOURCE_DIR}/src/memory/ring_buffer_impl.hpp"
    "${CMAKE_SOURCE_DIR}/src/types/types.hpp"
    "${CMAKE_SOURCE_DIR}/src/types/enhanced_types.hpp"
    "${CMAKE_SOURCE_DIR}/src/types/advanced_types.hpp"
    "${CMAKE_SOURCE_DIR}/src/psyne_impl.hpp"
    
    # SIMD optimizations
    "${CMAKE_SOURCE_DIR}/src/simd/simd_ops.cpp"
    
    # Transport implementations
    "${CMAKE_SOURCE_DIR}/src/transport/quic.cpp"
    
    # Channel implementations
    "${CMAKE_SOURCE_DIR}/src/channel/quic_channel.cpp"
    "${CMAKE_SOURCE_DIR}/src/channel/multicast_factory.cpp"
    "${CMAKE_SOURCE_DIR}/src/channel/udp_multicast_channel.cpp"
    
    # Tensor transport optimizations - temporarily disabled due to constructor issues
    # "${CMAKE_SOURCE_DIR}/src/tensor/tensor_transport.cpp"  TODO: Fix constructor issues
)

# Add GPU sources if enabled
if(PSYNE_ENABLE_GPU)
    list(APPEND PSYNE_SOURCES
        "${CMAKE_SOURCE_DIR}/src/gpu/gpu_context.cpp"
        "${CMAKE_SOURCE_DIR}/src/gpu/gpu_message.cpp"
    )
    
    # Add Metal sources for macOS
    if(PSYNE_METAL_ENABLED)
        list(APPEND PSYNE_SOURCES
            "${CMAKE_SOURCE_DIR}/src/gpu/metal/metal_buffer.cpp"
        )
    endif()
    
    # Add CUDA sources if available
    if(PSYNE_CUDA_ENABLED)
        list(APPEND PSYNE_SOURCES
            "${CMAKE_SOURCE_DIR}/src/gpu/cuda/cuda_buffer.cpp"
            "${CMAKE_SOURCE_DIR}/src/gpu/cuda/cuda_context.cpp"
            "${CMAKE_SOURCE_DIR}/src/gpu/cuda/cuda_kernels.cu"
        )
        
    endif()
    
    # Add Vulkan sources if available
    if(PSYNE_VULKAN_ENABLED)
        list(APPEND PSYNE_SOURCES
            "${CMAKE_SOURCE_DIR}/src/gpu/vulkan/vulkan_buffer.cpp"
        )
    endif()
endif()

add_library(psyne SHARED ${PSYNE_SOURCES})
set(PSYNE_LIBRARIES
    Eigen3::Eigen
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
    Boost::system
    Boost::thread
)

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

target_link_libraries(psyne PRIVATE ${PSYNE_LIBRARIES})
set(PSYNE_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/psyne
    ${CMAKE_SOURCE_DIR}/src/utils
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

# Add GPU include directories if enabled
if(PSYNE_ENABLE_GPU)
    if(PSYNE_CUDA_ENABLED)
        list(APPEND PSYNE_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    
    if(PSYNE_VULKAN_ENABLED)
        list(APPEND PSYNE_INCLUDE_DIRS ${Vulkan_INCLUDE_DIRS})
    endif()
endif()


# Set up public and private include directories
target_include_directories(psyne 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/psyne>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/utils>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/psyne>
        $<INSTALL_INTERFACE:include/psyne/utils>
    PRIVATE
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        $<$<BOOL:${PSYNE_CUDA_ENABLED}>:${CUDAToolkit_INCLUDE_DIRS}>
        $<$<BOOL:${PSYNE_VULKAN_ENABLED}>:${Vulkan_INCLUDE_DIRS}>
)

# Add compile definitions for GPU support
if(PSYNE_ENABLE_GPU)
    target_compile_definitions(psyne PRIVATE PSYNE_GPU_ENABLED)
    
    if(PSYNE_METAL_ENABLED)
        target_compile_definitions(psyne PRIVATE PSYNE_METAL_ENABLED)
    endif()
    
    if(PSYNE_CUDA_ENABLED)
        target_compile_definitions(psyne PRIVATE PSYNE_CUDA_ENABLED)
    endif()
    
    if(PSYNE_VULKAN_ENABLED)
        target_compile_definitions(psyne PRIVATE PSYNE_VULKAN_ENABLED)
        target_include_directories(psyne PRIVATE ${Vulkan_INCLUDE_DIRS})
    endif()
endif()

# ------------------------------------------------------------------------------
# Installation targets
# ------------------------------------------------------------------------------
include(GNUInstallDirs)

# Install the shared library
install(TARGETS psyne
    EXPORT PsyneTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the public header
install(FILES
    "${CMAKE_SOURCE_DIR}/include/psyne/psyne.hpp"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/psyne
)

# Install utility headers that are part of the public API
install(FILES
    "${CMAKE_SOURCE_DIR}/src/utils/logger.hpp"
    "${CMAKE_SOURCE_DIR}/src/utils/time_utils.hpp"
    "${CMAKE_SOURCE_DIR}/src/utils/random_utils.hpp"
    "${CMAKE_SOURCE_DIR}/src/utils/pthread.hpp"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/psyne/utils
)

# Create and install package config files
include(CMakePackageConfigHelpers)

# Generate PsyneConfig.cmake
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PsyneConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/PsyneConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Psyne
)

# Generate PsyneConfigVersion.cmake
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/PsyneConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Install CMake config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/PsyneConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/PsyneConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Psyne
)

# Install the export set
install(EXPORT PsyneTargets
    FILE PsyneTargets.cmake
    NAMESPACE Psyne::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Psyne
)

# Create a pkg-config file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/psyne.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/psyne.pc"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/psyne.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# ------------------------------------------------------------------------------
# Examples (enabled by default)
# ------------------------------------------------------------------------------
option(PSYNE_BUILD_EXAMPLES "Build example programs" ON)
if(PSYNE_BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_SOURCE_DIR}/examples)
endif()

# ------------------------------------------------------------------------------
# CUDA Test (optional) - only if examples are not built
# ------------------------------------------------------------------------------
if(PSYNE_CUDA_ENABLED AND NOT PSYNE_BUILD_EXAMPLES)
    add_executable(test_cuda examples/test_cuda.cpp)
    target_link_libraries(test_cuda PRIVATE psyne)
    target_include_directories(test_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_cuda PRIVATE PSYNE_GPU_ENABLED PSYNE_CUDA_ENABLED)
endif()

# ------------------------------------------------------------------------------
# Tests (enabled by default)
# ------------------------------------------------------------------------------
option(PSYNE_BUILD_TESTS "Build unit tests" ON)
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