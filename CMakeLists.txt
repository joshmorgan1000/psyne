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
            # Enable AVX-512 if available
            include(CheckCXXCompilerFlag)
            CHECK_CXX_COMPILER_FLAG("-mavx512f" COMPILER_SUPPORTS_AVX512)
            if(COMPILER_SUPPORTS_AVX512)
                add_compile_options("-mavx512f" "-mavx512dq" "-mavx512bw" "-mavx512vl")
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
    
    # RDMA/UCX/Fabric support removed - no hardware available for testing
    
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
    
    # Tensor transport optimizations - temporarily disabled due to constructor issues
    # "${CMAKE_SOURCE_DIR}/src/tensor/tensor_transport.cpp"
)

# Add GPU sources if enabled
if(PSYNE_ENABLE_GPU)
    list(APPEND PSYNE_SOURCES
        "${CMAKE_SOURCE_DIR}/src/gpu/gpu_context.cpp"
        "${CMAKE_SOURCE_DIR}/src/gpu/gpu_message.cpp"
    )
    
    # RDMA sources removed - no hardware for testing
    
    
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
    
    # RDMA libraries removed - no hardware for testing
    
    
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


target_include_directories(psyne PRIVATE ${PSYNE_INCLUDE_DIRS})

# Add compile definitions for GPU support
if(PSYNE_ENABLE_GPU)
    target_compile_definitions(psyne PRIVATE PSYNE_GPU_ENABLED)
    
    if(PSYNE_METAL_ENABLED)
        target_compile_definitions(psyne PRIVATE PSYNE_METAL_ENABLED)
    endif()
    
    if(PSYNE_CUDA_ENABLED)
        target_compile_definitions(psyne PRIVATE PSYNE_CUDA_ENABLED)
    endif()
    
    # RDMA compile definitions removed - no hardware for testing
    
    
    if(PSYNE_VULKAN_ENABLED)
        target_compile_definitions(psyne PRIVATE PSYNE_VULKAN_ENABLED)
        target_include_directories(psyne PRIVATE ${Vulkan_INCLUDE_DIRS})
    endif()
endif()

# ------------------------------------------------------------------------------
# Examples (optional)
# ------------------------------------------------------------------------------
option(PSYNE_BUILD_EXAMPLES "Build example programs" OFF)
if(PSYNE_BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_SOURCE_DIR}/examples)
endif()

# ------------------------------------------------------------------------------
# CUDA Test (optional)
# ------------------------------------------------------------------------------
if(PSYNE_CUDA_ENABLED)
    add_executable(test_cuda examples/test_cuda.cpp)
    target_link_libraries(test_cuda PRIVATE psyne)
    target_include_directories(test_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_cuda PRIVATE PSYNE_GPU_ENABLED PSYNE_CUDA_ENABLED)
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
message(STATUS "  GPU Support:       ${PSYNE_ENABLE_GPU}")
if(PSYNE_ENABLE_GPU)
    message(STATUS "    CUDA:            ${PSYNE_CUDA_ENABLED}")
    message(STATUS "    Metal:           ${PSYNE_METAL_ENABLED}")
    # RDMA/UCX/GPUDirect support removed - no hardware for testing
endif()
message(STATUS "")
message(STATUS "Use -DPSYNE_BUILD_EXAMPLES=ON to build examples")
message(STATUS "Use -DPSYNE_BUILD_TESTS=ON to build tests")
message(STATUS "===================================================")