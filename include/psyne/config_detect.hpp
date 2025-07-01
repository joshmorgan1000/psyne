#pragma once

/**
 * @file config_detect.hpp
 * @brief Fallback configuration detection for when not using CMake
 * 
 * This file attempts to auto-detect GPU support when including
 * Psyne headers directly without CMake build system.
 */

// First, check if we have a CMake-generated config
#if __has_include(<psyne/config.hpp>)
    #include <psyne/config.hpp>
#else
    // Fallback: Try to auto-detect GPU support

    // Detect CUDA
    #if defined(__CUDACC__) || defined(__NVCC__)
        #define PSYNE_CUDA_ENABLED
    #elif __has_include(<cuda_runtime.h>)
        #define PSYNE_CUDA_ENABLED
    #endif

    // Detect Metal (macOS/iOS only)
    #if defined(__APPLE__)
        #include <TargetConditionals.h>
        #if TARGET_OS_MAC || TARGET_OS_IPHONE
            #if __has_include(<Metal/Metal.h>)
                #define PSYNE_METAL_ENABLED
            #endif
        #endif
    #endif

    // Detect Vulkan
    #if __has_include(<vulkan/vulkan.h>)
        #define PSYNE_VULKAN_ENABLED
    #endif

    // Enable GPU support if any backend is available
    #if defined(PSYNE_CUDA_ENABLED) || defined(PSYNE_METAL_ENABLED) || defined(PSYNE_VULKAN_ENABLED)
        #define PSYNE_ENABLE_GPU
    #endif

    // Version fallback
    #ifndef PSYNE_VERSION_MAJOR
        #define PSYNE_VERSION_MAJOR 2
        #define PSYNE_VERSION_MINOR 0
        #define PSYNE_VERSION_PATCH 1
        #define PSYNE_VERSION_STRING "2.0.1"
    #endif
#endif