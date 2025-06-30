/**
 * @file gpu_context.cpp
 * @brief GPU context factory implementation
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include "gpu_buffer.hpp"

#if defined(__APPLE__) && defined(PSYNE_METAL_ENABLED)
#include "metal/metal_buffer.hpp"
#endif

#ifdef PSYNE_CUDA_ENABLED
#include "cuda/cuda_buffer.hpp"
#endif

#ifdef PSYNE_VULKAN_ENABLED
#include "vulkan/vulkan_buffer.hpp"
#endif

#include "../utils/logger.hpp"
#include <iostream>
#include <vector>

namespace psyne {
namespace gpu {

std::unique_ptr<GPUContext> create_gpu_context(GPUBackend backend) {
    log_info("Creating GPU context for backend: ", gpu_backend_name(backend));

    // Auto-detect if not specified
    if (backend == GPUBackend::None) {
        log_debug("Auto-detecting available GPU backends");
        auto backends = detect_gpu_backends();
        if (!backends.empty()) {
            backend = backends[0]; // Use first available
            log_info("Auto-selected GPU backend: ", gpu_backend_name(backend));
        } else {
            log_warn("No GPU backends available");
            return nullptr;
        }
    }

    switch (backend) {
    case GPUBackend::Metal:
#if defined(__APPLE__) && defined(PSYNE_METAL_ENABLED)
        try {
            log_debug("Creating Metal GPU context");
            auto context = std::make_unique<metal::MetalContext>();
            log_info("Metal GPU context created successfully");
            return context;
        } catch (const std::exception &e) {
            log_error("Failed to create Metal context: ", e.what());
            return nullptr;
        }
#else
        log_warn("Metal backend not available on this platform");
        return nullptr;
#endif

    case GPUBackend::CUDA:
#ifdef PSYNE_CUDA_ENABLED
        try {
            log_debug("Creating CUDA GPU context");
            auto context = std::make_unique<cuda::CudaContext>();
            log_info("CUDA GPU context created successfully");
            return context;
        } catch (const std::exception &e) {
            log_error("Failed to create CUDA context: ", e.what());
            return nullptr;
        }
#else
        log_warn("CUDA support not compiled in");
        return nullptr;
#endif

    case GPUBackend::Vulkan:
#ifdef PSYNE_VULKAN_ENABLED
        try {
            log_debug("Creating Vulkan GPU context");
            auto context = std::make_unique<vulkan::VulkanContext>();
            log_info("Vulkan GPU context created successfully");
            return context;
        } catch (const std::exception &e) {
            log_error("Failed to create Vulkan context: ", e.what());
            return nullptr;
        }
#else
        log_warn("Vulkan support not compiled in");
        return nullptr;
#endif

    default:
        log_error("Unknown GPU backend requested: ", static_cast<int>(backend));
        return nullptr;
    }
}

std::vector<GPUBackend> detect_gpu_backends() {
    log_debug("Detecting available GPU backends");
    std::vector<GPUBackend> backends;

#if defined(__APPLE__) && defined(PSYNE_METAL_ENABLED)
    // Check for Metal support
    try {
        log_trace("Checking Metal GPU backend availability");
        auto context = std::make_unique<metal::MetalContext>();
        if (context) {
            backends.push_back(GPUBackend::Metal);
            log_debug("Metal GPU backend available");
        }
    } catch (...) {
        log_trace("Metal GPU backend not available");
    }
#endif

#ifdef PSYNE_CUDA_ENABLED
    // Check for CUDA support
    try {
        log_trace("Checking CUDA GPU backend availability");
        if (cuda::utils::is_cuda_available()) {
            backends.push_back(GPUBackend::CUDA);
            log_debug("CUDA GPU backend available");
        } else {
            log_trace("CUDA runtime not available");
        }
    } catch (...) {
        log_trace("CUDA GPU backend not available (exception)");
    }
#endif

#ifdef PSYNE_VULKAN_ENABLED
    // Check for Vulkan support
    try {
        log_trace("Checking Vulkan GPU backend availability");
        auto context = std::make_unique<vulkan::VulkanContext>();
        if (context) {
            backends.push_back(GPUBackend::Vulkan);
            log_debug("Vulkan GPU backend available");
        }
    } catch (...) {
        log_trace("Vulkan GPU backend not available");
    }
#endif

    log_info("GPU backend detection completed, found ", backends.size(),
             " available backends");
    return backends;
}

const char *gpu_backend_name(GPUBackend backend) {
    switch (backend) {
    case GPUBackend::None:
        return "None";
    case GPUBackend::Metal:
        return "Metal";
    case GPUBackend::CUDA:
        return "CUDA";
    case GPUBackend::Vulkan:
        return "Vulkan";
    default:
        return "Unknown";
    }
}

} // namespace gpu
} // namespace psyne