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

#include <iostream>
#include <vector>

namespace psyne {
namespace gpu {

std::unique_ptr<GPUContext> create_gpu_context(GPUBackend backend) {
    // Auto-detect if not specified
    if (backend == GPUBackend::None) {
        auto backends = detect_gpu_backends();
        if (!backends.empty()) {
            backend = backends[0]; // Use first available
        }
    }

    switch (backend) {
    case GPUBackend::Metal:
#if defined(__APPLE__) && defined(PSYNE_METAL_ENABLED)
        try {
            return std::make_unique<metal::MetalContext>();
        } catch (const std::exception &e) {
            std::cerr << "Failed to create Metal context: " << e.what()
                      << std::endl;
            return nullptr;
        }
#else
        std::cerr << "Metal backend not available" << std::endl;
        return nullptr;
#endif

    case GPUBackend::CUDA:
#ifdef PSYNE_CUDA_ENABLED
        try {
            return std::make_unique<cuda::CudaContext>();
        } catch (const std::exception &e) {
            std::cerr << "Failed to create CUDA context: " << e.what()
                      << std::endl;
            return nullptr;
        }
#else
        std::cerr << "CUDA support not compiled in" << std::endl;
        return nullptr;
#endif

    case GPUBackend::Vulkan:
#ifdef PSYNE_VULKAN_ENABLED
        try {
            return std::make_unique<vulkan::VulkanContext>();
        } catch (const std::exception &e) {
            std::cerr << "Failed to create Vulkan context: " << e.what()
                      << std::endl;
            return nullptr;
        }
#else
        std::cerr << "Vulkan support not compiled in" << std::endl;
        return nullptr;
#endif

    default:
        return nullptr;
    }
}

std::vector<GPUBackend> detect_gpu_backends() {
    std::vector<GPUBackend> backends;

#if defined(__APPLE__) && defined(PSYNE_METAL_ENABLED)
    // Check for Metal support
    try {
        auto context = std::make_unique<metal::MetalContext>();
        if (context) {
            backends.push_back(GPUBackend::Metal);
        }
    } catch (...) {
        // Metal not available
    }
#endif

#ifdef PSYNE_CUDA_ENABLED
    // Check for CUDA support
    try {
        if (cuda::utils::is_cuda_available()) {
            backends.push_back(GPUBackend::CUDA);
        }
    } catch (...) {
        // CUDA not available
    }
#endif

#ifdef PSYNE_VULKAN_ENABLED
    // Check for Vulkan support
    try {
        auto context = std::make_unique<vulkan::VulkanContext>();
        if (context) {
            backends.push_back(GPUBackend::Vulkan);
        }
    } catch (...) {
        // Vulkan not available
    }
#endif

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