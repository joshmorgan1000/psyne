/**
 * @file gpu_context.cpp
 * @brief GPU context factory implementation
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include <psyne/gpu/gpu_buffer.hpp>

#ifdef __APPLE__
#include <psyne/gpu/metal/metal_buffer.hpp>
#endif

#include <iostream>

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
#ifdef __APPLE__
        case GPUBackend::Metal:
            try {
                return std::make_unique<metal::MetalContext>();
            } catch (const std::exception& e) {
                std::cerr << "Failed to create Metal context: " << e.what() << std::endl;
                return nullptr;
            }
#endif
            
        case GPUBackend::CUDA:
            // TODO: Implement CUDA support
            std::cerr << "CUDA support not yet implemented" << std::endl;
            return nullptr;
            
        case GPUBackend::ROCm:
            // TODO: Implement ROCm support
            std::cerr << "ROCm support not yet implemented" << std::endl;
            return nullptr;
            
        case GPUBackend::Vulkan:
            // TODO: Implement Vulkan support
            std::cerr << "Vulkan support not yet implemented" << std::endl;
            return nullptr;
            
        default:
            return nullptr;
    }
}

std::vector<GPUBackend> detect_gpu_backends() {
    std::vector<GPUBackend> backends;
    
#ifdef __APPLE__
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
    
#ifdef __NVCC__
    // TODO: Check for CUDA support
#endif
    
#ifdef __HIP__
    // TODO: Check for ROCm support
#endif
    
    return backends;
}

const char* gpu_backend_name(GPUBackend backend) {
    switch (backend) {
        case GPUBackend::None:   return "None";
        case GPUBackend::Metal:  return "Metal";
        case GPUBackend::CUDA:   return "CUDA";
        case GPUBackend::ROCm:   return "ROCm";
        case GPUBackend::Vulkan: return "Vulkan";
        default:                 return "Unknown";
    }
}

} // namespace gpu
} // namespace psyne