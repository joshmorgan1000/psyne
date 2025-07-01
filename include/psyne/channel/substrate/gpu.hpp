#pragma once

/**
 * @file gpu.hpp
 * @brief Unified GPU substrate interface that auto-selects the best available backend
 */

#include <psyne/config_detect.hpp>
#include <psyne/core/behaviors.hpp>
#include <memory>
#include <string>

// Include available GPU backends
#ifdef PSYNE_CUDA_ENABLED
#include "gpu_cuda.hpp"
#endif

#ifdef PSYNE_METAL_ENABLED
#include "gpu_metal.hpp"
#endif

#ifdef PSYNE_VULKAN_ENABLED
#include "gpu_vulkan.hpp"
#endif

namespace psyne::substrate {

/**
 * @brief GPU backend type enumeration
 */
enum class GPUBackend {
    Auto,    // Auto-select best available
    CUDA,    // NVIDIA CUDA
    Metal,   // Apple Metal
    Vulkan   // Khronos Vulkan
};

/**
 * @brief Unified GPU substrate that auto-selects the best available backend
 * 
 * This substrate provides a unified interface to GPU memory across different
 * platforms and GPU APIs. It automatically selects the best available backend
 * or allows manual selection.
 */
class GPU : public behaviors::SubstrateBehavior {
public:
    /**
     * @brief Construct GPU substrate
     * @param backend Preferred backend (Auto for automatic selection)
     * @param device_index GPU device index
     */
    explicit GPU(GPUBackend backend = GPUBackend::Auto, int device_index = 0) {
        if (backend == GPUBackend::Auto) {
            backend = detect_best_backend();
        }
        
        switch (backend) {
#ifdef PSYNE_CUDA_ENABLED
            case GPUBackend::CUDA:
                impl_ = std::make_unique<GPUCuda>(device_index);
                backend_name_ = "CUDA";
                substrate_name_ = "GPU-CUDA";
                break;
#endif

#ifdef PSYNE_METAL_ENABLED
            case GPUBackend::Metal:
                impl_ = std::make_unique<GPUMetal>(device_index);
                backend_name_ = "Metal";
                substrate_name_ = "GPU-Metal";
                break;
#endif

#ifdef PSYNE_VULKAN_ENABLED
            case GPUBackend::Vulkan:
                impl_ = std::make_unique<GPUVulkan>(device_index);
                backend_name_ = "Vulkan";
                substrate_name_ = "GPU-Vulkan";
                break;
#endif
            
            default:
                throw std::runtime_error("Requested GPU backend not available. "
                                       "Rebuild with appropriate GPU support enabled.");
        }
    }
    
    // Delegating interface
    void* allocate_memory_slab(size_t size_bytes) override {
        return impl_->allocate_memory_slab(size_bytes);
    }
    
    void deallocate_memory_slab(void* memory) override {
        impl_->deallocate_memory_slab(memory);
    }
    
    void transport_send(void* data, size_t size) override {
        impl_->transport_send(data, size);
    }
    
    void transport_receive(void* buffer, size_t buffer_size) override {
        impl_->transport_receive(buffer, buffer_size);
    }
    
    const char* substrate_name() const override { 
        return substrate_name_.c_str();
    }
    
    bool is_zero_copy() const override { return true; }
    bool is_cross_process() const override { return false; }
    
    /**
     * @brief Get the active GPU backend
     */
    const std::string& get_backend_name() const { return backend_name_; }
    
    /**
     * @brief Check if a specific backend is available
     */
    static bool is_backend_available(GPUBackend backend) {
        switch (backend) {
            case GPUBackend::CUDA:
#ifdef PSYNE_CUDA_ENABLED
                return true;
#else
                return false;
#endif
            
            case GPUBackend::Metal:
#ifdef PSYNE_METAL_ENABLED
                return true;
#else
                return false;
#endif
            
            case GPUBackend::Vulkan:
#ifdef PSYNE_VULKAN_ENABLED
                return true;
#else
                return false;
#endif
            
            default:
                return false;
        }
    }
    
    /**
     * @brief Get list of available GPU backends
     */
    static std::vector<std::string> get_available_backends() {
        std::vector<std::string> backends;
        
#ifdef PSYNE_CUDA_ENABLED
        backends.push_back("CUDA");
#endif
        
#ifdef PSYNE_METAL_ENABLED
        backends.push_back("Metal");
#endif
        
#ifdef PSYNE_VULKAN_ENABLED
        backends.push_back("Vulkan");
#endif
        
        return backends;
    }

private:
    std::unique_ptr<behaviors::SubstrateBehavior> impl_;
    std::string backend_name_;
    std::string substrate_name_;
    
    GPUBackend detect_best_backend() {
        // Priority order: Native platform APIs first, then fallbacks
        
#ifdef __APPLE__
        // Prefer Metal on Apple platforms
        #ifdef PSYNE_METAL_ENABLED
            return GPUBackend::Metal;
        #endif
#endif
        
#ifdef PSYNE_CUDA_ENABLED
        // Check if CUDA is actually available (device present)
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            return GPUBackend::CUDA;
        }
#endif
        
#ifdef PSYNE_VULKAN_ENABLED
        // Vulkan as fallback - works everywhere
        return GPUBackend::Vulkan;
#endif
        
        throw std::runtime_error("No GPU backend available. "
                               "Build with PSYNE_CUDA_ENABLED, PSYNE_METAL_ENABLED, "
                               "or PSYNE_VULKAN_ENABLED");
    }
};

// Convenience aliases for specific backends
#ifdef PSYNE_CUDA_ENABLED
using gpu_cuda = GPUCuda;
#endif

#ifdef PSYNE_METAL_ENABLED  
using gpu_metal = GPUMetal;
#endif

#ifdef PSYNE_VULKAN_ENABLED
using gpu_vulkan = GPUVulkan;
#endif

using gpu = GPU;

} // namespace psyne::substrate