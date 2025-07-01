#pragma once

/**
 * @file gpu_cuda.hpp
 * @brief CUDA GPU substrate for zero-copy host-visible memory
 */

#include <psyne/config_detect.hpp>
#include <psyne/core/behaviors.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef PSYNE_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace psyne::substrate {

/**
 * @brief CUDA GPU substrate implementation with host-visible memory
 * 
 * Provides zero-copy access to GPU memory through CUDA's unified memory
 * or pinned host memory mechanisms.
 */
class GPUCuda : public behaviors::SubstrateBehavior {
public:
    /**
     * @brief Construct CUDA substrate
     * @param device_id CUDA device ID to use
     * @param use_unified_memory Use CUDA unified memory (true) or pinned memory (false)
     */
    explicit GPUCuda(int device_id = 0, bool use_unified_memory = true) 
        : device_id_(device_id), use_unified_memory_(use_unified_memory) {
#ifdef PSYNE_CUDA_ENABLED
        cudaError_t err = cudaSetDevice(device_id_);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // Get device properties
        cudaGetDeviceProperties(&device_props_, device_id_);
#else
        throw std::runtime_error("CUDA support not enabled. Rebuild with PSYNE_CUDA_ENABLED");
#endif
    }
    
    ~GPUCuda() {
        cleanup_allocations();
    }
    
    void* allocate_memory_slab(size_t size_bytes) override {
#ifdef PSYNE_CUDA_ENABLED
        void* ptr = nullptr;
        cudaError_t err;
        
        if (use_unified_memory_) {
            // Use CUDA unified memory for automatic host/device synchronization
            err = cudaMallocManaged(&ptr, size_bytes, cudaMemAttachGlobal);
        } else {
            // Use pinned host memory for zero-copy access
            err = cudaHostAlloc(&ptr, size_bytes, cudaHostAllocMapped);
        }
        
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA allocation failed: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        allocations_.push_back({ptr, size_bytes});
        return ptr;
#else
        throw std::runtime_error("CUDA support not enabled");
#endif
    }
    
    void deallocate_memory_slab(void* memory) override {
#ifdef PSYNE_CUDA_ENABLED
        if (!memory) return;
        
        // Remove from tracking
        auto it = std::find_if(allocations_.begin(), allocations_.end(),
                              [memory](const auto& alloc) { return alloc.ptr == memory; });
        if (it != allocations_.end()) {
            allocations_.erase(it);
        }
        
        cudaError_t err;
        if (use_unified_memory_) {
            err = cudaFree(memory);
        } else {
            err = cudaFreeHost(memory);
        }
        
        if (err != cudaSuccess) {
            // Log error but don't throw in destructor path
        }
#endif
    }
    
    void transport_send(void* data, size_t size) override {
        // GPU memory doesn't need network transport
        // This is for cross-GPU communication in the future
    }
    
    void transport_receive(void* buffer, size_t buffer_size) override {
        // GPU memory doesn't need network transport
        // This is for cross-GPU communication in the future
    }
    
    const char* substrate_name() const override { 
        return use_unified_memory_ ? "GPUCuda-Unified" : "GPUCuda-Pinned"; 
    }
    
    bool is_zero_copy() const override { return true; }
    bool is_cross_process() const override { return false; }
    
    /**
     * @brief Get the CUDA device ID
     */
    int get_device_id() const { return device_id_; }
    
    /**
     * @brief Ensure memory is accessible from device
     * @param ptr Memory pointer
     * @param size Memory size
     */
    void prefetch_to_device(void* ptr, size_t size) {
#ifdef PSYNE_CUDA_ENABLED
        if (use_unified_memory_) {
            cudaMemPrefetchAsync(ptr, size, device_id_, 0);
        }
#endif
    }
    
    /**
     * @brief Ensure memory is accessible from host
     * @param ptr Memory pointer  
     * @param size Memory size
     */
    void prefetch_to_host(void* ptr, size_t size) {
#ifdef PSYNE_CUDA_ENABLED
        if (use_unified_memory_) {
            cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, 0);
        }
#endif
    }

private:
    int device_id_;
    bool use_unified_memory_;
    
    struct Allocation {
        void* ptr;
        size_t size;
    };
    std::vector<Allocation> allocations_;
    
#ifdef PSYNE_CUDA_ENABLED
    cudaDeviceProp device_props_;
#endif
    
    void cleanup_allocations() {
        for (auto& alloc : allocations_) {
            deallocate_memory_slab(alloc.ptr);
        }
        allocations_.clear();
    }
};

} // namespace psyne::substrate