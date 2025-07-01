#pragma once

/**
 * @file gpu_metal.hpp
 * @brief Metal GPU substrate for zero-copy host-visible memory on macOS/iOS
 */

#include <psyne/config_detect.hpp>
#include <psyne/core/behaviors.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef PSYNE_METAL_ENABLED
#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#else
// Forward declarations for C++ only compilation
typedef void* id;
#endif
#endif

namespace psyne::substrate {

/**
 * @brief Metal GPU substrate implementation with shared memory
 * 
 * Provides zero-copy access to GPU memory through Metal's shared storage mode,
 * allowing both CPU and GPU to access the same memory without copies.
 */
class GPUMetal : public behaviors::SubstrateBehavior {
public:
    /**
     * @brief Construct Metal substrate
     * @param device_index Metal device index to use
     */
    explicit GPUMetal(int device_index = 0) : device_index_(device_index) {
#ifdef PSYNE_METAL_ENABLED
        init_metal_device();
#else
        throw std::runtime_error("Metal support not enabled. Rebuild with PSYNE_METAL_ENABLED");
#endif
    }
    
    ~GPUMetal() {
        cleanup_allocations();
    }
    
    void* allocate_memory_slab(size_t size_bytes) override {
#ifdef PSYNE_METAL_ENABLED
        // Create a Metal buffer with shared storage mode
        // This allows both CPU and GPU access without copies
        id buffer = create_shared_buffer(size_bytes);
        
        if (!buffer) {
            throw std::runtime_error("Failed to allocate Metal buffer");
        }
        
        // Get the CPU-accessible pointer
        void* ptr = get_buffer_contents(buffer);
        
        allocations_.push_back({buffer, ptr, size_bytes});
        return ptr;
#else
        throw std::runtime_error("Metal support not enabled");
#endif
    }
    
    void deallocate_memory_slab(void* memory) override {
#ifdef PSYNE_METAL_ENABLED
        if (!memory) return;
        
        // Find and release the Metal buffer
        auto it = std::find_if(allocations_.begin(), allocations_.end(),
                              [memory](const auto& alloc) { return alloc.cpu_ptr == memory; });
        
        if (it != allocations_.end()) {
            release_buffer(it->metal_buffer);
            allocations_.erase(it);
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
    
    const char* substrate_name() const override { return "GPUMetal"; }
    bool is_zero_copy() const override { return true; }
    bool is_cross_process() const override { return false; }
    
    /**
     * @brief Get the Metal device index
     */
    int get_device_index() const { return device_index_; }
    
    /**
     * @brief Ensure CPU/GPU coherency for a memory region
     * @param ptr Memory pointer
     * @param size Memory size
     */
    void synchronize_memory(void* ptr, size_t size) {
#ifdef PSYNE_METAL_ENABLED
        // Metal's shared buffers are automatically coherent
        // This is a no-op but provided for API consistency
#endif
    }

private:
    int device_index_;
    
    struct Allocation {
        id metal_buffer;     // MTLBuffer*
        void* cpu_ptr;       // CPU-accessible pointer
        size_t size;
    };
    std::vector<Allocation> allocations_;
    
#ifdef PSYNE_METAL_ENABLED
    id metal_device_;        // id<MTLDevice>
    
    void init_metal_device();
    id create_shared_buffer(size_t size);
    void* get_buffer_contents(id buffer);
    void release_buffer(id buffer);
#endif
    
    void cleanup_allocations() {
        for (auto& alloc : allocations_) {
            deallocate_memory_slab(alloc.cpu_ptr);
        }
        allocations_.clear();
    }
};

} // namespace psyne::substrate

// Implementation details
#ifdef PSYNE_METAL_ENABLED

namespace psyne::substrate {

#ifdef __OBJC__
// Full implementation for Objective-C++ compilation units
inline void GPUMetal::init_metal_device() {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    
    if (device_index_ >= [devices count]) {
        throw std::runtime_error("Metal device index out of range");
    }
    
    metal_device_ = devices[device_index_];
    [metal_device_ retain];
    [devices release];
}

inline id GPUMetal::create_shared_buffer(size_t size) {
    id<MTLDevice> device = (id<MTLDevice>)metal_device_;
    id<MTLBuffer> buffer = [device newBufferWithLength:size
                                               options:MTLResourceStorageModeShared];
    return buffer;
}

inline void* GPUMetal::get_buffer_contents(id buffer) {
    return [(id<MTLBuffer>)buffer contents];
}

inline void GPUMetal::release_buffer(id buffer) {
    [(id<MTLBuffer>)buffer release];
}

#else
// Stub implementations for C++ only compilation
inline void GPUMetal::init_metal_device() {
    throw std::runtime_error("Metal substrate requires Objective-C++ compilation. "
                           "Compile with .mm extension or use -x objective-c++");
}

inline id GPUMetal::create_shared_buffer(size_t size) {
    (void)size;
    throw std::runtime_error("Metal substrate requires Objective-C++ compilation");
}

inline void* GPUMetal::get_buffer_contents(id buffer) {
    (void)buffer;
    throw std::runtime_error("Metal substrate requires Objective-C++ compilation");
}

inline void GPUMetal::release_buffer(id buffer) {
    (void)buffer;
    throw std::runtime_error("Metal substrate requires Objective-C++ compilation");
}
#endif // __OBJC__

} // namespace psyne::substrate

#endif // PSYNE_METAL_ENABLED