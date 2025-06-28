#pragma once

#include "../../include/psyne/gpu/gpu_buffer.hpp"
#include <unordered_map>
#include <string>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#else
// Forward declarations for C++ files
typedef struct objc_object* id;
#endif

namespace psyne {
namespace gpu {

// Metal GPU buffer implementation
class MetalBuffer : public GPUBuffer {
public:
    MetalBuffer(id<MTLBuffer> buffer, BufferUsage usage, MemoryAccess access);
    ~MetalBuffer();
    
    // GPUBuffer interface
    size_t size() const override;
    GPUBackend backend() const override { return GPUBackend::Metal; }
    BufferUsage usage() const override { return usage_; }
    MemoryAccess access() const override { return access_; }
    
    void* map() override;
    const void* map() const override;
    void unmap() override;
    bool is_mapped() const override { return mapped_; }
    
    void flush() override;
    void invalidate() override;
    
    void* native_handle() override;
    const void* native_handle() const override;
    
    // Metal-specific
    id<MTLBuffer> metal_buffer() const { return buffer_; }

private:
    id<MTLBuffer> buffer_;
    BufferUsage usage_;
    MemoryAccess access_;
    bool mapped_;
    void* mapped_ptr_;
};

// Metal buffer factory
class MetalBufferFactory : public GPUBufferFactory {
public:
    explicit MetalBufferFactory(id<MTLDevice> device);
    ~MetalBufferFactory();
    
    std::unique_ptr<GPUBuffer> create_buffer(
        size_t size,
        BufferUsage usage = BufferUsage::Dynamic,
        MemoryAccess access = MemoryAccess::Shared
    ) override;
    
    GPUBackend backend() const override { return GPUBackend::Metal; }
    bool is_available() const override;

private:
    id<MTLDevice> device_;
};

// Metal GPU context
class MetalContext : public GPUContext {
public:
    MetalContext();
    ~MetalContext();
    
    GPUBackend backend() const override { return GPUBackend::Metal; }
    std::unique_ptr<GPUBufferFactory> create_buffer_factory() override;
    
    void dispatch_compute(
        const void* shader_data, size_t shader_size,
        GPUBuffer& input, GPUBuffer& output,
        size_t workgroup_x, size_t workgroup_y = 1, size_t workgroup_z = 1
    ) override;
    
    void sync() override;
    
    // Metal-specific
    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> command_queue() const { return command_queue_; }
    
    // Compile and cache compute shaders
    id<MTLComputePipelineState> get_compute_pipeline(const std::string& shader_source);

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLLibrary> library_;
    
    // Shader cache
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache_;
    
    bool initialize();
};

} // namespace gpu
} // namespace psyne