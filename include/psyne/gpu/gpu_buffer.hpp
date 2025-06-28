#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

namespace psyne {
namespace gpu {

// GPU backend types
enum class GPUBackend {
    Metal,
    Vulkan,
    CUDA,
    OpenCL
};

// GPU buffer usage patterns
enum class BufferUsage {
    Static,      // Written once, read many times
    Dynamic,     // Updated frequently
    Streaming,   // Written once per frame
    Staging      // CPU->GPU or GPU->CPU transfer
};

// GPU memory access patterns
enum class MemoryAccess {
    GPUOnly,     // GPU-only memory (fastest)
    Shared,      // CPU/GPU shared memory (unified memory)
    Staging      // CPU-writable, GPU-readable
};

// Abstract GPU buffer interface
class GPUBuffer {
public:
    virtual ~GPUBuffer() = default;
    
    // Buffer properties
    virtual size_t size() const = 0;
    virtual GPUBackend backend() const = 0;
    virtual BufferUsage usage() const = 0;
    virtual MemoryAccess access() const = 0;
    
    // CPU access (only valid for Shared/Staging memory)
    virtual void* map() = 0;
    virtual const void* map() const = 0;
    virtual void unmap() = 0;
    virtual bool is_mapped() const = 0;
    
    // GPU synchronization
    virtual void flush() = 0;      // CPU->GPU sync
    virtual void invalidate() = 0; // GPU->CPU sync
    
    // Backend-specific handles
    virtual void* native_handle() = 0;
    virtual const void* native_handle() const = 0;
};

// GPU buffer factory
class GPUBufferFactory {
public:
    virtual ~GPUBufferFactory() = default;
    
    virtual std::unique_ptr<GPUBuffer> create_buffer(
        size_t size,
        BufferUsage usage = BufferUsage::Dynamic,
        MemoryAccess access = MemoryAccess::Shared
    ) = 0;
    
    virtual GPUBackend backend() const = 0;
    virtual bool is_available() const = 0;
};

// GPU context management
class GPUContext {
public:
    virtual ~GPUContext() = default;
    
    virtual GPUBackend backend() const = 0;
    virtual std::unique_ptr<GPUBufferFactory> create_buffer_factory() = 0;
    
    // Compute operations
    virtual void dispatch_compute(
        const void* shader_data, size_t shader_size,
        GPUBuffer& input, GPUBuffer& output,
        size_t workgroup_x, size_t workgroup_y = 1, size_t workgroup_z = 1
    ) = 0;
    
    virtual void sync() = 0;  // Wait for GPU operations to complete
};

// Factory functions
std::unique_ptr<GPUContext> create_metal_context();
std::unique_ptr<GPUContext> create_vulkan_context();
std::unique_ptr<GPUContext> create_cuda_context();

// Auto-detect best available GPU backend
std::unique_ptr<GPUContext> create_gpu_context(GPUBackend preferred = GPUBackend::Metal);

} // namespace gpu
} // namespace psyne