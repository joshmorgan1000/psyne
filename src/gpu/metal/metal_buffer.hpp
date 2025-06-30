/**
 * @file metal_buffer.hpp
 * @brief Apple Metal buffer implementation for unified memory
 *
 * Leverages Apple Silicon's unified memory architecture for true zero-copy
 * GPU operations. CPU and GPU share the same physical memory.
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include "../gpu_buffer.hpp"

#ifdef __APPLE__

#include <Metal/Metal.h>
#include <memory>
#include <string>

namespace psyne {
namespace gpu {
namespace metal {

/**
 * @brief Metal-specific GPU buffer implementation
 */
class MetalBuffer : public GPUBuffer {
public:
    MetalBuffer(MTL::Buffer *buffer, BufferUsage usage, MemoryAccess access);
    ~MetalBuffer() override;

    // GPUBuffer interface
    size_t size() const override;
    BufferUsage usage() const override;
    MemoryAccess access() const override;
    void *map() override;
    void unmap() override;
    void flush() override;
    void *native_handle() override;
    bool is_mapped() const override;
    void upload(const void *data, size_t size, size_t offset = 0) override;
    void download(void *data, size_t size, size_t offset = 0) override;

    /**
     * @brief Get the Metal buffer object
     */
    MTL::Buffer *metal_buffer() const {
        return buffer_;
    }

    /**
     * @brief Get direct pointer for zero-copy access on unified memory
     * Returns the buffer's contents pointer for direct CPU/GPU access
     */
    void *zero_copy_ptr() const {
        return buffer_->contents();
    }

private:
    MTL::Buffer *buffer_;
    BufferUsage usage_;
    MemoryAccess access_;
    void *mapped_ptr_;
    bool is_mapped_;
};

/**
 * @brief Metal GPU buffer factory
 */
class MetalBufferFactory : public GPUBufferFactory {
public:
    explicit MetalBufferFactory(MTL::Device *device);
    ~MetalBufferFactory() override;

    std::unique_ptr<GPUBuffer>
    create_buffer(size_t size, BufferUsage usage = BufferUsage::Dynamic,
                  MemoryAccess access = MemoryAccess::Shared) override;

    GPUBackend backend() const override {
        return GPUBackend::Metal;
    }
    bool supports_unified_memory() const override {
        return true;
    }
    size_t max_buffer_size() const override;

private:
    MTL::Device *device_;

    MTL::ResourceOptions get_resource_options(BufferUsage usage,
                                              MemoryAccess access) const;
};

/**
 * @brief Metal GPU context
 */
class MetalContext : public GPUContext {
public:
    MetalContext();
    ~MetalContext() override;

    GPUBackend backend() const override {
        return GPUBackend::Metal;
    }
    std::unique_ptr<GPUBufferFactory> create_buffer_factory() override;
    std::string device_name() const override;
    size_t total_memory() const override;
    size_t available_memory() const override;
    bool is_unified_memory() const override;
    void synchronize() override;

    /**
     * @brief Get the Metal device
     */
    MTL::Device *device() const {
        return device_;
    }

    /**
     * @brief Get the default command queue
     */
    MTL::CommandQueue *command_queue() const {
        return command_queue_;
    }

    /**
     * @brief Create a compute pipeline state
     */
    MTL::ComputePipelineState *
    create_compute_pipeline(const std::string &function_name);

    /**
     * @brief Load default Metal library
     */
    MTL::Library *default_library() const {
        return default_library_;
    }

private:
    MTL::Device *device_;
    MTL::CommandQueue *command_queue_;
    MTL::Library *default_library_;

    bool initialize();
};

/**
 * @brief RAII wrapper for Metal command buffer
 */
class MetalCommandBuffer {
public:
    explicit MetalCommandBuffer(MTL::CommandQueue *queue);
    ~MetalCommandBuffer();

    MTL::CommandBuffer *get() const {
        return buffer_;
    }
    MTL::ComputeCommandEncoder *compute_encoder();

    void commit();
    void wait_until_completed();

private:
    MTL::CommandBuffer *buffer_;
    MTL::ComputeCommandEncoder *encoder_;
};

} // namespace metal
} // namespace gpu
} // namespace psyne

#endif // __APPLE__