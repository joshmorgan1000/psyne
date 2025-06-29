/**
 * @file gpu_buffer.hpp
 * @brief GPU buffer abstraction for zero-copy operations
 * 
 * Provides a unified interface for GPU buffers across different backends:
 * - Apple Metal (unified memory)
 * - NVIDIA CUDA (GPUDirect)
 * - AMD ROCm (DirectGMA)
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#pragma once

#include <memory>
#include <span>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace psyne {
namespace gpu {

/**
 * @brief Supported GPU backends
 */
enum class GPUBackend {
    None,      ///< No GPU support
    Metal,     ///< Apple Metal
    CUDA,      ///< NVIDIA CUDA
    ROCm,      ///< AMD ROCm
    Vulkan     ///< Vulkan (future)
};

/**
 * @brief GPU buffer usage patterns
 */
enum class BufferUsage {
    Static,    ///< Buffer content rarely changes
    Dynamic,   ///< Buffer content changes frequently
    Stream     ///< Buffer content changes every frame
};

/**
 * @brief Memory access modes
 */
enum class MemoryAccess {
    DeviceOnly,    ///< GPU access only
    HostOnly,      ///< CPU access only
    Shared,        ///< Both CPU and GPU access (unified memory)
    Managed        ///< Automatic migration between CPU/GPU
};

/**
 * @brief Abstract GPU buffer interface
 */
class GPUBuffer {
public:
    virtual ~GPUBuffer() = default;
    
    /**
     * @brief Get the size of the buffer in bytes
     */
    virtual size_t size() const = 0;
    
    /**
     * @brief Get the buffer usage pattern
     */
    virtual BufferUsage usage() const = 0;
    
    /**
     * @brief Get the memory access mode
     */
    virtual MemoryAccess access() const = 0;
    
    /**
     * @brief Map the buffer for CPU access
     * @return Pointer to mapped memory or nullptr if mapping fails
     */
    virtual void* map() = 0;
    
    /**
     * @brief Unmap the buffer after CPU access
     */
    virtual void unmap() = 0;
    
    /**
     * @brief Flush any pending changes to ensure GPU visibility
     */
    virtual void flush() = 0;
    
    /**
     * @brief Get native handle for backend-specific operations
     */
    virtual void* native_handle() = 0;
    
    /**
     * @brief Check if buffer is currently mapped
     */
    virtual bool is_mapped() const = 0;
    
    /**
     * @brief Copy data from host memory to buffer
     */
    virtual void upload(const void* data, size_t size, size_t offset = 0) = 0;
    
    /**
     * @brief Copy data from buffer to host memory
     */
    virtual void download(void* data, size_t size, size_t offset = 0) = 0;
};

/**
 * @brief GPU buffer factory interface
 */
class GPUBufferFactory {
public:
    virtual ~GPUBufferFactory() = default;
    
    /**
     * @brief Create a GPU buffer
     * @param size Buffer size in bytes
     * @param usage Buffer usage pattern
     * @param access Memory access mode
     * @return Created buffer or nullptr on failure
     */
    virtual std::unique_ptr<GPUBuffer> create_buffer(
        size_t size,
        BufferUsage usage = BufferUsage::Dynamic,
        MemoryAccess access = MemoryAccess::Shared
    ) = 0;
    
    /**
     * @brief Get the GPU backend type
     */
    virtual GPUBackend backend() const = 0;
    
    /**
     * @brief Check if unified memory is supported
     */
    virtual bool supports_unified_memory() const = 0;
    
    /**
     * @brief Get maximum buffer size supported
     */
    virtual size_t max_buffer_size() const = 0;
};

/**
 * @brief GPU context for managing GPU resources
 */
class GPUContext {
public:
    virtual ~GPUContext() = default;
    
    /**
     * @brief Get the GPU backend type
     */
    virtual GPUBackend backend() const = 0;
    
    /**
     * @brief Create a buffer factory
     */
    virtual std::unique_ptr<GPUBufferFactory> create_buffer_factory() = 0;
    
    /**
     * @brief Get device name
     */
    virtual std::string device_name() const = 0;
    
    /**
     * @brief Get total GPU memory in bytes
     */
    virtual size_t total_memory() const = 0;
    
    /**
     * @brief Get available GPU memory in bytes
     */
    virtual size_t available_memory() const = 0;
    
    /**
     * @brief Check if this is a unified memory architecture
     */
    virtual bool is_unified_memory() const = 0;
    
    /**
     * @brief Synchronize GPU operations
     */
    virtual void synchronize() = 0;
};

/**
 * @brief Create a GPU context for the specified backend
 * @param backend The GPU backend to use (or None for auto-detection)
 * @return GPU context or nullptr if no suitable GPU found
 */
std::unique_ptr<GPUContext> create_gpu_context(GPUBackend backend = GPUBackend::None);

/**
 * @brief Detect available GPU backends
 * @return Vector of available backends
 */
std::vector<GPUBackend> detect_gpu_backends();

/**
 * @brief Get the name of a GPU backend
 */
const char* gpu_backend_name(GPUBackend backend);

} // namespace gpu
} // namespace psyne