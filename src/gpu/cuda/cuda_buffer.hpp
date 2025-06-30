/**
 * @file cuda_buffer.hpp
 * @brief CUDA buffer implementation for GPU-accelerated operations
 *
 * Provides CUDA-specific implementations of the GPU buffer interface with
 * support for:
 * - Unified memory
 * - Zero-copy operations where supported
 * - Multi-GPU support
 * - CUDA streams for asynchronous operations
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#ifdef PSYNE_CUDA_ENABLED

#include "../gpu_buffer.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

namespace psyne {
namespace gpu {
namespace cuda {

/**
 * @brief CUDA buffer implementation
 */
class CudaBuffer : public GPUBuffer {
public:
    /**
     * @brief Constructor for CUDA buffer
     * @param device_ptr Device memory pointer
     * @param size Buffer size in bytes
     * @param usage Buffer usage pattern
     * @param access Memory access mode
     * @param stream CUDA stream for operations (optional)
     */
    CudaBuffer(void *device_ptr, size_t size, BufferUsage usage,
               MemoryAccess access, cudaStream_t stream = nullptr);

    virtual ~CudaBuffer();

    // GPUBuffer interface implementation
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
     * @brief Get CUDA device pointer
     */
    void *device_ptr() const {
        return device_ptr_;
    }

    /**
     * @brief Get associated CUDA stream
     */
    cudaStream_t stream() const {
        return stream_;
    }

    /**
     * @brief Set CUDA stream for operations
     */
    void set_stream(cudaStream_t stream) {
        stream_ = stream;
    }

    /**
     * @brief Synchronize with CUDA stream
     */
    void synchronize();

private:
    void *device_ptr_;
    void *host_ptr_;
    size_t size_;
    BufferUsage usage_;
    MemoryAccess access_;
    cudaStream_t stream_;
    bool is_mapped_;
    bool is_unified_memory_;

    void initialize_unified_memory();
    void cleanup();
};

/**
 * @brief CUDA buffer factory
 */
class CudaBufferFactory : public GPUBufferFactory {
public:
    /**
     * @brief Constructor
     * @param device_id CUDA device ID
     */
    explicit CudaBufferFactory(int device_id = 0);

    virtual ~CudaBufferFactory();

    // GPUBufferFactory interface implementation
    std::unique_ptr<GPUBuffer>
    create_buffer(size_t size, BufferUsage usage = BufferUsage::Dynamic,
                  MemoryAccess access = MemoryAccess::Shared) override;

    GPUBackend backend() const override {
        return GPUBackend::CUDA;
    }
    bool supports_unified_memory() const override;
    size_t max_buffer_size() const override;

    /**
     * @brief Get CUDA device ID
     */
    int device_id() const {
        return device_id_;
    }

    /**
     * @brief Create buffer with specific CUDA stream
     */
    std::unique_ptr<CudaBuffer>
    create_buffer_with_stream(size_t size, cudaStream_t stream,
                              BufferUsage usage = BufferUsage::Dynamic,
                              MemoryAccess access = MemoryAccess::Shared);

private:
    int device_id_;
    cudaDeviceProp device_props_;
    bool unified_memory_supported_;

    void initialize_device();
    void *allocate_memory(size_t size, MemoryAccess access);
};

/**
 * @brief CUDA context for managing GPU resources
 */
class CudaContext : public GPUContext {
public:
    /**
     * @brief Constructor
     * @param device_id CUDA device ID (default: 0)
     */
    explicit CudaContext(int device_id = 0);

    virtual ~CudaContext();

    // GPUContext interface implementation
    GPUBackend backend() const override {
        return GPUBackend::CUDA;
    }
    std::unique_ptr<GPUBufferFactory> create_buffer_factory() override;
    std::string device_name() const override;
    size_t total_memory() const override;
    size_t available_memory() const override;
    bool is_unified_memory() const override;
    void synchronize() override;

    /**
     * @brief Get CUDA device ID
     */
    int device_id() const {
        return device_id_;
    }

    /**
     * @brief Get CUDA device properties
     */
    const cudaDeviceProp &device_properties() const {
        return device_props_;
    }

    /**
     * @brief Create CUDA stream
     */
    cudaStream_t create_stream();

    /**
     * @brief Destroy CUDA stream
     */
    void destroy_stream(cudaStream_t stream);

    /**
     * @brief Check if peer-to-peer access is supported
     */
    bool supports_p2p_access(int peer_device_id) const;

    /**
     * @brief Enable peer-to-peer access
     */
    bool enable_p2p_access(int peer_device_id);

private:
    int device_id_;
    cudaDeviceProp device_props_;
    bool initialized_;

    bool initialize();
    void cleanup();
};

/**
 * @brief CUDA utility functions
 */
namespace utils {

/**
 * @brief Get CUDA error string
 */
const char *get_cuda_error_string(cudaError_t error);

/**
 * @brief Check CUDA error and throw exception if failed
 */
void check_cuda_error(cudaError_t error, const char *operation);

/**
 * @brief Get number of CUDA devices
 */
int get_device_count();

/**
 * @brief Check if CUDA is available
 */
bool is_cuda_available();

/**
 * @brief Get CUDA driver version
 */
int get_cuda_driver_version();

/**
 * @brief Get CUDA runtime version
 */
int get_cuda_runtime_version();

} // namespace utils

} // namespace cuda
} // namespace gpu
} // namespace psyne

#endif // PSYNE_CUDA_ENABLED