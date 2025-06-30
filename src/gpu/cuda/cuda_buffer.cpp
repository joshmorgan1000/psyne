/**
 * @file cuda_buffer.cpp
 * @brief CUDA buffer implementation
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#ifdef PSYNE_CUDA_ENABLED

#include "cuda_buffer.hpp"
#include <cstring>
#include <iostream>
#include <sstream>

namespace psyne {
namespace gpu {
namespace cuda {

// CudaBuffer implementation

CudaBuffer::CudaBuffer(void *device_ptr, size_t size, BufferUsage usage,
                       MemoryAccess access, cudaStream_t stream)
    : device_ptr_(device_ptr), host_ptr_(nullptr), size_(size), usage_(usage),
      access_(access), stream_(stream), is_mapped_(false),
      is_unified_memory_(access == MemoryAccess::Managed) {
    if (!device_ptr_) {
        throw std::runtime_error("Invalid CUDA device pointer");
    }

    if (is_unified_memory_) {
        initialize_unified_memory();
    }
}

CudaBuffer::~CudaBuffer() {
    cleanup();
}

void CudaBuffer::initialize_unified_memory() {
    // For unified memory, device_ptr_ is also accessible from host
    if (access_ == MemoryAccess::Managed) {
        host_ptr_ = device_ptr_;
    }
}

void CudaBuffer::cleanup() {
    if (is_mapped_ && host_ptr_ && !is_unified_memory_) {
        // For non-unified memory, we might have allocated host memory
        cudaFreeHost(host_ptr_);
        host_ptr_ = nullptr;
    }

    if (device_ptr_) {
        if (is_unified_memory_) {
            cudaFree(device_ptr_);
        } else {
            cudaFree(device_ptr_);
        }
        device_ptr_ = nullptr;
    }
}

size_t CudaBuffer::size() const {
    return size_;
}

BufferUsage CudaBuffer::usage() const {
    return usage_;
}

MemoryAccess CudaBuffer::access() const {
    return access_;
}

void *CudaBuffer::map() {
    if (is_mapped_) {
        return host_ptr_;
    }

    if (is_unified_memory_) {
        // Unified memory is directly accessible
        host_ptr_ = device_ptr_;
        is_mapped_ = true;
        return host_ptr_;
    }

    if (access_ == MemoryAccess::DeviceOnly) {
        throw std::runtime_error(
            "Cannot map device-only buffer for CPU access");
    }

    // Allocate pinned host memory for efficient transfers
    cudaError_t error = cudaMallocHost(&host_ptr_, size_);
    if (error != cudaSuccess) {
        utils::check_cuda_error(error, "cudaMallocHost in map()");
    }

    // Copy data from device to host
    error = cudaMemcpyAsync(host_ptr_, device_ptr_, size_,
                            cudaMemcpyDeviceToHost, stream_);
    if (error != cudaSuccess) {
        cudaFreeHost(host_ptr_);
        host_ptr_ = nullptr;
        utils::check_cuda_error(error, "cudaMemcpyAsync in map()");
    }

    // Synchronize if no stream or default stream
    if (stream_ == nullptr) {
        error = cudaDeviceSynchronize();
        utils::check_cuda_error(error, "cudaDeviceSynchronize in map()");
    } else {
        error = cudaStreamSynchronize(stream_);
        utils::check_cuda_error(error, "cudaStreamSynchronize in map()");
    }

    is_mapped_ = true;
    return host_ptr_;
}

void CudaBuffer::unmap() {
    if (!is_mapped_) {
        return;
    }

    if (is_unified_memory_) {
        // For unified memory, just mark as unmapped
        host_ptr_ = nullptr;
        is_mapped_ = false;
        return;
    }

    if (host_ptr_) {
        // Copy data back to device
        cudaError_t error = cudaMemcpyAsync(device_ptr_, host_ptr_, size_,
                                            cudaMemcpyHostToDevice, stream_);
        if (error != cudaSuccess) {
            std::cerr << "Warning: Failed to copy data back to device: "
                      << utils::get_cuda_error_string(error) << std::endl;
        }

        // Free host memory
        cudaFreeHost(host_ptr_);
        host_ptr_ = nullptr;
    }

    is_mapped_ = false;
}

void CudaBuffer::flush() {
    if (is_unified_memory_) {
        // For unified memory, ensure GPU sees the latest data
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "Warning: Failed to synchronize unified memory: "
                      << utils::get_cuda_error_string(error) << std::endl;
        }
    }
    // For non-unified memory, flush happens during unmap
}

void *CudaBuffer::native_handle() {
    return device_ptr_;
}

bool CudaBuffer::is_mapped() const {
    return is_mapped_;
}

void CudaBuffer::upload(const void *data, size_t upload_size, size_t offset) {
    if (offset + upload_size > size_) {
        throw std::out_of_range("Upload exceeds buffer size");
    }

    void *dst = static_cast<uint8_t *>(device_ptr_) + offset;

    cudaError_t error = cudaMemcpyAsync(dst, data, upload_size,
                                        cudaMemcpyHostToDevice, stream_);
    utils::check_cuda_error(error, "cudaMemcpyAsync in upload()");

    if (stream_ == nullptr) {
        error = cudaDeviceSynchronize();
        utils::check_cuda_error(error, "cudaDeviceSynchronize in upload()");
    }
}

void CudaBuffer::download(void *data, size_t download_size, size_t offset) {
    if (offset + download_size > size_) {
        throw std::out_of_range("Download exceeds buffer size");
    }

    void *src = static_cast<uint8_t *>(device_ptr_) + offset;

    cudaError_t error = cudaMemcpyAsync(data, src, download_size,
                                        cudaMemcpyDeviceToHost, stream_);
    utils::check_cuda_error(error, "cudaMemcpyAsync in download()");

    if (stream_ == nullptr) {
        error = cudaDeviceSynchronize();
        utils::check_cuda_error(error, "cudaDeviceSynchronize in download()");
    }
}

void CudaBuffer::synchronize() {
    cudaError_t error;
    if (stream_ == nullptr) {
        error = cudaDeviceSynchronize();
        utils::check_cuda_error(error, "cudaDeviceSynchronize");
    } else {
        error = cudaStreamSynchronize(stream_);
        utils::check_cuda_error(error, "cudaStreamSynchronize");
    }
}

// CudaBufferFactory implementation

CudaBufferFactory::CudaBufferFactory(int device_id)
    : device_id_(device_id), unified_memory_supported_(false) {
    initialize_device();
}

CudaBufferFactory::~CudaBufferFactory() = default;

void CudaBufferFactory::initialize_device() {
    // Set device
    cudaError_t error = cudaSetDevice(device_id_);
    utils::check_cuda_error(error, "cudaSetDevice");

    // Get device properties
    error = cudaGetDeviceProperties(&device_props_, device_id_);
    utils::check_cuda_error(error, "cudaGetDeviceProperties");

    // Check unified memory support
    unified_memory_supported_ = (device_props_.managedMemory != 0);

    std::cout << "CUDA device " << device_id_
              << " initialized: " << device_props_.name << std::endl;
    std::cout << "Unified memory: "
              << (unified_memory_supported_ ? "Yes" : "No") << std::endl;
}

std::unique_ptr<GPUBuffer>
CudaBufferFactory::create_buffer(size_t size, BufferUsage usage,
                                 MemoryAccess access) {
    return create_buffer_with_stream(size, nullptr, usage, access);
}

std::unique_ptr<CudaBuffer> CudaBufferFactory::create_buffer_with_stream(
    size_t size, cudaStream_t stream, BufferUsage usage, MemoryAccess access) {
    if (size == 0) {
        throw std::invalid_argument("Buffer size cannot be zero");
    }

    if (size > max_buffer_size()) {
        throw std::invalid_argument("Buffer size exceeds maximum");
    }

    void *device_ptr = allocate_memory(size, access);
    if (!device_ptr) {
        throw std::runtime_error("Failed to allocate CUDA memory");
    }

    try {
        return std::make_unique<CudaBuffer>(device_ptr, size, usage, access,
                                            stream);
    } catch (...) {
        cudaFree(device_ptr);
        throw;
    }
}

bool CudaBufferFactory::supports_unified_memory() const {
    return unified_memory_supported_;
}

size_t CudaBufferFactory::max_buffer_size() const {
    return device_props_.totalGlobalMem;
}

void *CudaBufferFactory::allocate_memory(size_t size, MemoryAccess access) {
    void *ptr = nullptr;
    cudaError_t error;

    switch (access) {
    case MemoryAccess::DeviceOnly:
    case MemoryAccess::Shared:
        error = cudaMalloc(&ptr, size);
        break;

    case MemoryAccess::Managed:
        if (unified_memory_supported_) {
            error = cudaMallocManaged(&ptr, size);
        } else {
            // Fall back to regular device memory
            error = cudaMalloc(&ptr, size);
        }
        break;

    case MemoryAccess::HostOnly:
        error = cudaMallocHost(&ptr, size);
        break;

    default:
        throw std::invalid_argument("Unsupported memory access mode");
    }

    if (error != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: "
                  << utils::get_cuda_error_string(error) << std::endl;
        return nullptr;
    }

    return ptr;
}

// Utility functions implementation

namespace utils {

const char *get_cuda_error_string(cudaError_t error) {
    return cudaGetErrorString(error);
}

void check_cuda_error(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error in " << operation << ": "
            << get_cuda_error_string(error) << " (" << error << ")";
        throw std::runtime_error(oss.str());
    }
}

int get_device_count() {
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) {
        return 0;
    }
    return count;
}

bool is_cuda_available() {
    return get_device_count() > 0;
}

int get_cuda_driver_version() {
    int version = 0;
    cudaDriverGetVersion(&version);
    return version;
}

int get_cuda_runtime_version() {
    int version = 0;
    cudaRuntimeGetVersion(&version);
    return version;
}

} // namespace utils

} // namespace cuda
} // namespace gpu
} // namespace psyne

#endif // PSYNE_CUDA_ENABLED
