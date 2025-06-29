/**
 * @file cuda_context.cpp
 * @brief CUDA context implementation
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#ifdef PSYNE_CUDA_ENABLED

#include "cuda_buffer.hpp"
#include <iostream>
#include <sstream>

namespace psyne {
namespace gpu {
namespace cuda {

// CudaContext implementation

CudaContext::CudaContext(int device_id)
    : device_id_(device_id)
    , initialized_(false) {
    
    if (!initialize()) {
        throw std::runtime_error("Failed to initialize CUDA context");
    }
}

CudaContext::~CudaContext() {
    cleanup();
}

bool CudaContext::initialize() {
    try {
        // Check if CUDA is available
        if (!utils::is_cuda_available()) {
            std::cerr << "No CUDA devices found" << std::endl;
            return false;
        }
        
        // Validate device ID
        int device_count = utils::get_device_count();
        if (device_id_ < 0 || device_id_ >= device_count) {
            std::cerr << "Invalid CUDA device ID: " << device_id_ 
                      << " (available: 0-" << (device_count - 1) << ")" << std::endl;
            return false;
        }
        
        // Set device
        cudaError_t error = cudaSetDevice(device_id_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to set CUDA device " << device_id_ << ": " 
                      << utils::get_cuda_error_string(error) << std::endl;
            return false;
        }
        
        // Get device properties
        error = cudaGetDeviceProperties(&device_props_, device_id_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to get CUDA device properties: " 
                      << utils::get_cuda_error_string(error) << std::endl;
            return false;
        }
        
        // Initialize device (this creates the CUDA context)
        error = cudaFree(0);
        if (error != cudaSuccess) {
            std::cerr << "Failed to initialize CUDA context: " 
                      << utils::get_cuda_error_string(error) << std::endl;
            return false;
        }
        
        initialized_ = true;
        
        std::cout << "CUDA context initialized successfully" << std::endl;
        std::cout << "Device: " << device_name() << std::endl;
        std::cout << "Compute capability: " << device_props_.major 
                  << "." << device_props_.minor << std::endl;
        std::cout << "Total memory: " << (total_memory() / (1024 * 1024)) 
                  << " MB" << std::endl;
        std::cout << "Unified memory: " << (is_unified_memory() ? "Yes" : "No") << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during CUDA context initialization: " 
                  << e.what() << std::endl;
        return false;
    }
}

void CudaContext::cleanup() {
    if (initialized_) {
        // Reset device to clean up context
        cudaDeviceReset();
        initialized_ = false;
    }
}

std::unique_ptr<GPUBufferFactory> CudaContext::create_buffer_factory() {
    if (!initialized_) {
        throw std::runtime_error("CUDA context not initialized");
    }
    
    return std::make_unique<CudaBufferFactory>(device_id_);
}

std::string CudaContext::device_name() const {
    if (!initialized_) {
        return "Unknown";
    }
    
    return std::string(device_props_.name);
}

size_t CudaContext::total_memory() const {
    if (!initialized_) {
        return 0;
    }
    
    return device_props_.totalGlobalMem;
}

size_t CudaContext::available_memory() const {
    if (!initialized_) {
        return 0;
    }
    
    size_t free_mem = 0;
    size_t total_mem = 0;
    
    cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get CUDA memory info: " 
                  << utils::get_cuda_error_string(error) << std::endl;
        return 0;
    }
    
    return free_mem;
}

bool CudaContext::is_unified_memory() const {
    if (!initialized_) {
        return false;
    }
    
    return device_props_.managedMemory != 0;
}

void CudaContext::synchronize() {
    if (!initialized_) {
        throw std::runtime_error("CUDA context not initialized");
    }
    
    cudaError_t error = cudaDeviceSynchronize();
    utils::check_cuda_error(error, "cudaDeviceSynchronize");
}

cudaStream_t CudaContext::create_stream() {
    if (!initialized_) {
        throw std::runtime_error("CUDA context not initialized");
    }
    
    cudaStream_t stream;
    cudaError_t error = cudaStreamCreate(&stream);
    if (error != cudaSuccess) {
        utils::check_cuda_error(error, "cudaStreamCreate");
    }
    
    return stream;
}

void CudaContext::destroy_stream(cudaStream_t stream) {
    if (!initialized_) {
        return;
    }
    
    if (stream != nullptr) {
        cudaError_t error = cudaStreamDestroy(stream);
        if (error != cudaSuccess) {
            std::cerr << "Warning: Failed to destroy CUDA stream: " 
                      << utils::get_cuda_error_string(error) << std::endl;
        }
    }
}

bool CudaContext::supports_p2p_access(int peer_device_id) const {
    if (!initialized_) {
        return false;
    }
    
    int can_access = 0;
    cudaError_t error = cudaDeviceCanAccessPeer(&can_access, device_id_, peer_device_id);
    if (error != cudaSuccess) {
        return false;
    }
    
    return can_access != 0;
}

bool CudaContext::enable_p2p_access(int peer_device_id) {
    if (!initialized_) {
        return false;
    }
    
    if (!supports_p2p_access(peer_device_id)) {
        return false;
    }
    
    cudaError_t error = cudaDeviceEnablePeerAccess(peer_device_id, 0);
    if (error == cudaErrorPeerAccessAlreadyEnabled) {
        // Already enabled, that's fine
        return true;
    } else if (error != cudaSuccess) {
        std::cerr << "Failed to enable peer access to device " << peer_device_id 
                  << ": " << utils::get_cuda_error_string(error) << std::endl;
        return false;
    }
    
    return true;
}

} // namespace cuda
} // namespace gpu
} // namespace psyne

#endif // PSYNE_CUDA_ENABLED