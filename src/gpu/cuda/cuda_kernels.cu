/**
 * @file cuda_kernels.cu
 * @brief CUDA kernels for GPU vector operations
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include "cuda_kernels.hpp"
#include <cuda_runtime.h>

namespace psyne {
namespace gpu {
namespace cuda {

/**
 * @brief CUDA kernel for scaling float vectors
 */
__global__ void scale_float_kernel(float *data, float scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scalar;
    }
}

/**
 * @brief CUDA kernel for scaling double vectors
 */
__global__ void scale_double_kernel(double *data, double scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scalar;
    }
}

/**
 * @brief CUDA kernel for adding float vectors
 */
__global__ void add_float_kernel(float *result, const float *a, const float *b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief CUDA kernel for adding double vectors
 */
__global__ void add_double_kernel(double *result, const double *a, const double *b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// Host wrapper functions

void launch_scale_float_kernel(float *device_data, float scalar, size_t size, cudaStream_t stream) {
    if (size == 0) return;
    
    // Calculate grid and block dimensions
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel
    scale_float_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(device_data, scalar, size);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(error)));
    }
}

void launch_scale_double_kernel(double *device_data, double scalar, size_t size, cudaStream_t stream) {
    if (size == 0) return;
    
    // Calculate grid and block dimensions
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel
    scale_double_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(device_data, scalar, size);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(error)));
    }
}

void launch_add_float_kernel(float *result, const float *a, const float *b, size_t size, cudaStream_t stream) {
    if (size == 0) return;
    
    // Calculate grid and block dimensions
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel
    add_float_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(result, a, b, size);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(error)));
    }
}

void launch_add_double_kernel(double *result, const double *a, const double *b, size_t size, cudaStream_t stream) {
    if (size == 0) return;
    
    // Calculate grid and block dimensions
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel
    add_double_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(result, a, b, size);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(error)));
    }
}

} // namespace cuda
} // namespace gpu
} // namespace psyne