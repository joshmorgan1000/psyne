/**
 * @file cuda_kernels.hpp
 * @brief CUDA kernel declarations for GPU vector operations
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#ifdef PSYNE_CUDA_ENABLED

#include <cuda_runtime.h>
#include <stdexcept>

namespace psyne {
namespace gpu {
namespace cuda {

/**
 * @brief Launch CUDA kernel for scaling float vectors
 * @param device_data Device pointer to float data
 * @param scalar Scaling factor
 * @param size Number of elements
 * @param stream CUDA stream (optional, uses default stream if nullptr)
 */
void launch_scale_float_kernel(float *device_data, float scalar, size_t size,
                               cudaStream_t stream = nullptr);

/**
 * @brief Launch CUDA kernel for scaling double vectors
 * @param device_data Device pointer to double data
 * @param scalar Scaling factor
 * @param size Number of elements
 * @param stream CUDA stream (optional, uses default stream if nullptr)
 */
void launch_scale_double_kernel(double *device_data, double scalar, size_t size,
                                cudaStream_t stream = nullptr);

/**
 * @brief Launch CUDA kernel for adding float vectors
 * @param result Device pointer to result array
 * @param a Device pointer to first input array
 * @param b Device pointer to second input array
 * @param size Number of elements
 * @param stream CUDA stream (optional, uses default stream if nullptr)
 */
void launch_add_float_kernel(float *result, const float *a, const float *b,
                             size_t size, cudaStream_t stream = nullptr);

/**
 * @brief Launch CUDA kernel for adding double vectors
 * @param result Device pointer to result array
 * @param a Device pointer to first input array
 * @param b Device pointer to second input array
 * @param size Number of elements
 * @param stream CUDA stream (optional, uses default stream if nullptr)
 */
void launch_add_double_kernel(double *result, const double *a, const double *b,
                              size_t size, cudaStream_t stream = nullptr);

} // namespace cuda
} // namespace gpu
} // namespace psyne

#endif // PSYNE_CUDA_ENABLED