/**
 * @file compute_kernels.metal
 * @brief Metal compute shaders for GPU operations
 * 
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include <metal_stdlib>
using namespace metal;

/**
 * @brief Scale all elements of a float buffer by a scalar
 */
kernel void scale_float(device float* buffer [[buffer(0)]],
                       constant float& scalar [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    buffer[index] *= scalar;
}

/**
 * @brief Scale all elements of a double buffer by a scalar
 */
kernel void scale_double(device double* buffer [[buffer(0)]],
                        constant double& scalar [[buffer(1)]],
                        uint index [[thread_position_in_grid]]) {
    buffer[index] *= scalar;
}

/**
 * @brief Element-wise addition of two float buffers
 */
kernel void add_float(device float* result [[buffer(0)]],
                     device const float* a [[buffer(1)]],
                     device const float* b [[buffer(2)]],
                     uint index [[thread_position_in_grid]]) {
    result[index] = a[index] + b[index];
}

/**
 * @brief Element-wise addition of two double buffers
 */
kernel void add_double(device double* result [[buffer(0)]],
                      device const double* a [[buffer(1)]],
                      device const double* b [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
    result[index] = a[index] + b[index];
}

/**
 * @brief Matrix multiplication kernel (float)
 * C = A * B where A is MxK, B is KxN, C is MxN
 */
kernel void matmul_float(device float* C [[buffer(0)]],
                        device const float* A [[buffer(1)]],
                        device const float* B [[buffer(2)]],
                        constant uint& M [[buffer(3)]],
                        constant uint& K [[buffer(4)]],
                        constant uint& N [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

/**
 * @brief Optimized matrix multiplication using threadgroup memory
 */
constant uint TILE_SIZE = 16;

kernel void matmul_tiled_float(device float* C [[buffer(0)]],
                              device const float* A [[buffer(1)]],
                              device const float* B [[buffer(2)]],
                              constant uint& M [[buffer(3)]],
                              constant uint& K [[buffer(4)]],
                              constant uint& N [[buffer(5)]],
                              uint2 gid [[thread_position_in_grid]],
                              uint2 tid [[thread_position_in_threadgroup]],
                              uint2 tgid [[threadgroup_position_in_grid]]) {
    
    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];
    
    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (uint tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load A tile
        uint a_col = tile * TILE_SIZE + tid.x;
        if (row < M && a_col < K) {
            A_tile[tid.y][tid.x] = A[row * K + a_col];
        } else {
            A_tile[tid.y][tid.x] = 0.0f;
        }
        
        // Load B tile
        uint b_row = tile * TILE_SIZE + tid.y;
        if (b_row < K && col < N) {
            B_tile[tid.y][tid.x] = B[b_row * N + col];
        } else {
            B_tile[tid.y][tid.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += A_tile[tid.y][k] * B_tile[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * @brief Reduction kernel - sum all elements
 */
kernel void reduce_sum_float(device float* output [[buffer(0)]],
                            device const float* input [[buffer(1)]],
                            threadgroup float* shared [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint block_size [[threads_per_threadgroup]]) {
    
    // Load data to shared memory
    shared[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        output[gid / block_size] = shared[0];
    }
}

/**
 * @brief Apply activation function (ReLU)
 */
kernel void relu_float(device float* buffer [[buffer(0)]],
                      uint index [[thread_position_in_grid]]) {
    buffer[index] = max(buffer[index], 0.0f);
}

/**
 * @brief Apply activation function (Sigmoid)
 */
kernel void sigmoid_float(device float* buffer [[buffer(0)]],
                         uint index [[thread_position_in_grid]]) {
    float x = buffer[index];
    buffer[index] = 1.0f / (1.0f + exp(-x));
}

/**
 * @brief Softmax kernel (numerically stable)
 */
kernel void softmax_float(device float* output [[buffer(0)]],
                         device const float* input [[buffer(1)]],
                         constant uint& size [[buffer(2)]],
                         uint batch_idx [[thread_position_in_grid]]) {
    
    uint offset = batch_idx * size;
    
    // Find max for numerical stability
    float max_val = input[offset];
    for (uint i = 1; i < size; ++i) {
        max_val = max(max_val, input[offset + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < size; ++i) {
        float exp_val = exp(input[offset + i] - max_val);
        output[offset + i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (uint i = 0; i < size; ++i) {
        output[offset + i] /= sum;
    }
}

/**
 * @brief Batch normalization kernel
 */
kernel void batch_norm_float(device float* output [[buffer(0)]],
                            device const float* input [[buffer(1)]],
                            device const float* mean [[buffer(2)]],
                            device const float* variance [[buffer(3)]],
                            device const float* gamma [[buffer(4)]],
                            device const float* beta [[buffer(5)]],
                            constant float& epsilon [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]]) {
    
    uint batch_idx = gid.x;
    uint feature_idx = gid.y;
    
    float x = input[batch_idx * feature_idx];
    float normalized = (x - mean[feature_idx]) / sqrt(variance[feature_idx] + epsilon);
    output[batch_idx * feature_idx] = gamma[feature_idx] * normalized + beta[feature_idx];
}