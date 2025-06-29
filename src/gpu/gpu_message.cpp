/**
 * @file gpu_message.cpp
 * @brief GPU message operations implementation
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include "gpu_message.hpp"

#ifdef __APPLE__
#include "metal/metal_buffer.hpp"
#endif

#ifdef PSYNE_CUDA_ENABLED
#include "cuda/cuda_buffer.hpp"
#include <cuda_runtime.h>
#endif

namespace psyne {
namespace gpu {

// Explicit instantiation for common types
template class GPUVector<float>;
template class GPUVector<double>;
template class GPUVector<int32_t>;

// GPU scale operation for float
template<>
void GPUVector<float>::gpu_scale(GPUContext& context, float scalar) {
#ifdef __APPLE__
    if (context.backend() == GPUBackend::Metal) {
        auto metal_context = static_cast<metal::MetalContext*>(&context);
        
        // Ensure data is on GPU
        auto gpu_buffer = to_gpu_buffer(context);
        auto metal_buffer = static_cast<metal::MetalBuffer*>(gpu_buffer.get());
        
        // Create compute pipeline
        auto pipeline = metal_context->create_compute_pipeline("scale_float");
        
        // Create command buffer
        metal::MetalCommandBuffer cmd_buffer(metal_context->command_queue());
        auto encoder = cmd_buffer.compute_encoder();
        
        // Set compute pipeline
        encoder->setComputePipelineState(pipeline);
        
        // Set buffers
        encoder->setBuffer(metal_buffer->metal_buffer(), 0, 0);
        encoder->setBytes(&scalar, sizeof(float), 1);
        
        // Calculate thread groups
        MTL::Size grid_size = MTL::Size(size_, 1, 1);
        MTL::Size thread_group_size = MTL::Size(
            std::min(size_t(256), size_), 1, 1
        );
        
        // Dispatch compute
        encoder->dispatchThreads(grid_size, thread_group_size);
        
        // Commit and wait
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
        
        // Clean up
        pipeline->release();
        
        // Mark CPU data as dirty
        is_cpu_dirty_ = true;
    }
#endif

#ifdef PSYNE_CUDA_ENABLED
    if (context.backend() == GPUBackend::CUDA) {
        auto cuda_context = static_cast<cuda::CudaContext*>(&context);
        
        // Ensure data is on GPU
        auto gpu_buffer = to_gpu_buffer(context);
        auto cuda_buffer = static_cast<cuda::CudaBuffer*>(gpu_buffer.get());
        
        // Launch CUDA kernel for scaling
        float* device_data = static_cast<float*>(cuda_buffer->device_ptr());
        
        // Simple CUDA kernel call (would need actual kernel implementation)
        // For now, we'll use a host-side implementation with CUDA memory
        float* host_data = static_cast<float*>(cuda_buffer->map());
        for (size_t i = 0; i < size_; ++i) {
            host_data[i] *= scalar;
        }
        cuda_buffer->unmap();
        cuda_buffer->synchronize();
        
        // Mark CPU data as dirty
        is_cpu_dirty_ = true;
    }
#endif
}

// GPU scale operation for double
template<>
void GPUVector<double>::gpu_scale(GPUContext& context, double scalar) {
#ifdef __APPLE__
    if (context.backend() == GPUBackend::Metal) {
        auto metal_context = static_cast<metal::MetalContext*>(&context);
        
        // Ensure data is on GPU
        auto gpu_buffer = to_gpu_buffer(context);
        auto metal_buffer = static_cast<metal::MetalBuffer*>(gpu_buffer.get());
        
        // Create compute pipeline
        auto pipeline = metal_context->create_compute_pipeline("scale_double");
        
        // Create command buffer
        metal::MetalCommandBuffer cmd_buffer(metal_context->command_queue());
        auto encoder = cmd_buffer.compute_encoder();
        
        // Set compute pipeline
        encoder->setComputePipelineState(pipeline);
        
        // Set buffers
        encoder->setBuffer(metal_buffer->metal_buffer(), 0, 0);
        encoder->setBytes(&scalar, sizeof(double), 1);
        
        // Calculate thread groups
        MTL::Size grid_size = MTL::Size(size_, 1, 1);
        MTL::Size thread_group_size = MTL::Size(
            std::min(size_t(256), size_), 1, 1
        );
        
        // Dispatch compute
        encoder->dispatchThreads(grid_size, thread_group_size);
        
        // Commit and wait
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
        
        // Clean up
        pipeline->release();
        
        // Mark CPU data as dirty
        is_cpu_dirty_ = true;
    }
#endif

#ifdef PSYNE_CUDA_ENABLED
    if (context.backend() == GPUBackend::CUDA) {
        auto cuda_context = static_cast<cuda::CudaContext*>(&context);
        
        // Ensure data is on GPU
        auto gpu_buffer = to_gpu_buffer(context);
        auto cuda_buffer = static_cast<cuda::CudaBuffer*>(gpu_buffer.get());
        
        // Launch CUDA kernel for scaling
        double* device_data = static_cast<double*>(cuda_buffer->device_ptr());
        
        // Simple CUDA kernel call (would need actual kernel implementation)
        // For now, we'll use a host-side implementation with CUDA memory
        double* host_data = static_cast<double*>(cuda_buffer->map());
        for (size_t i = 0; i < size_; ++i) {
            host_data[i] *= scalar;
        }
        cuda_buffer->unmap();
        cuda_buffer->synchronize();
        
        // Mark CPU data as dirty
        is_cpu_dirty_ = true;
    }
#endif
}

// GPU add operation for float
template<>
void GPUVector<float>::gpu_add(GPUContext& context, const GPUVector<float>& other) {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector sizes must match for addition");
    }
    
#ifdef __APPLE__
    if (context.backend() == GPUBackend::Metal) {
        auto metal_context = static_cast<metal::MetalContext*>(&context);
        
        // Ensure both vectors are on GPU
        auto gpu_buffer_a = to_gpu_buffer(context);
        auto gpu_buffer_b = const_cast<GPUVector<float>&>(other).to_gpu_buffer(context);
        
        auto metal_buffer_a = static_cast<metal::MetalBuffer*>(gpu_buffer_a.get());
        auto metal_buffer_b = static_cast<metal::MetalBuffer*>(gpu_buffer_b.get());
        
        // Create compute pipeline
        auto pipeline = metal_context->create_compute_pipeline("add_float");
        
        // Create command buffer
        metal::MetalCommandBuffer cmd_buffer(metal_context->command_queue());
        auto encoder = cmd_buffer.compute_encoder();
        
        // Set compute pipeline
        encoder->setComputePipelineState(pipeline);
        
        // Set buffers (result goes to first buffer)
        encoder->setBuffer(metal_buffer_a->metal_buffer(), 0, 0); // result
        encoder->setBuffer(metal_buffer_a->metal_buffer(), 0, 1); // a
        encoder->setBuffer(metal_buffer_b->metal_buffer(), 0, 2); // b
        
        // Calculate thread groups
        MTL::Size grid_size = MTL::Size(size_, 1, 1);
        MTL::Size thread_group_size = MTL::Size(
            std::min(size_t(256), size_), 1, 1
        );
        
        // Dispatch compute
        encoder->dispatchThreads(grid_size, thread_group_size);
        
        // Commit and wait
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
        
        // Clean up
        pipeline->release();
        
        // Mark CPU data as dirty
        is_cpu_dirty_ = true;
    }
#endif

#ifdef PSYNE_CUDA_ENABLED
    if (context.backend() == GPUBackend::CUDA) {
        auto cuda_context = static_cast<cuda::CudaContext*>(&context);
        
        // Ensure both vectors are on GPU
        auto gpu_buffer_a = to_gpu_buffer(context);
        auto gpu_buffer_b = const_cast<GPUVector<float>&>(other).to_gpu_buffer(context);
        
        auto cuda_buffer_a = static_cast<cuda::CudaBuffer*>(gpu_buffer_a.get());
        auto cuda_buffer_b = static_cast<cuda::CudaBuffer*>(gpu_buffer_b.get());
        
        // Simple CUDA kernel call (would need actual kernel implementation)
        // For now, we'll use a host-side implementation with CUDA memory
        float* host_data_a = static_cast<float*>(cuda_buffer_a->map());
        float* host_data_b = static_cast<float*>(cuda_buffer_b->map());
        
        for (size_t i = 0; i < size_; ++i) {
            host_data_a[i] += host_data_b[i];
        }
        
        cuda_buffer_a->unmap();
        cuda_buffer_b->unmap();
        cuda_buffer_a->synchronize();
        
        // Mark CPU data as dirty
        is_cpu_dirty_ = true;
    }
#endif
}

} // namespace gpu
} // namespace psyne