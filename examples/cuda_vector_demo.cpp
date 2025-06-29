/**
 * @file cuda_vector_demo.cpp
 * @brief CUDA GPU vector operations demo
 * 
 * Demonstrates using Psyne with CUDA for GPU-accelerated vector operations.
 * Shows zero-copy messaging between CPU and GPU memory.
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include <psyne/psyne.hpp>

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace psyne;
using namespace psyne::gpu;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << " " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void print_vector_info(const std::string& name, const auto& vec) {
    std::cout << name << ": size=" << vec.size();
    if (!vec.empty()) {
        std::cout << ", first=" << vec[0] << ", last=" << vec[vec.size()-1];
    }
    std::cout << std::endl;
}

int main() {
    print_separator("Psyne CUDA GPU Vector Demo");
    
    try {
        // Check if CUDA is available
        auto backends = detect_gpu_backends();
        bool cuda_available = false;
        
        std::cout << "Available GPU backends:" << std::endl;
        for (auto backend : backends) {
            std::cout << "  - " << gpu_backend_name(backend) << std::endl;
            if (backend == GPUBackend::CUDA) {
                cuda_available = true;
            }
        }
        
        if (!cuda_available) {
            std::cout << "\nCUDA not available on this system. Exiting." << std::endl;
            return 0;
        }
        
#ifdef PSYNE_CUDA_ENABLED
        // Create CUDA context
        print_separator("Creating CUDA Context");
        auto gpu_context = create_gpu_context(GPUBackend::CUDA);
        if (!gpu_context) {
            throw std::runtime_error("Failed to create CUDA context");
        }
        
        std::cout << "CUDA device: " << gpu_context->device_name() << std::endl;
        std::cout << "Total memory: " << (gpu_context->total_memory() / (1024*1024)) << " MB" << std::endl;
        std::cout << "Available memory: " << (gpu_context->available_memory() / (1024*1024)) << " MB" << std::endl;
        std::cout << "Unified memory: " << (gpu_context->is_unified_memory() ? "Yes" : "No") << std::endl;
        
        // Create channel for vector communication
        print_separator("Creating Communication Channel");
        auto channel = std::make_unique<Channel>(Transport::Memory, "cuda_demo");
        
        // Create GPU vectors
        print_separator("Creating GPU Vectors");
        const size_t vector_size = 1000000; // 1M elements
        
        GPUFloatVector vec_a(*channel);
        GPUFloatVector vec_b(*channel);
        GPUFloatVector vec_result(*channel);
        
        // Initialize vectors with test data
        std::cout << "Initializing vectors with " << vector_size << " elements..." << std::endl;
        
        vec_a.resize(vector_size);
        vec_b.resize(vector_size);
        vec_result.resize(vector_size);
        
        // Fill with test data
        for (size_t i = 0; i < vector_size; ++i) {
            vec_a[i] = static_cast<float>(i) * 0.001f;
            vec_b[i] = static_cast<float>(i) * 0.002f;
            vec_result[i] = 0.0f;
        }
        
        print_vector_info("Vector A", vec_a);
        print_vector_info("Vector B", vec_b);
        print_vector_info("Vector Result", vec_result);
        
        // Transfer to GPU
        print_separator("GPU Memory Transfer");
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto gpu_buffer_a = vec_a.to_gpu_buffer(*gpu_context);
        auto gpu_buffer_b = vec_b.to_gpu_buffer(*gpu_context);
        auto gpu_buffer_result = vec_result.to_gpu_buffer(*gpu_context);
        
        auto transfer_time = std::chrono::high_resolution_clock::now();
        auto transfer_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            transfer_time - start_time).count();
        
        std::cout << "GPU transfer completed in " << transfer_duration << " µs" << std::endl;
        std::cout << "Vector A on GPU: " << (vec_a.is_on_gpu() ? "Yes" : "No") << std::endl;
        std::cout << "Vector B on GPU: " << (vec_b.is_on_gpu() ? "Yes" : "No") << std::endl;
        
        // GPU scaling operation
        print_separator("GPU Scaling Operation");
        const float scale_factor = 2.5f;
        
        start_time = std::chrono::high_resolution_clock::now();
        vec_a.gpu_scale(*gpu_context, scale_factor);
        auto scale_time = std::chrono::high_resolution_clock::now();
        
        auto scale_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            scale_time - start_time).count();
        
        std::cout << "GPU scaling (" << scale_factor << "x) completed in " 
                  << scale_duration << " µs" << std::endl;
        
        print_vector_info("Scaled Vector A", vec_a);
        
        // GPU vector addition
        print_separator("GPU Vector Addition");
        
        start_time = std::chrono::high_resolution_clock::now();
        vec_result = vec_a; // Copy A to result
        vec_result.gpu_add(*gpu_context, vec_b);
        auto add_time = std::chrono::high_resolution_clock::now();
        
        auto add_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            add_time - start_time).count();
        
        std::cout << "GPU vector addition completed in " << add_duration << " µs" << std::endl;
        print_vector_info("Result Vector", vec_result);
        
        // Memory performance comparison
        print_separator("Performance Comparison");
        
        // CPU version for comparison
        std::vector<float> cpu_a(vector_size), cpu_b(vector_size), cpu_result(vector_size);
        for (size_t i = 0; i < vector_size; ++i) {
            cpu_a[i] = static_cast<float>(i) * 0.001f;
            cpu_b[i] = static_cast<float>(i) * 0.002f;
        }
        
        start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < vector_size; ++i) {
            cpu_a[i] *= scale_factor;
            cpu_result[i] = cpu_a[i] + cpu_b[i];
        }
        auto cpu_time = std::chrono::high_resolution_clock::now();
        
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            cpu_time - start_time).count();
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "CPU operation time:    " << cpu_duration << " µs" << std::endl;
        std::cout << "GPU operation time:    " << (scale_duration + add_duration) << " µs" << std::endl;
        std::cout << "GPU transfer overhead: " << transfer_duration << " µs" << std::endl;
        std::cout << "Total GPU time:        " << (transfer_duration + scale_duration + add_duration) << " µs" << std::endl;
        
        if (cpu_duration > 0) {
            double speedup = static_cast<double>(cpu_duration) / (scale_duration + add_duration);
            std::cout << "GPU compute speedup:   " << speedup << "x" << std::endl;
        }
        
        // Memory usage information
        print_separator("Memory Usage");
        std::cout << "Vector size: " << vector_size << " elements" << std::endl;
        std::cout << "Element size: " << sizeof(float) << " bytes" << std::endl;
        std::cout << "Total memory per vector: " << (vector_size * sizeof(float) / 1024) << " KB" << std::endl;
        std::cout << "Total GPU memory used: " << (3 * vector_size * sizeof(float) / 1024) << " KB" << std::endl;
        
        // Synchronize GPU before exit
        gpu_context->synchronize();
        
        print_separator("Demo Completed Successfully");
        
#else
        std::cout << "CUDA support not compiled in. Please rebuild with CUDA enabled." << std::endl;
        return 1;
#endif
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}