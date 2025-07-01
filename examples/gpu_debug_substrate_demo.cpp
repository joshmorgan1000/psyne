/**
 * @file gpu_debug_substrate_demo.cpp
 * @brief Demonstration of GPU and Debug substrates in the unified channel system
 * 
 * Shows how Channel<MessageType, GPU, Pattern> and Channel<MessageType, Debug, Pattern>
 * provide specialized behavior while maintaining the same API.
 */

#include "psyne/channel/channel.hpp"
#include "psyne/core/tensor_message.hpp"
#include "psyne/global/logger.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace psyne;

/**
 * @brief Demo GPU substrate with different memory types
 */
void demo_gpu_substrate() {
    std::cout << "\n=== GPU Substrate Demo ===\n";
    
    // GPU channel for tensor operations
    using GPUTensorChannel = Channel<Float32VectorMessage<512>, substrate::GPU, pattern::SPSC>;
    auto gpu_channel = std::make_shared<GPUTensorChannel>();
    
    std::cout << "GPU Channel traits:\n";
    std::cout << "  Zero-copy: " << GPUTensorChannel::is_zero_copy() << "\n";
    std::cout << "  Needs serialization: " << GPUTensorChannel::needs_serialization() << "\n";
    std::cout << "  Cross-process: " << GPUTensorChannel::is_cross_process() << "\n";
    std::cout << "  Needs locks: " << GPUTensorChannel::needs_locks() << "\n\n";
    
    // Register a listener that could be called from GPU kernels
    gpu_channel->register_listener([](Float32VectorMessage<512>* msg) {
        std::cout << "GPU: Received tensor from batch " << msg->batch_idx 
                  << " with norm " << msg->as_eigen().norm() << "\n";
        
#ifdef PSYNE_CUDA_ENABLED
        // Example: This data could be processed by CUDA kernels
        std::cout << "GPU: Tensor data is GPU-accessible for kernel processing\n";
#else
        std::cout << "GPU: CUDA not enabled, using host memory fallback\n";
#endif
    });
    
    // Producer: Create messages in GPU memory
    std::thread gpu_producer([&gpu_channel]() {
        for (int i = 0; i < 3; ++i) {
            try {
                // Message allocated in GPU-accessible memory
                Message<Float32VectorMessage<512>, substrate::GPU, pattern::SPSC> msg(*gpu_channel);
                
                // Fill with test data - this writes directly to GPU memory
                msg->as_eigen().setRandom();
                msg->batch_idx = i;
                msg->layer_id = 100 + i;
                
                std::cout << "GPU Producer: Created tensor " << i << " in GPU memory\n";
                
                // Send - data stays in GPU memory, just notifies listeners
                msg.send();
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
            } catch (const std::exception& e) {
                std::cerr << "GPU Producer error: " << e.what() << "\n";
            }
        }
    });
    
    gpu_producer.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

/**
 * @brief Demo Debug substrate with instrumentation
 */
void demo_debug_substrate() {
    std::cout << "\n=== Debug Substrate Demo ===\n";
    
    // Debug channel with instrumentation
    using DebugMatrixChannel = Channel<Float32MatrixMessage<4, 4>, substrate::Debug, pattern::SPSC>;
    auto debug_channel = std::make_shared<DebugMatrixChannel>();
    
    std::cout << "Debug Channel traits:\n";
    std::cout << "  Zero-copy: " << DebugMatrixChannel::is_zero_copy() << "\n";
    std::cout << "  Needs serialization: " << DebugMatrixChannel::needs_serialization() << "\n";
    std::cout << "  Cross-process: " << DebugMatrixChannel::is_cross_process() << "\n";
    std::cout << "  Needs locks: " << DebugMatrixChannel::needs_locks() << "\n\n";
    
    // Reset debug statistics
    substrate::Debug::reset_stats();
    
    // Register listeners to see debug instrumentation
    debug_channel->register_listener([](Float32MatrixMessage<4, 4>* msg) {
        std::cout << "Debug Listener 1: Processing matrix from batch " << msg->batch_idx << "\n";
        // Simulate some processing time
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    });
    
    debug_channel->register_listener([](Float32MatrixMessage<4, 4>* msg) {
        std::cout << "Debug Listener 2: Matrix determinant = " << msg->as_eigen().determinant() << "\n";
        // Simulate more processing time
        std::this_thread::sleep_for(std::chrono::microseconds(30));
    });
    
    // Producer: Create messages with debug instrumentation
    std::cout << "Creating debug messages (watch for instrumentation):\n";
    
    for (int i = 0; i < 4; ++i) {
        try {
            // Message creation is instrumented by Debug substrate
            Message<Float32MatrixMessage<4, 4>, substrate::Debug, pattern::SPSC> msg(*debug_channel);
            
            // Create test matrix
            msg->as_eigen().setIdentity();
            msg->as_eigen() *= (i + 1);
            msg->batch_idx = i;
            msg->layer_id = 200 + i;
            
            std::cout << "Debug Producer: Created matrix " << i << "\n";
            
            // Send is instrumented - shows timing and listener info
            msg.send();
            
        } catch (const std::exception& e) {
            std::cerr << "Debug Producer error: " << e.what() << "\n";
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Show debug statistics
    std::cout << "\nDebug Statistics:\n";
    std::cout << "  Total allocations: " << substrate::Debug::get_allocation_count() << "\n";
    std::cout << "  Total sends: " << substrate::Debug::get_send_count() << "\n";
    std::cout << "  Total bytes allocated: " << substrate::Debug::get_total_bytes_allocated() << "\n";
}

/**
 * @brief Demo combining GPU and Debug with different patterns
 */
void demo_advanced_combinations() {
    std::cout << "\n=== Advanced Substrate/Pattern Combinations ===\n";
    
    // Different combinations show compile-time optimization
    using FastGPU = Channel<Float32VectorMessage<64>, substrate::GPU, pattern::SPSC>;
    using DebugMPSC = Channel<Float32VectorMessage<64>, substrate::Debug, pattern::MPSC>;
    using GPUBroadcast = Channel<Float32VectorMessage<64>, substrate::GPU, pattern::SPMC>;
    
    auto fast_gpu = std::make_shared<FastGPU>();
    auto debug_mpsc = std::make_shared<DebugMPSC>();
    auto gpu_broadcast = std::make_shared<GPUBroadcast>();
    
    std::cout << "FastGPU (GPU + SPSC): " 
              << "zero-copy=" << FastGPU::is_zero_copy()
              << ", needs_locks=" << FastGPU::needs_locks() << "\n";
              
    std::cout << "DebugMPSC (Debug + MPSC): "
              << "zero-copy=" << DebugMPSC::is_zero_copy() 
              << ", needs_locks=" << DebugMPSC::needs_locks() << "\n";
              
    std::cout << "GPUBroadcast (GPU + SPMC): "
              << "zero-copy=" << GPUBroadcast::is_zero_copy()
              << ", needs_locks=" << GPUBroadcast::needs_locks() << "\n";
    
    // Each combination compiles to different optimized code
    std::cout << "All combinations created successfully with compile-time optimization!\n";
}

/**
 * @brief Performance comparison between substrates
 */
void demo_substrate_performance() {
    std::cout << "\n=== Substrate Performance Comparison ===\n";
    
    constexpr int NUM_MESSAGES = 1000;
    
    // Compare InProcess vs Debug overhead
    using FastChannel = Channel<Float32VectorMessage<128>, substrate::InProcess, pattern::SPSC>;
    using DebugChannel = Channel<Float32VectorMessage<128>, substrate::Debug, pattern::SPSC>;
    
    auto fast_channel = std::make_shared<FastChannel>();
    auto debug_channel = std::make_shared<DebugChannel>();
    
    // Benchmark fast channel
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_MESSAGES; ++i) {
        Message<Float32VectorMessage<128>, substrate::InProcess, pattern::SPSC> msg(*fast_channel);
        msg->as_eigen().setConstant(i);
        msg.send();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto fast_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Benchmark debug channel (with instrumentation overhead)
    substrate::Debug::reset_stats();
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_MESSAGES; ++i) {
        Message<Float32VectorMessage<128>, substrate::Debug, pattern::SPSC> msg(*debug_channel);
        msg->as_eigen().setConstant(i);
        msg.send();
    }
    end = std::chrono::high_resolution_clock::now();
    auto debug_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Performance Results (" << NUM_MESSAGES << " messages):\n";
    std::cout << "  InProcess: " << fast_duration.count() << " μs\n";
    std::cout << "  Debug: " << debug_duration.count() << " μs\n";
    std::cout << "  Debug overhead: " << (debug_duration.count() - fast_duration.count()) << " μs\n";
    std::cout << "  Debug sends tracked: " << substrate::Debug::get_send_count() << "\n";
}

int main() {
    LogManager::set_level(LogLevel::DEBUG);
    
    std::cout << "Psyne GPU and Debug Substrate Demo\n";
    std::cout << "==================================\n";
    std::cout << "Demonstrating unified Channel<MessageType, Substrate, Pattern> system\n";
    
    try {
        demo_gpu_substrate();
        demo_debug_substrate();
        demo_advanced_combinations();
        demo_substrate_performance();
        
        std::cout << "\nAll demos completed successfully!\n";
        std::cout << "GPU and Debug substrates are fully integrated into the unified system.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}