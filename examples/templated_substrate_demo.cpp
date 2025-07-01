/**
 * @file templated_substrate_demo.cpp
 * @brief Demonstration of compile-time optimized channels
 * 
 * Shows how Channel<MessageType, Substrate, Pattern> provides
 * compile-time optimization for both message type and transport.
 */

#include "psyne/channel/channel_substrate.hpp"
#include "psyne/core/tensor_message.hpp"
#include "psyne/global/logger.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace psyne;

/**
 * @brief Demo compile-time optimization differences
 */
void demo_compile_time_optimization() {
    std::cout << "\n=== Compile-Time Optimization Demo ===\n";
    
    // Different channel configurations with compile-time traits
    using FastChannel = Channel<Float32VectorMessage<64>, Substrate::IN_PROCESS, Pattern::SPSC>;
    using IPCChannel = Channel<Float32VectorMessage<64>, Substrate::IPC, Pattern::SPSC>;
    using NetworkChannel = Channel<Float32VectorMessage<64>, Substrate::TCP, Pattern::SPSC>;
    
    auto fast_channel = std::make_shared<FastChannel>();
    auto ipc_channel = std::make_shared<IPCChannel>();
    auto network_channel = std::make_shared<NetworkChannel>();
    
    // Show compile-time traits
    std::cout << "FastChannel (IN_PROCESS, SPSC):\n";
    std::cout << "  Zero-copy: " << FastChannel::is_zero_copy() << "\n";
    std::cout << "  Needs serialization: " << FastChannel::needs_serialization() << "\n";
    std::cout << "  Cross-process: " << FastChannel::is_cross_process() << "\n";
    std::cout << "  Needs locks: " << FastChannel::needs_locks() << "\n\n";
    
    std::cout << "IPCChannel (IPC, SPSC):\n";
    std::cout << "  Zero-copy: " << IPCChannel::is_zero_copy() << "\n";
    std::cout << "  Needs serialization: " << IPCChannel::needs_serialization() << "\n";
    std::cout << "  Cross-process: " << IPCChannel::is_cross_process() << "\n";
    std::cout << "  Needs locks: " << IPCChannel::needs_locks() << "\n\n";
    
    std::cout << "NetworkChannel (TCP, SPSC):\n";
    std::cout << "  Zero-copy: " << NetworkChannel::is_zero_copy() << "\n";
    std::cout << "  Needs serialization: " << NetworkChannel::needs_serialization() << "\n";
    std::cout << "  Cross-process: " << NetworkChannel::is_cross_process() << "\n";
    std::cout << "  Needs locks: " << NetworkChannel::needs_locks() << "\n\n";
}

/**
 * @brief Demo using the fastest possible configuration
 */
void demo_ultimate_performance() {
    std::cout << "=== Ultimate Performance Demo ===\n";
    
    // Fastest possible: IN_PROCESS + SPSC + Fixed-size message
    using UltimateChannel = Channel<Float32VectorMessage<256>, Substrate::IN_PROCESS, Pattern::SPSC>;
    auto channel = std::make_shared<UltimateChannel>();
    
    constexpr int NUM_MESSAGES = 1000000;
    
    std::thread producer([&channel]() {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < NUM_MESSAGES; ++i) {
            // Message allocated directly in channel slab
            Message<Float32VectorMessage<256>, Substrate::IN_PROCESS, Pattern::SPSC> msg(*channel);
            
            // Fill with data directly in final location
            msg->as_eigen().setConstant(i);
            msg->batch_idx = i;
            
            // Send just updates pointers - zero copy!
            msg.send();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Producer sent " << NUM_MESSAGES << " messages in " 
                  << duration.count() << " μs\n";
        std::cout << "Send rate: " << (NUM_MESSAGES * 1000000.0 / duration.count()) 
                  << " msgs/sec\n";
    });
    
    // Register listener to receive messages
    int received = 0;
    auto start_receive = std::chrono::high_resolution_clock::now();
    
    channel->register_listener([&received, &start_receive](Float32VectorMessage<256>* msg) {
        // Direct access to message in slab memory - no copy!
        volatile float sum = msg->as_eigen().sum();
        (void)sum; // Prevent optimization
        
        received++;
        
        if (received == NUM_MESSAGES) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_receive);
            
            std::cout << "Consumer received " << NUM_MESSAGES << " messages in " 
                      << duration.count() << " μs\n";
            std::cout << "Receive rate: " << (NUM_MESSAGES * 1000000.0 / duration.count()) 
                      << " msgs/sec\n";
        }
    });
    
    producer.join();
    
    // Wait for all messages to be processed
    while (received < NUM_MESSAGES) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

/**
 * @brief Demo different pattern optimizations
 */
void demo_pattern_variations() {
    std::cout << "\n=== Pattern Variations Demo ===\n";
    
    // Show how different patterns affect compilation
    using SPSCChan = Channel<Float32VectorMessage<64>, Substrate::IN_PROCESS, Pattern::SPSC>;
    using MPSCChan = Channel<Float32VectorMessage<64>, Substrate::IN_PROCESS, Pattern::MPSC>;
    using SPMCChan = Channel<Float32VectorMessage<64>, Substrate::IN_PROCESS, Pattern::SPMC>;
    using MPMCChan = Channel<Float32VectorMessage<64>, Substrate::IN_PROCESS, Pattern::MPMC>;
    
    std::cout << "SPSC needs locks: " << SPSCChan::needs_locks() << "\n";
    std::cout << "MPSC needs locks: " << MPSCChan::needs_locks() << "\n";
    std::cout << "SPMC needs locks: " << SPMCChan::needs_locks() << "\n";
    std::cout << "MPMC needs locks: " << MPMCChan::needs_locks() << "\n";
    
    // Create instances to show they compile differently
    auto spsc = std::make_shared<SPSCChan>();
    auto mpsc = std::make_shared<MPSCChan>();
    auto spmc = std::make_shared<SPMCChan>();
    auto mpmc = std::make_shared<MPMCChan>();
    
    std::cout << "All pattern variations created successfully!\n";
}

/**
 * @brief Demo message type specialization
 */
void demo_message_specialization() {
    std::cout << "\n=== Message Type Specialization Demo ===\n";
    
    // Different message types get different optimizations
    using VectorChannel = InProcessSPSC<Float32VectorMessage<128>>;
    using MatrixChannel = InProcessSPSC<Float32MatrixMessage<8, 8>>;
    using GradientChannel = InProcessSPSC<GradientMessage<256>>;
    
    auto vec_channel = std::make_shared<VectorChannel>();
    auto mat_channel = std::make_shared<MatrixChannel>();
    auto grad_channel = std::make_shared<GradientChannel>();
    
    // Show zero-copy message creation
    {
        Message<Float32VectorMessage<128>, Substrate::IN_PROCESS, Pattern::SPSC> vec_msg(*vec_channel);
        vec_msg->as_eigen().setRandom();
        std::cout << "Vector message created and filled in slab memory\n";
        vec_msg.send();
    }
    
    {
        Message<Float32MatrixMessage<8, 8>, Substrate::IN_PROCESS, Pattern::SPSC> mat_msg(*mat_channel);
        mat_msg->as_eigen().setIdentity();
        std::cout << "Matrix message created and filled in slab memory\n";
        mat_msg.send();
    }
    
    {
        Message<GradientMessage<256>, Substrate::IN_PROCESS, Pattern::SPSC> grad_msg(*grad_channel);
        std::fill_n(grad_msg->gradients, 256, 0.01f);
        grad_msg->learning_rate = 0.001f;
        std::cout << "Gradient message created and filled in slab memory\n";
        grad_msg.send();
    }
}

int main() {
    LogManager::set_level(LogLevel::INFO);
    
    std::cout << "Psyne Templated Substrate Demo\n";
    std::cout << "==============================\n";
    std::cout << "Channel<MessageType, Substrate, Pattern> compile-time optimization\n";
    
    try {
        demo_compile_time_optimization();
        demo_pattern_variations();
        demo_message_specialization();
        demo_ultimate_performance();
        
        std::cout << "\nAll demos completed successfully!\n";
        std::cout << "The compiler optimized each channel configuration differently.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}