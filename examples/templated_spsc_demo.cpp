/**
 * @file templated_spsc_demo.cpp
 * @brief Demonstration of zero-copy templated SPSC channels
 * 
 * Shows how messages are allocated directly in channel memory
 * without any copying or runtime headers.
 */

#include "psyne/channel/templated_spsc.hpp"
#include "psyne/core/tensor_message.hpp"
#include "psyne/global/logger.hpp"
#include <chrono>
#include <iostream>
#include <thread>

using namespace psyne;

/**
 * @brief Demo using fixed-size Float32 vector messages
 * 
 * This demonstrates the core zero-copy principle:
 * - Messages allocated directly in channel memory slab
 * - No runtime headers needed for fixed-size messages
 * - Direct Eigen integration for computational efficiency
 */
void demo_float32_vectors() {
    std::cout << "\n=== Float32 Vector Demo (Zero-Copy) ===\n";
    
    // Create channel for 64-dimensional float vectors
    auto channel = make_spsc_channel<Float32VectorMessage<64>>();
    
    // Producer thread
    std::thread producer([&channel]() {
        for (int i = 0; i < 5; ++i) {
            // Allocate message DIRECTLY in channel memory slab
            auto msg = channel->try_send();
            if (msg) {
                // Fill data directly in final location - no copying!
                msg->as_eigen().setRandom();
                msg->batch_idx = i;
                msg->layer_id = 1;
                
                std::cout << "Produced vector " << i 
                          << " (norm: " << msg->as_eigen().norm() << ")\n";
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    
    // Consumer thread
    std::thread consumer([&channel]() {
        int received = 0;
        while (received < 5) {
            auto msg = channel->try_receive();
            if (msg) {
                // Direct access to message in slab memory
                auto eigen_vec = msg->as_eigen();
                
                std::cout << "Consumed vector " << msg->batch_idx 
                          << " from layer " << msg->layer_id
                          << " (mean: " << eigen_vec.mean() << ")\n";
                
                received++;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    // Show channel statistics
    auto stats = channel->get_stats();
    std::cout << "Channel stats:\n";
    std::cout << "  Slab utilization: " << (stats.utilization * 100) << "%\n";
    std::cout << "  Ring utilization: " << (stats.ring_utilization * 100) << "%\n";
}

/**
 * @brief Demo using matrix messages with in-place construction
 */
void demo_matrix_emplace() {
    std::cout << "\n=== Matrix Emplace Demo ===\n";
    
    // Channel for 4x4 matrices
    auto channel = make_spsc_channel<Float32MatrixMessage<4, 4>>();
    
    // Producer: construct messages with specific values
    for (int i = 0; i < 3; ++i) {
        // Emplace constructs the message directly in slab memory
        auto msg = channel->try_emplace();
        if (msg) {
            // Initialize as identity matrix scaled by (i+1)
            msg->as_eigen().setIdentity();
            msg->as_eigen() *= (i + 1);
            msg->batch_idx = i;
            
            std::cout << "Produced matrix " << i << ":\n" 
                      << msg->as_eigen() << "\n\n";
        }
    }
    
    // Consumer: process matrices
    while (!channel->empty()) {
        auto msg = channel->try_receive();
        if (msg) {
            auto matrix = msg->as_eigen();
            float determinant = matrix.determinant();
            
            std::cout << "Consumed matrix " << msg->batch_idx 
                      << " (det: " << determinant << ")\n";
        }
    }
}

/**
 * @brief Performance comparison showing zero-copy benefits
 */
void performance_demo() {
    std::cout << "\n=== Performance Demo ===\n";
    
    auto channel = make_spsc_channel<Float32VectorMessage<1024>>(0, 2048);
    constexpr int NUM_MESSAGES = 10000;
    
    // Producer
    auto start = std::chrono::high_resolution_clock::now();
    
    std::thread producer([&channel]() {
        for (int i = 0; i < NUM_MESSAGES; ++i) {
            while (true) {
                auto msg = channel->try_send();
                if (msg) {
                    // Fill with test data
                    msg->as_eigen().setConstant(i);
                    msg->batch_idx = i;
                    break;
                }
                // Channel full, yield
                std::this_thread::yield();
            }
        }
    });
    
    // Consumer
    std::thread consumer([&channel]() {
        int received = 0;
        while (received < NUM_MESSAGES) {
            auto msg = channel->try_receive();
            if (msg) {
                // Simple processing
                volatile float sum = msg->as_eigen().sum();
                (void)sum; // Prevent optimization
                received++;
            } else {
                std::this_thread::yield();
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double throughput = (NUM_MESSAGES * 1000000.0) / duration.count();
    std::cout << "Processed " << NUM_MESSAGES << " messages in " 
              << duration.count() << " μs\n";
    std::cout << "Throughput: " << throughput << " messages/second\n";
    
    auto stats = channel->get_stats();
    std::cout << "Peak slab utilization: " << (stats.utilization * 100) << "%\n";
}

int main() {
    // Initialize logging
    LogManager::set_level(LogLevel::INFO);
    
    std::cout << "Psyne Templated SPSC Demo\n";
    std::cout << "========================\n";
    std::cout << "Demonstrating zero-copy messaging with direct slab allocation\n";
    
    try {
        demo_float32_vectors();
        demo_matrix_emplace();
        performance_demo();
        
        std::cout << "\nDemo completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}