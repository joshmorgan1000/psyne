/**
 * @file gpu_example.cpp
 * @brief Simple example of GPU substrate usage with zero-copy host-visible memory
 * 
 * This example demonstrates:
 * - Creating channels with GPU substrate
 * - Zero-copy message passing with GPU memory
 * - Automatic backend selection
 * - Host and device memory access
 */

#include <psyne/core/behaviors.hpp>
#include <psyne/channel/substrate/gpu.hpp>
#include <psyne/core/simple_patterns.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>

// Message type that will live in GPU memory
struct GPUMessage {
    int id;
    float values[1024];  // 4KB of data
    char label[64];
    
    void initialize(int msg_id) {
        id = msg_id;
        for (int i = 0; i < 1024; ++i) {
            values[i] = static_cast<float>(msg_id * 1000 + i);
        }
        std::snprintf(label, sizeof(label), "GPU_Message_%d", msg_id);
    }
    
    float compute_sum() const {
        float sum = 0.0f;
        for (int i = 0; i < 1024; ++i) {
            sum += values[i];
        }
        return sum;
    }
};

int main() {
    std::cout << "=== Psyne GPU Substrate Example ===\n\n";
    
    // Check available GPU backends
    auto backends = psyne::substrate::GPU::get_available_backends();
    std::cout << "Available GPU backends: ";
    for (const auto& backend : backends) {
        std::cout << backend << " ";
    }
    std::cout << "\n\n";
    
    if (backends.empty()) {
        std::cerr << "No GPU backends available. "
                  << "Rebuild with PSYNE_CUDA_ENABLED, PSYNE_METAL_ENABLED, "
                  << "or PSYNE_VULKAN_ENABLED\n";
        return 1;
    }
    
    try {
        // For C++ compilation, we need to explicitly use Vulkan
        // Metal requires Objective-C++ compilation
        std::cout << "Creating GPU channel...\n";
        
        // First, let's try to create a GPU substrate directly with Vulkan
        std::unique_ptr<psyne::substrate::GPU> gpu_substrate;
        try {
            // Try default (will attempt Metal on macOS)
            gpu_substrate = std::make_unique<psyne::substrate::GPU>();
            std::cout << "Using " << gpu_substrate->get_backend_name() << " backend\n";
        } catch (const std::exception& e) {
            // If Metal fails, explicitly use Vulkan
            std::cout << "Default backend failed: " << e.what() << "\n";
            std::cout << "Falling back to Vulkan...\n";
            gpu_substrate = std::make_unique<psyne::substrate::GPU>(psyne::substrate::GPUBackend::Vulkan);
            std::cout << "Using " << gpu_substrate->get_backend_name() << " backend\n";
        }
        
        // Now create the channel with a simpler approach
        // Note: ChannelBridge creates its own substrate, so we can't use our test substrate
        auto channel = std::make_shared<psyne::behaviors::ChannelBridge<GPUMessage, 
            psyne::substrate::GPU, psyne::simple_patterns::SimpleSPSC>>();
        
        std::cout << "GPU channel created successfully\n\n";
        
        // Producer thread - writes to GPU memory
        std::thread producer([channel]() {
            std::cout << "Producer: Starting to send messages\n";
            
            for (int i = 0; i < 10; ++i) {
                // Create message directly in GPU memory
                auto msg_lens = channel->create_message();
                msg_lens->initialize(i);
                std::cout << "Producer: Sending message " << i 
                          << " (sum: " << msg_lens->compute_sum() << ")\n";
                channel->send_message(std::move(msg_lens));
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            std::cout << "Producer: Done\n";
        });
        
        // Consumer thread - reads from GPU memory
        std::thread consumer([channel]() {
            std::cout << "Consumer: Starting to receive messages\n";
            
            int received = 0;
            while (received < 10) {
                auto msg_opt = channel->try_receive();
                if (msg_opt.has_value()) {
                    auto& msg = msg_opt.value();
                    std::cout << "Consumer: Received message " << msg->id 
                              << " - " << msg->label
                              << " (sum: " << msg->compute_sum() << ")\n";
                    
                    // Verify data integrity
                    float expected_sum = 0.0f;
                    for (int i = 0; i < 1024; ++i) {
                        expected_sum += msg->id * 1000 + i;
                    }
                    
                    if (std::abs(msg->compute_sum() - expected_sum) > 0.001f) {
                        std::cerr << "Consumer: Data corruption detected!\n";
                    }
                    
                    // MessageLens automatically releases when it goes out of scope
                    received++;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            }
            
            std::cout << "Consumer: Done\n";
        });
        
        // Wait for threads to complete
        producer.join();
        consumer.join();
        
        std::cout << "\nGPU substrate example completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}