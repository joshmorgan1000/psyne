#include <psyne/psyne.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace psyne;

int main() {
    try {
        std::cout << "Psyne Compression Demo" << std::endl;
        std::cout << "=====================" << std::endl;
        
        // Create compression configuration
        compression::CompressionConfig comp_config;
        comp_config.type = compression::CompressionType::LZ4;
        comp_config.min_size_threshold = 64;  // Compress messages > 64 bytes
        comp_config.level = 1;
        
        // Create TCP server with compression
        std::cout << "\nCreating TCP server with compression enabled..." << std::endl;
        auto server = create_channel("tcp://:8888", 1024*1024, 
                                   ChannelMode::SPSC, ChannelType::SingleType, 
                                   true, comp_config);
        
        // Give server time to start
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Create TCP client with compression
        std::cout << "Creating TCP client with compression enabled..." << std::endl;
        auto client = create_channel("tcp://localhost:8888", 1024*1024,
                                   ChannelMode::SPSC, ChannelType::SingleType,
                                   true, comp_config);
        
        // Wait for connection
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        std::cout << "Testing compression with various message sizes..." << std::endl;
        
        // Test different message sizes
        std::vector<size_t> test_sizes = {32, 64, 128, 512, 1024, 4096};
        
        for (size_t size : test_sizes) {
            std::cout << "\nTesting " << size << " element vectors:" << std::endl;
            
            // Reset metrics for clean measurement
            client->reset_metrics();
            server->reset_metrics();
            
            // Create test data with some patterns (good for compression)
            FloatVector send_msg(*client);
            send_msg.resize(size);
            
            // Fill with repeating pattern
            for (size_t i = 0; i < size; ++i) {
                send_msg[i] = static_cast<float>(i % 10) * 0.1f;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Send message
            client->send(send_msg);
            
            // Receive message
            auto recv_msg = server->receive<FloatVector>();
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            if (recv_msg && recv_msg->size() == size) {
                std::cout << "  ✓ Round-trip successful in " << duration.count() << " μs" << std::endl;
                
                // Verify data integrity
                bool data_ok = true;
                for (size_t i = 0; i < size; ++i) {
                    if (std::abs((*recv_msg)[i] - send_msg[i]) > 1e-6f) {
                        data_ok = false;
                        break;
                    }
                }
                
                if (data_ok) {
                    std::cout << "  ✓ Data integrity verified" << std::endl;
                } else {
                    std::cout << "  ✗ Data corruption detected!" << std::endl;
                }
                
                // Show metrics
                if (client->has_metrics()) {
                    auto send_metrics = client->get_metrics();
                    auto recv_metrics = server->get_metrics();
                    
                    size_t original_size = size * sizeof(float);
                    std::cout << "  Original size: " << original_size << " bytes" << std::endl;
                    std::cout << "  Transmitted: " << send_metrics.bytes_sent << " bytes";
                    
                    if (send_metrics.bytes_sent < original_size) {
                        float compression_ratio = (float)original_size / send_metrics.bytes_sent;
                        std::cout << " (compression ratio: " << compression_ratio << ":1)";
                    }
                    std::cout << std::endl;
                }
            } else {
                std::cout << "  ✗ Failed to receive message or size mismatch" << std::endl;
            }
        }
        
        std::cout << "\n=== Testing without compression ===" << std::endl;
        
        // Test same sizes without compression
        compression::CompressionConfig no_comp_config;
        no_comp_config.type = compression::CompressionType::None;
        
        auto client_no_comp = create_channel("tcp://localhost:8889", 1024*1024,
                                           ChannelMode::SPSC, ChannelType::SingleType,
                                           true, no_comp_config);
        auto server_no_comp = create_channel("tcp://:8889", 1024*1024,
                                           ChannelMode::SPSC, ChannelType::SingleType,
                                           true, no_comp_config);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Test largest size without compression
        size_t test_size = 4096;
        std::cout << "\nTesting " << test_size << " elements without compression:" << std::endl;
        
        client_no_comp->reset_metrics();
        server_no_comp->reset_metrics();
        
        FloatVector send_msg_no_comp(*client_no_comp);
        send_msg_no_comp.resize(test_size);
        for (size_t i = 0; i < test_size; ++i) {
            send_msg_no_comp[i] = static_cast<float>(i % 10) * 0.1f;
        }
        
        client_no_comp->send(send_msg_no_comp);
        auto recv_msg_no_comp = server_no_comp->receive<FloatVector>();
        
        if (recv_msg_no_comp && client_no_comp->has_metrics()) {
            auto metrics = client_no_comp->get_metrics();
            std::cout << "  Transmitted without compression: " << metrics.bytes_sent << " bytes" << std::endl;
        }
        
        std::cout << "\nCompression demo completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}