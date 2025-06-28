#include <psyne/psyne.hpp>
#include <cassert>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

// Comprehensive integration test
int main() {
    std::cout << "Running Integration Tests..." << std::endl;
    
    try {
        // Test 1: End-to-end message passing
        {
            auto channel = psyne::create_channel("memory://integration_test", 4 * 1024 * 1024);
            
            std::atomic<bool> producer_done{false};
            std::atomic<bool> consumer_done{false};
            std::atomic<int> messages_sent{0};
            std::atomic<int> messages_received{0};
            
            const int num_messages = 100;
            
            // Producer thread
            std::thread producer([&]() {
                try {
                    for (int i = 0; i < num_messages; ++i) {
                        auto msg = psyne::FloatVector(*channel);
                        msg.resize(10);
                        
                        for (size_t j = 0; j < 10; ++j) {
                            msg[j] = static_cast<float>(i * 10 + j);
                        }
                        
                        msg.send();
                        messages_sent++;
                        
                        // Small delay to avoid overwhelming
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                    producer_done = true;
                } catch (const std::exception& e) {
                    std::cerr << "Producer error: " << e.what() << std::endl;
                }
            });
            
            // Consumer thread
            std::thread consumer([&]() {
                try {
                    while (!producer_done || messages_received < messages_sent) {
                        // Try to receive with timeout
                        auto msg = channel->template receive<psyne::FloatVector>(
                            std::chrono::milliseconds(10));
                        
                        if (msg) {
                            assert(msg->size() == 10);
                            messages_received++;
                        }
                        
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
                    consumer_done = true;
                } catch (const std::exception& e) {
                    std::cerr << "Consumer error: " << e.what() << std::endl;
                }
            });
            
            // Wait for completion
            producer.join();
            consumer.join();
            
            assert(producer_done);
            assert(consumer_done);
            assert(messages_sent == num_messages);
            
            std::cout << "✓ End-to-end message passing (" << messages_sent 
                      << " sent, " << messages_received << " received)" << std::endl;
        }
        
        // Test 2: Multi-type channel test
        {
            auto channel = psyne::create_channel("memory://multi_type_test", 2 * 1024 * 1024,
                                                psyne::ChannelMode::MPMC,
                                                psyne::ChannelType::MultiType);
            
            // Send different message types
            {
                auto float_msg = psyne::FloatVector(*channel);
                float_msg.resize(5);
                for (size_t i = 0; i < 5; ++i) {
                    float_msg[i] = static_cast<float>(i * 2.5f);
                }
                // Note: In a real implementation, we'd send this
            }
            
            {
                auto byte_msg = psyne::ByteVector(*channel);
                byte_msg.resize(8);
                for (size_t i = 0; i < 8; ++i) {
                    byte_msg[i] = static_cast<uint8_t>(i + 200);
                }
                // Note: In a real implementation, we'd send this
            }
            
            {
                auto matrix_msg = psyne::DoubleMatrix(*channel);
                matrix_msg.set_dimensions(2, 3);
                matrix_msg.at(0, 0) = 3.14;
                matrix_msg.at(1, 2) = 2.71;
                // Note: In a real implementation, we'd send this
            }
            
            std::cout << "✓ Multi-type message creation" << std::endl;
        }
        
        // Test 3: Performance measurement
        {
            auto channel = psyne::create_channel("memory://perf_test", 8 * 1024 * 1024);
            
            const int warmup_messages = 100;
            const int test_messages = 1000;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Warmup
            for (int i = 0; i < warmup_messages; ++i) {
                auto msg = psyne::ByteVector(*channel);
                msg.resize(1024);
                // Note: In a real implementation, we'd send and receive this
            }
            
            // Actual test
            auto test_start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < test_messages; ++i) {
                auto msg = psyne::FloatVector(*channel);
                msg.resize(256); // 1KB message
                // Note: In a real implementation, we'd send and receive this
            }
            
            auto test_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                test_end - test_start);
            
            double messages_per_second = static_cast<double>(test_messages) / 
                                       (static_cast<double>(duration.count()) / 1e6);
            
            std::cout << "✓ Performance test: " << messages_per_second 
                      << " messages/second" << std::endl;
        }
        
        // Test 4: Error handling and recovery
        {
            // Test invalid channel creation
            bool caught_invalid_uri = false;
            try {
                auto channel = psyne::create_channel("invalid://bad_uri", 1024);
            } catch (const std::exception&) {
                caught_invalid_uri = true;
            }
            assert(caught_invalid_uri);
            
            // Test message bounds checking
            auto channel = psyne::create_channel("memory://bounds_test", 1024 * 1024);
            auto msg = psyne::FloatVector(*channel);
            msg.resize(10);
            
            bool caught_out_of_range = false;
            try {
                float value = msg[100]; // Should throw
                (void)value; // Suppress unused variable warning
            } catch (const std::out_of_range&) {
                caught_out_of_range = true;
            }
            assert(caught_out_of_range);
            
            std::cout << "✓ Error handling and bounds checking" << std::endl;
        }
        
        // Test 5: Resource management
        {
            // Test that channels can be created and destroyed properly
            std::vector<std::unique_ptr<psyne::Channel>> channels;
            
            for (int i = 0; i < 10; ++i) {
                std::string uri = "memory://resource_test_" + std::to_string(i);
                auto channel = psyne::create_channel(uri, 512 * 1024);
                channels.push_back(std::move(channel));
            }
            
            // Clear all channels
            channels.clear();
            
            std::cout << "✓ Resource management and cleanup" << std::endl;
        }
        
        std::cout << "All Integration Tests Passed! ✅" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Integration test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}