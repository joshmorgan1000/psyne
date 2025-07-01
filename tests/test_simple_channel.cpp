/**
 * @file test_simple_channel.cpp
 * @brief Test channel functionality without boost dependencies
 */

#include "../include/psyne/core/behaviors.hpp"
#include "../include/psyne/core/simple_patterns.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>
#include <cstring>

struct TestMessage {
    uint64_t id;
    char data[56]; // 64 bytes total
    
    TestMessage() : id(0) {
        std::memset(data, 0, sizeof(data));
    }
    
    TestMessage(uint64_t id, const char* msg) : id(id) {
        std::strncpy(data, msg, sizeof(data) - 1);
    }
};

int main() {
    std::cout << "Testing Psyne channel without boost..." << std::endl;
    
    try {
        // Create a simple channel
        using Channel = psyne::behaviors::ChannelBridge<
            TestMessage, 
            psyne::simple_patterns::SimpleInProcess,
            psyne::simple_patterns::SimpleSPSC
        >;
        
        Channel channel(1024 * 1024); // 1MB slab
        
        // Test 1: Send and receive single message
        {
            auto msg = channel.create_message(42, "Hello Psyne!");
            assert(msg->id == 42);
            assert(std::strcmp(msg->data, "Hello Psyne!") == 0);
            channel.send_message(msg);
            
            auto received = channel.try_receive();
            assert(received.has_value());
            assert((*received)->id == 42);
            assert(std::strcmp((*received)->data, "Hello Psyne!") == 0);
            std::cout << "✓ Single message test passed" << std::endl;
        }
        
        // Test 2: Multiple messages
        {
            const int NUM_MESSAGES = 100;
            
            // Send messages
            for (int i = 0; i < NUM_MESSAGES; ++i) {
                auto msg = channel.create_message(i, "Test");
                channel.send_message(msg);
            }
            
            // Receive messages
            for (int i = 0; i < NUM_MESSAGES; ++i) {
                auto received = channel.try_receive();
                assert(received.has_value());
                assert((*received)->id == static_cast<uint64_t>(i));
            }
            
            // Should be empty now
            auto empty = channel.try_receive();
            assert(!empty.has_value());
            std::cout << "✓ Multiple messages test passed" << std::endl;
        }
        
        // Test 3: Producer/Consumer threads
        {
            std::atomic<bool> running{true};
            std::atomic<uint64_t> sent_count{0};
            std::atomic<uint64_t> received_count{0};
            
            // Producer thread
            std::thread producer([&]() {
                while (running) {
                    try {
                        auto msg = channel.create_message(sent_count, "Threaded");
                        channel.send_message(msg);
                        sent_count++;
                        std::this_thread::sleep_for(std::chrono::microseconds(10));
                    } catch (...) {
                        // Channel full, retry
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                }
            });
            
            // Consumer thread
            std::thread consumer([&]() {
                while (running || received_count < sent_count) {
                    auto msg = channel.try_receive();
                    if (msg.has_value()) {
                        // Messages might arrive out of order in a real system
                        // For this test, just count them
                        received_count++;
                    } else {
                        std::this_thread::sleep_for(std::chrono::microseconds(10));
                    }
                }
            });
            
            // Run for a short time
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            running = false;
            
            producer.join();
            consumer.join();
            
            assert(sent_count > 0);
            assert(received_count == sent_count);
            std::cout << "✓ Producer/Consumer test passed (" 
                      << sent_count << " messages)" << std::endl;
        }
        
        std::cout << "\n✅ All channel tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}