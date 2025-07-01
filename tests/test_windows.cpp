/**
 * @file test_windows.cpp
 * @brief Windows-compatible test without boost dependencies
 */

#include "../include/psyne/core/platform.hpp"
#include "../include/psyne/core/behaviors.hpp"
#include "../include/psyne/core/simple_patterns.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>
#include <cstring>
#include <atomic>

struct TestMessage {
    uint64_t id;
    char data[56];
    
    TestMessage() : id(0) {
        std::memset(data, 0, sizeof(data));
    }
    
    TestMessage(uint64_t id, const char* msg) : id(id) {
        std::strncpy(data, msg, sizeof(data) - 1);
    }
};

int main() {
    std::cout << "Testing Psyne on Windows..." << std::endl;
    
    try {
        // Test 1: Platform detection
#ifdef PSYNE_PLATFORM_WINDOWS
        std::cout << "✓ Running on Windows" << std::endl;
#elif defined(PSYNE_PLATFORM_MACOS)
        std::cout << "✓ Running on macOS" << std::endl;
#elif defined(PSYNE_PLATFORM_LINUX)
        std::cout << "✓ Running on Linux" << std::endl;
#else
        std::cout << "✓ Running on unknown platform" << std::endl;
#endif

        // Test 2: Basic channel operations
        using Channel = psyne::behaviors::ChannelBridge<
            TestMessage, 
            psyne::simple_patterns::SimpleInProcess,
            psyne::simple_patterns::SimpleSPSC
        >;
        
        Channel channel(1024 * 1024);
        
        // Send and receive
        auto msg = channel.create_message(42, "Hello Windows!");
        channel.send_message(msg);
        
        auto received = channel.try_receive();
        assert(received.has_value());
        assert((*received)->id == 42);
        std::cout << "✓ Basic messaging works" << std::endl;
        
        // Test 3: Multi-threaded operation
        std::atomic<int> counter{0};
        std::atomic<bool> running{true};
        
        std::thread producer([&]() {
            set_thread_name("Producer");
            while (running) {
                try {
                    auto m = channel.create_message(counter++, "Test");
                    channel.send_message(m);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                } catch (...) {
                    // Channel full
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        });
        
        std::thread consumer([&]() {
            set_thread_name("Consumer");
            int received_count = 0;
            while (running || received_count < counter) {
                auto m = channel.try_receive();
                if (m.has_value()) {
                    received_count++;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::cout << "✓ Received " << received_count << " messages" << std::endl;
        });
        
        // Run for 100ms
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        running = false;
        
        producer.join();
        consumer.join();
        
        std::cout << "\n✅ All Windows tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}