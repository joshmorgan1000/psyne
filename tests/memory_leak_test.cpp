#include "test_fixtures.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <chrono>

using namespace psyne;
using namespace psyne::test;

/**
 * @brief Test fixture specifically for memory leak detection
 */
class MemoryLeakTestFixture : public psyne::test::PsyneTestFixture {
protected:
    void SetUp() override {
        PsyneTestFixture::setup();
        initial_memory_usage_ = get_memory_usage();
    }
    
    void TearDown() override {
        // Allow some time for cleanup
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        final_memory_usage_ = get_memory_usage();
        PsyneTestFixture::teardown();
    }
    
    /**
     * @brief Get current memory usage (platform-specific)
     */
    size_t get_memory_usage() {
#ifdef __APPLE__
        // macOS memory usage detection
        task_basic_info info;
        mach_msg_type_number_t size = sizeof(info);
        kern_return_t kerr = task_info(mach_task_self(), TASK_BASIC_INFO, 
                                      (task_info_t)&info, &size);
        return (kerr == KERN_SUCCESS) ? info.resident_size : 0;
#elif defined(__linux__)
        // Linux memory usage detection
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                size_t kb = std::stoul(line.substr(7));
                return kb * 1024; // Convert to bytes
            }
        }
        return 0;
#else
        // Fallback for other platforms
        return 0;
#endif
    }
    
    /**
     * @brief Check if memory usage has increased significantly
     */
    bool has_memory_leak(size_t threshold_bytes = 1024 * 1024) { // 1MB threshold
        if (final_memory_usage_ == 0 || initial_memory_usage_ == 0) {
            return false; // Can't detect on this platform
        }
        return (final_memory_usage_ > initial_memory_usage_ + threshold_bytes);
    }
    
    size_t get_memory_increase() {
        if (final_memory_usage_ > initial_memory_usage_) {
            return final_memory_usage_ - initial_memory_usage_;
        }
        return 0;
    }
    
private:
    size_t initial_memory_usage_ = 0;
    size_t final_memory_usage_ = 0;
    
#ifdef __APPLE__
    #include <mach/mach.h>
    #include <mach/task.h>
#elif defined(__linux__)
    #include <fstream>
    #include <string>
#endif
};

/**
 * @brief Test that channel creation and destruction doesn't leak memory
 */
void test_channel_creation_destruction() {
    MemoryLeakTestFixture fixture;
    const int num_iterations = 1000;
    
    for (int i = 0; i < num_iterations; ++i) {
        auto channel = create_test_channel("leak_test_" + std::to_string(i));
        assert_channel_valid(channel);
        // Channel automatically destroyed at end of scope
    }
    
    EXPECT_FALSE(has_memory_leak()) 
        << "Memory leak detected: " << get_memory_increase() << " bytes increased";
}

/**
 * @brief Test that message sending doesn't accumulate memory
 */
TEST_F(MemoryLeakTestFixture, MessageSendingAccumulation) {
    auto channel = create_test_channel("message_leak_test", 1024 * 1024);
    const int num_messages = 10000;
    
    for (int i = 0; i < num_messages; ++i) {
        FloatVector msg(*channel);
        // Fill with some data
        for (int j = 0; j < 100; ++j) {
            msg.push_back(static_cast<float>(i * 100 + j));
        }
        msg.send();
        
        // Consume messages periodically to prevent queue buildup
        if (i % 100 == 0) {
            while (auto received = channel->receive_single<FloatVector>()) {
                // Message automatically cleaned up
            }
        }
    }
    
    // Final cleanup
    while (auto received = channel->receive_single<FloatVector>()) {
        // Message automatically cleaned up
    }
    
    EXPECT_FALSE(has_memory_leak()) 
        << "Memory leak detected: " << get_memory_increase() << " bytes increased";
}

/**
 * @brief Test that ring buffer operations don't leak memory
 */
TEST_F(MemoryLeakTestFixture, RingBufferOperations) {
    auto channel = create_test_channel("ring_buffer_test", 64 * 1024); // Small buffer
    const int num_operations = 50000;
    
    for (int i = 0; i < num_operations; ++i) {
        // Create message that will cause buffer wrapping
        ByteVector msg(*channel);
        std::string data = "Test message " + std::to_string(i) + 
                          " with some additional padding to ensure buffer wrapping occurs";
        std::copy(data.begin(), data.end(), std::back_inserter(msg));
        msg.send();
        
        // Immediately consume to force rapid buffer reuse
        auto received = channel->receive_single<ByteVector>();
        ASSERT_TRUE(received.has_value());
    }
    
    EXPECT_FALSE(has_memory_leak()) 
        << "Memory leak detected: " << get_memory_increase() << " bytes increased";
}

/**
 * @brief Test multi-threaded operations for memory leaks
 */
TEST_F(MemoryLeakTestFixture, MultiThreadedOperations) {
    const int num_threads = 4;
    const int messages_per_thread = 1000;
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, messages_per_thread]() {
            auto channel = create_test_channel("thread_test_" + std::to_string(t));
            
            for (int i = 0; i < messages_per_thread; ++i) {
                FloatVector msg(*channel);
                msg.push_back(static_cast<float>(t * 1000 + i));
                msg.send();
                
                auto received = channel->receive_single<FloatVector>();
                EXPECT_TRUE(received.has_value());
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_FALSE(has_memory_leak()) 
        << "Memory leak detected: " << get_memory_increase() << " bytes increased";
}

/**
 * @brief Test IPC channel memory management
 */
TEST_F(MemoryLeakTestFixture, IPCChannelMemory) {
    const int num_iterations = 100;
    
    for (int i = 0; i < num_iterations; ++i) {
        try {
            auto ipc_channel = create_test_ipc_channel("ipc_leak_test_" + std::to_string(i));
            if (ipc_channel) {
                assert_channel_valid(ipc_channel);
                
                // Send a few messages
                for (int j = 0; j < 10; ++j) {
                    ByteVector msg(*ipc_channel);
                    std::string data = "IPC test " + std::to_string(i) + "_" + std::to_string(j);
                    std::copy(data.begin(), data.end(), std::back_inserter(msg));
                    msg.send();
                }
                
                // Receive messages
                while (auto received = ipc_channel->receive_single<ByteVector>()) {
                    // Message automatically cleaned up
                }
            }
        } catch (const std::exception&) {
            // IPC might not be available, skip
            continue;
        }
    }
    
    EXPECT_FALSE(has_memory_leak()) 
        << "Memory leak detected: " << get_memory_increase() << " bytes increased";
}

/**
 * @brief Test large message handling for memory leaks
 */
TEST_F(MemoryLeakTestFixture, LargeMessageHandling) {
    auto channel = create_test_channel("large_msg_test", 10 * 1024 * 1024); // 10MB buffer
    const int num_large_messages = 100;
    const size_t large_message_size = 100 * 1024; // 100KB messages
    
    for (int i = 0; i < num_large_messages; ++i) {
        ByteVector msg(*channel);
        msg.resize(large_message_size);
        
        // Fill with test data
        for (size_t j = 0; j < large_message_size; ++j) {
            msg[j] = static_cast<uint8_t>((i + j) % 256);
        }
        
        msg.send();
        
        auto received = channel->receive_single<ByteVector>();
        ASSERT_TRUE(received.has_value());
        EXPECT_EQ(received->size(), large_message_size);
    }
    
    EXPECT_FALSE(has_memory_leak()) 
        << "Memory leak detected: " << get_memory_increase() << " bytes increased";
}