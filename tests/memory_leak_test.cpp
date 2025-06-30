#include "test_fixtures.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <fstream>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#include <mach/task_info.h>
#endif

using namespace psyne;
using namespace psyne::test;

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
bool has_memory_leak(size_t initial, size_t final, size_t threshold_bytes = 2 * 1024 * 1024) {
    if (final == 0 || initial == 0) {
        return false; // Can't detect on this platform
    }
    return (final > initial + threshold_bytes);
}

size_t get_memory_increase(size_t initial, size_t final) {
    if (final > initial) {
        return final - initial;
    }
    return 0;
}

/**
 * @brief Test that channel creation and destruction doesn't leak memory
 */
void test_channel_creation_destruction() {
    std::cout << "Testing channel creation/destruction for memory leaks..." << std::endl;
    
    size_t initial_memory = get_memory_usage();
    
    // Create and destroy many channels
    for (int i = 0; i < 1000; ++i) {
        auto channel = Channel::create("memory://test_" + std::to_string(i), 1024 * 1024);
        // Channel is automatically destroyed when going out of scope
    }
    
    // Force garbage collection and wait for OS memory reclaim
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Check intermediate memory usage
    size_t intermediate_memory = get_memory_usage();
    std::cout << "  Memory after channel destruction: " << intermediate_memory << " bytes" << std::endl;
    
    // Try to trigger memory reclaim
    for (int i = 0; i < 10; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (get_memory_usage() <= initial_memory + 512 * 1024) { // Allow 512KB tolerance
            break;
        }
    }
    
    size_t final_memory = get_memory_usage();
    
    if (has_memory_leak(initial_memory, final_memory)) {
        std::cerr << "FAIL: Memory leak detected in channel creation/destruction: " 
                  << get_memory_increase(initial_memory, final_memory) << " bytes increased" << std::endl;
        exit(1);
    } else {
        std::cout << "PASS: Channel creation/destruction test" << std::endl;
    }
}

/**
 * @brief Test that message sending doesn't accumulate memory
 */
void test_message_sending() {
    std::cout << "Testing message sending for memory leaks..." << std::endl;
    
    size_t initial_memory = get_memory_usage();
    
    auto channel = Channel::create("memory://message_test", 1024 * 1024);
    const int num_messages = 1000;

    for (int i = 0; i < num_messages; ++i) {
        // Test with byte vector to avoid Message constructor issues
        auto slot = channel->reserve_write_slot(100);
        if (slot != 0xFFFFFFFF) {
            auto span = channel->get_write_span(100);
            // Fill with test data
            for (size_t j = 0; j < 100; ++j) {
                span[j] = static_cast<uint8_t>(i + j);
            }
            channel->notify_message_ready(slot, 100);
        }

        // Consume messages periodically to prevent queue buildup
        if (i % 100 == 0) {
            size_t size;
            uint32_t type;
            while (void* msg_data = channel->receive_raw_message(size, type)) {
                channel->release_raw_message(msg_data);
            }
        }
    }

    // Final cleanup
    size_t size;
    uint32_t type;
    while (void* msg_data = channel->receive_raw_message(size, type)) {
        channel->release_raw_message(msg_data);
    }
    
    // Allow some time for cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    size_t final_memory = get_memory_usage();
    
    if (has_memory_leak(initial_memory, final_memory)) {
        std::cerr << "FAIL: Memory leak detected in message sending: " 
                  << get_memory_increase(initial_memory, final_memory) << " bytes increased" << std::endl;
        exit(1);
    } else {
        std::cout << "PASS: Message sending test" << std::endl;
    }
}

/**
 * @brief Test ring buffer operations for memory leaks
 */
void test_ring_buffer_operations() {
    std::cout << "Testing ring buffer operations for memory leaks..." << std::endl;
    
    size_t initial_memory = get_memory_usage();
    
    auto channel = Channel::create("memory://ring_test", 64 * 1024); // Small buffer
    const int num_operations = 5000;

    for (int i = 0; i < num_operations; ++i) {
        // Create message that will cause buffer wrapping
        std::string data = "Test message " + std::to_string(i) + 
                          " with some additional padding to ensure buffer wrapping occurs";
        
        auto slot = channel->reserve_write_slot(data.size());
        if (slot != 0xFFFFFFFF) {
            auto span = channel->get_write_span(data.size());
            std::memcpy(span.data(), data.c_str(), data.size());
            channel->notify_message_ready(slot, data.size());
        }

        // Immediately consume to force rapid buffer reuse
        size_t size;
        uint32_t type;
        if (void* msg_data = channel->receive_raw_message(size, type)) {
            channel->release_raw_message(msg_data);
        }
    }
    
    // Allow some time for cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    size_t final_memory = get_memory_usage();
    
    if (has_memory_leak(initial_memory, final_memory)) {
        std::cerr << "FAIL: Memory leak detected in ring buffer operations: " 
                  << get_memory_increase(initial_memory, final_memory) << " bytes increased" << std::endl;
        exit(1);
    } else {
        std::cout << "PASS: Ring buffer operations test" << std::endl;
    }
}

/**
 * @brief Test multi-threaded operations for memory leaks
 */
void test_multithreaded_operations() {
    std::cout << "Testing multi-threaded operations for memory leaks..." << std::endl;
    
    size_t initial_memory = get_memory_usage();
    
    const int num_threads = 2;
    const int messages_per_thread = 50;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t, messages_per_thread]() {
            auto channel = Channel::create("memory://thread_test_" + std::to_string(t), 1024 * 1024);

            for (int i = 0; i < messages_per_thread; ++i) {
                std::string data = "Thread " + std::to_string(t) + " message " + std::to_string(i);
                
                auto slot = channel->reserve_write_slot(data.size());
                if (slot != 0xFFFFFFFF) {
                    auto span = channel->get_write_span(data.size());
                    std::memcpy(span.data(), data.c_str(), data.size());
                    channel->notify_message_ready(slot, data.size());
                }

                size_t size;
                uint32_t type;
                if (void* msg_data = channel->receive_raw_message(size, type)) {
                    channel->release_raw_message(msg_data);
                }
            }
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }
    
    // Allow some time for cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    size_t final_memory = get_memory_usage();
    
    if (has_memory_leak(initial_memory, final_memory)) {
        std::cerr << "FAIL: Memory leak detected in multi-threaded operations: " 
                  << get_memory_increase(initial_memory, final_memory) << " bytes increased" << std::endl;
        exit(1);
    } else {
        std::cout << "PASS: Multi-threaded operations test" << std::endl;
    }
}

int main() {
    std::cout << "=== Psyne Memory Leak Tests ===" << std::endl;
    std::cout << "Platform: ";
#ifdef __APPLE__
    std::cout << "macOS" << std::endl;
#elif defined(__linux__)
    std::cout << "Linux" << std::endl;
#else
    std::cout << "Unknown (memory tracking disabled)" << std::endl;
#endif
    
    std::cout << "Initial memory usage: " << get_memory_usage() << " bytes" << std::endl;
    
    try {
        test_channel_creation_destruction();
        test_message_sending();
        test_ring_buffer_operations();
        test_multithreaded_operations();
        
        std::cout << "\n=== All Memory Leak Tests PASSED ===" << std::endl;
        std::cout << "Final memory usage: " << get_memory_usage() << " bytes" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}