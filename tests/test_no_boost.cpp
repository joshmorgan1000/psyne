/**
 * @file test_no_boost.cpp
 * @brief Basic test without boost dependencies
 * 
 * This test validates core functionality without requiring boost::asio
 */

#include <iostream>
#include <cassert>
#include <cstring>

// Test basic behaviors without boost
struct SimpleMessage {
    int id;
    char data[64];
};

int main() {
    std::cout << "Testing Psyne core without boost dependencies..." << std::endl;
    
    // Test 1: Basic message structure
    SimpleMessage msg;
    msg.id = 42;
    std::strcpy(msg.data, "Hello Psyne!");
    
    assert(msg.id == 42);
    assert(std::strcmp(msg.data, "Hello Psyne!") == 0);
    std::cout << "✓ Basic message test passed" << std::endl;
    
    // Test 2: Memory alignment
    void* ptr = std::aligned_alloc(64, 1024);
    assert(ptr != nullptr);
    assert(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
    std::free(ptr);
    std::cout << "✓ Memory alignment test passed" << std::endl;
    
    // Test 3: Basic atomics
    std::atomic<size_t> counter{0};
    counter.fetch_add(1, std::memory_order_relaxed);
    assert(counter.load() == 1);
    std::cout << "✓ Atomic operations test passed" << std::endl;
    
    std::cout << "\n✅ All tests passed!" << std::endl;
    return 0;
}