#include <psyne/psyne.hpp>
#include <cassert>
#include <iostream>
#include <thread>
#include <chrono>

// Test channel creation and basic operations
int main() {
    std::cout << "Running Channel Tests..." << std::endl;
    
    try {
        // Test 1: Memory channel
        {
            auto channel = psyne::create_channel("memory://test_channel", 1024 * 1024);
            assert(channel != nullptr);
            std::cout << "✓ Memory channel creation" << std::endl;
        }
        
        // Test 2: IPC channel
        {
            auto channel = psyne::create_channel("ipc://test_ipc", 512 * 1024);
            assert(channel != nullptr);
            std::cout << "✓ IPC channel creation" << std::endl;
        }
        
        // Test 3: Unix socket channel
        {
            auto channel = psyne::create_channel("unix:///tmp/psyne_test_socket", 256 * 1024);
            assert(channel != nullptr);
            std::cout << "✓ Unix socket channel creation" << std::endl;
        }
        
        // Test 4: Invalid URI handling
        {
            bool caught_exception = false;
            try {
                auto channel = psyne::create_channel("invalid://uri", 1024);
            } catch (const std::exception&) {
                caught_exception = true;
            }
            assert(caught_exception);
            (void)caught_exception; // Suppress unused variable warning
            std::cout << "✓ Invalid URI exception handling" << std::endl;
        }
        
        // Test 5: Zero buffer size handling
        {
            bool caught_exception = false;
            try {
                auto channel = psyne::create_channel("memory://zero_buffer", 0);
            } catch (const std::exception&) {
                caught_exception = true;
            }
            assert(caught_exception);
            (void)caught_exception; // Suppress unused variable warning
            std::cout << "✓ Zero buffer size exception handling" << std::endl;
        }
        
        // Test 6: Channel modes
        {
            auto channel = psyne::create_channel("memory://mode_test", 1024 * 1024, 
                                                psyne::ChannelMode::SPSC, 
                                                psyne::ChannelType::SingleType);
            assert(channel != nullptr);
            std::cout << "✓ Channel mode specification" << std::endl;
        }
        
        // Test 7: Multi-type channel
        {
            auto channel = psyne::create_channel("memory://multi_type", 1024 * 1024,
                                                psyne::ChannelMode::MPMC,
                                                psyne::ChannelType::MultiType);
            assert(channel != nullptr);
            std::cout << "✓ Multi-type channel creation" << std::endl;
        }
        
        // Test 8: Channel factory reuse
        {
            auto channel1 = psyne::create_channel("memory://reuse_test", 1024 * 1024);
            auto channel2 = psyne::create_channel("memory://reuse_test", 1024 * 1024);
            
            // Should be able to create multiple channels with same URI
            assert(channel1 != nullptr);
            assert(channel2 != nullptr);
            std::cout << "✓ Channel URI reuse" << std::endl;
        }
        
        std::cout << "All Channel Tests Passed! ✅" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}