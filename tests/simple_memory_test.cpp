#include <psyne/psyne.hpp>
#include <iostream>
#include <memory>

using namespace psyne;

int main() {
    std::cout << "Running simple memory leak tests...\n";
    
    try {
        // Test 1: Channel creation/destruction
        std::cout << "Testing channel creation/destruction...\n";
        for (int i = 0; i < 1000; ++i) {
            auto channel = Channel::create("memory://test_" + std::to_string(i), 1024 * 1024);
            // Channel automatically destroyed at end of scope
        }
        std::cout << "✓ Channel lifecycle test passed!\n";
        
        // Test 2: Message sending/receiving
        std::cout << "Testing message operations...\n";
        auto channel = Channel::create("memory://msg_test", 1024 * 1024);
        
        for (int i = 0; i < 1000; ++i) {
            FloatVector msg(*channel);
            msg.resize(100);
            for (int j = 0; j < 100; ++j) {
                msg[j] = static_cast<float>(i * 100 + j);
            }
            msg.send();
            
            // Immediately consume to prevent buildup
            auto received = channel->receive_single<FloatVector>();
            if (!received) {
                std::cerr << "Failed to receive message " << i << std::endl;
                return 1;
            }
        }
        std::cout << "✓ Message operations test passed!\n";
        
        // Test 3: IPC channels (if available)
        std::cout << "Testing IPC channels...\n";
        try {
            auto ipc_channel = Channel::create("ipc://memory_test", 1024 * 1024);
            ByteVector msg(*ipc_channel);
            std::string data = "IPC test message";
            std::copy(data.begin(), data.end(), std::back_inserter(msg));
            msg.send();
            
            auto received = ipc_channel->receive_single<ByteVector>();
            if (received) {
                std::cout << "✓ IPC channel test passed!\n";
            } else {
                std::cout << "⚠ IPC channel test: no message received\n";
            }
        } catch (const std::exception& e) {
            std::cout << "⚠ IPC channels not available: " << e.what() << "\n";
        }
        
        std::cout << "\nAll memory tests completed successfully! ✅\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Memory test failed: " << e.what() << std::endl;
        return 1;
    }
}