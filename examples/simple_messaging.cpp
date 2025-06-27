#include <psyne/psyne.hpp>
#include <iostream>
#include <chrono>

using namespace psyne;

int main() {
    std::cout << "Psyne Simple Messaging Example\n";
    std::cout << "==============================\n\n";
    
    // Create a single-type channel optimized for FloatVector
    SPSCChannel channel("local://simple", 1024 * 1024, ChannelType::SingleType);
    
    std::cout << "Channel created with 1MB buffer\n";
    
    // Demonstrate zero-copy write
    std::cout << "\n1. Creating message directly in buffer (zero allocation)...\n";
    std::cout << "   Required size for FloatVector: " << FloatVector::calculate_size() << " bytes\n";
    
    FloatVector msg(channel);
    
    if (!msg.is_valid()) {
        std::cerr << "Failed to allocate message\n";
        return 1;
    }
    
    std::cout << "   Message is valid, size: " << msg.size() << ", capacity: " << msg.capacity() << "\n";
    
    // Write data directly into the buffer
    msg = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    std::cout << "   Wrote " << msg.size() << " floats directly to buffer\n";
    std::cout << "   Buffer address: " << static_cast<void*>(msg.begin()) << "\n";
    
    // Send the message
    channel.send(msg);
    std::cout << "   Message sent\n";
    
    // Receive the message (zero-copy)
    std::cout << "\n2. Receiving message (zero-copy view)...\n";
    auto received = channel.receive_single<FloatVector>();
    
    if (received) {
        std::cout << "   Received " << received->size() << " floats\n";
        std::cout << "   Buffer address: " << static_cast<const void*>(received->begin()) << "\n";
        std::cout << "   Values: ";
        for (float val : *received) {
            std::cout << val << " ";
        }
        std::cout << "\n";
        
        // Verify zero-copy by checking addresses
        std::cout << "\n3. Verifying zero-copy...\n";
        std::cout << "   Same memory? " << (msg.begin() == received->begin() ? "YES" : "NO") << "\n";
    } else {
        std::cerr << "Failed to receive message\n";
    }
    
    // Demonstrate capacity and resizing
    std::cout << "\n4. Dynamic sizing within pre-allocated capacity...\n";
    FloatVector large_msg(channel);
    
    std::cout << "   Message capacity: " << large_msg.capacity() << " floats\n";
    
    // Resize and fill
    large_msg.resize(100);
    for (size_t i = 0; i < 100; ++i) {
        large_msg[i] = static_cast<float>(i) * 0.1f;
    }
    
    std::cout << "   Filled " << large_msg.size() << " floats\n";
    channel.send(large_msg);
    
    // Receive and verify
    auto large_received = channel.receive_single<FloatVector>();
    if (large_received) {
        std::cout << "   Received " << large_received->size() << " floats\n";
        std::cout << "   First 5 values: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << (*large_received)[i] << " ";
        }
        std::cout << "...\n";
    }
    
    std::cout << "\nExample completed successfully!\n";
    return 0;
}