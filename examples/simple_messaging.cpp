#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>

using namespace psyne;

int main() {
    std::cout << "Psyne Simple Messaging Example\n";
    std::cout << "==============================\n\n";

    // Create a single-type channel optimized for FloatVector
    auto channel = create_channel("memory://simple", 128 * 1024 * 1024,
                                  ChannelMode::SPSC, ChannelType::SingleType);

    std::cout << "Channel created with 128MB buffer\n";

    // Demonstrate zero-copy write
    std::cout
        << "\n1. Creating message directly in buffer (zero allocation)...\n";
    std::cout << "   Required size for FloatVector: "
              << FloatVector::calculate_size() << " bytes\n";

    FloatVector msg(*channel);

    if (!msg.is_valid()) {
        std::cerr << "Failed to allocate message\n";
        return 1;
    }

    std::cout << "   Message is valid, size: " << msg.size()
              << ", capacity: " << msg.capacity() << "\n";

    // Write data directly into the buffer
    msg = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    std::cout << "   Wrote " << msg.size() << " floats directly to buffer\n";
    std::cout << "   Buffer address: " << static_cast<void *>(msg.begin())
              << "\n";

    // Send the message (zero-copy notification)
    msg.send();
    std::cout << "   Message sent (zero-copy notification to channel)\n";

    // Receive the message (zero-copy)
    std::cout << "\n2. Receiving message (zero-copy view)...\n";
    auto received = channel->receive<FloatVector>();

    if (received) {
        std::cout << "   Received " << received->size() << " floats\n";
        std::cout << "   Buffer address: "
                  << static_cast<const void *>(received->begin()) << "\n";
        std::cout << "   Values: ";
        for (float val : *received) {
            std::cout << val << " ";
        }
        std::cout << "\n";

        // Verify zero-copy nature
        std::cout << "\n3. Understanding zero-copy...\n";
        std::cout << "   The received message is a view into the ring buffer\n";
        std::cout
            << "   No data was copied during send or receive operations\n";
    } else {
        std::cerr << "Failed to receive message\n";
    }

    // Demonstrate capacity and resizing
    std::cout << "\n4. Dynamic sizing within pre-allocated capacity...\n";
    FloatVector large_msg(*channel);

    std::cout << "   Message capacity: " << large_msg.capacity() << " floats\n";

    // Resize and fill
    large_msg.resize(100);
    for (size_t i = 0; i < 100; ++i) {
        large_msg[i] = static_cast<float>(i) * 0.1f;
    }

    std::cout << "   Filled " << large_msg.size() << " floats\n";
    large_msg.send();

    // Receive and verify
    auto large_received = channel->receive<FloatVector>();
    if (large_received) {
        size_t recv_size = large_received->size();
        std::cout << "   Received " << recv_size << " floats\n";
        if (recv_size == 100) {
            std::cout << "   First 5 values: ";
            for (size_t i = 0; i < 5; ++i) {
                std::cout << (*large_received)[i] << " ";
            }
            std::cout << "...\n";
        } else {
            std::cout << "   ERROR: Expected 100 floats, got " << recv_size
                      << "\n";
        }
    }

    std::cout << "\nExample completed successfully!\n";
    return 0;
}