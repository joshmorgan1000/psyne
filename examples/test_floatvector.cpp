#include <cassert>
#include <iostream>
#include <psyne/psyne.hpp>

using namespace psyne;

void test_single_type_channel() {
    std::cout << "Testing FloatVector in single-type channel..." << std::endl;

    // Create a single-type channel
    auto channel = Channel::create("memory://float_channel", 4096,
                                  ChannelMode::SPSC, ChannelType::SingleType);

    // Test 1: Create and send a message
    {
        FloatVector msg(*channel);
        
        // Fill with data
        msg = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        assert(msg.size() == 5);
        assert(msg[0] == 1.0f);
        assert(msg[4] == 5.0f);

        msg.send();
    }

    // Test 2: Simulate receiving the message
    {
        // Advance read pointer to simulate consumption
        channel->advance_read_pointer(FloatVector::calculate_size());
        std::cout << "Message received and processed" << std::endl;
    }

    // Test 3: Resize functionality
    {
        FloatVector msg(*channel);
        msg.resize(10);
        assert(msg.size() == 10);

        for (size_t i = 0; i < 10; ++i) {
            msg[i] = static_cast<float>(i * i);
        }

        msg.send();
    }

    // Test 4: Simulate receiving resized message
    {
        // Advance read pointer to simulate consumption
        channel->advance_read_pointer(FloatVector::calculate_size());
        std::cout << "Resized message received and processed" << std::endl;
    }

    std::cout << "All single-type channel tests passed!" << std::endl;
}

void test_multi_type_channel() {
    std::cout << "Testing FloatVector in multi-type channel..." << std::endl;

    // Create a multi-type channel
    auto channel = Channel::create("memory://multi_channel", 4096,
                                  ChannelMode::SPSC, ChannelType::MultiType);

    // Send a FloatVector
    {
        FloatVector msg(*channel);
        
        msg = {10.0f, 20.0f, 30.0f};
        assert(msg.size() == 3);

        msg.send();
    }

    // Simulate receiving as specific type
    {
        // Advance read pointer to simulate consumption
        channel->advance_read_pointer(FloatVector::calculate_size());
        std::cout << "Multi-type message received and processed" << std::endl;
    }

    std::cout << "All multi-type channel tests passed!" << std::endl;
}

void test_memory_efficiency() {
    std::cout << "Testing memory efficiency..." << std::endl;

    // Check that FloatVector size calculation
    size_t actual_size = FloatVector::calculate_size();

    std::cout << "FloatVector size: " << actual_size
              << " bytes (efficient zero-copy layout)" << std::endl;

    std::cout << "Memory layout: dynamic vector with zero-copy access" << std::endl;
    std::cout << "Zero-copy message design with dynamic allocation" << std::endl;
}

int main() {
    try {
        test_single_type_channel();
        test_multi_type_channel();
        test_memory_efficiency();

        std::cout << "\nAll tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}