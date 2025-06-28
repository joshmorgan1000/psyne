#include <iostream>
#include <cassert>
#include <psyne/psyne.hpp>

using namespace psyne;

void test_single_type_channel() {
    std::cout << "Testing FloatVector in single-type channel..." << std::endl;
    
    // Create a single-type channel
    auto channel = create_channel("memory://float_channel", 4096, 
                                 ChannelMode::SPSC, ChannelType::SingleType);
    
    // Test 1: Create and send a message
    {
        auto msg = channel->create_message<FloatVector>();
        assert(msg.is_valid());
        
        // Fill with data
        msg = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        assert(msg.size() == 5);
        assert(msg[0] == 1.0f);
        assert(msg[4] == 5.0f);
        
        channel->send(msg);
    }
    
    // Test 2: Receive the message
    {
        auto received = channel->receive_single<FloatVector>();
        assert(received.has_value());
        
        auto& msg = received.value();
        assert(msg.size() == 5);
        assert(msg[0] == 1.0f);
        assert(msg[4] == 5.0f);
        
        // Test Eigen view
        auto eigen_view = msg.as_eigen();
        assert(eigen_view.size() == 5);
        assert(eigen_view(0) == 1.0f);
    }
    
    // Test 3: Resize functionality
    {
        auto msg = channel->create_message<FloatVector>();
        msg.resize(10);
        assert(msg.size() == 10);
        
        for (size_t i = 0; i < 10; ++i) {
            msg[i] = static_cast<float>(i * i);
        }
        
        channel->send(msg);
    }
    
    // Test 4: Receive resized message
    {
        auto received = channel->receive_single<FloatVector>();
        assert(received.has_value());
        
        auto& msg = received.value();
        assert(msg.size() == 10);
        for (size_t i = 0; i < 10; ++i) {
            assert(msg[i] == static_cast<float>(i * i));
        }
    }
    
    std::cout << "All single-type channel tests passed!" << std::endl;
}

void test_multi_type_channel() {
    std::cout << "Testing FloatVector in multi-type channel..." << std::endl;
    
    // Create a multi-type channel
    auto channel = create_channel("memory://multi_channel", 4096, 
                                 ChannelMode::SPSC, ChannelType::MultiType);
    
    // Send a FloatVector
    {
        auto msg = channel->create_message<FloatVector>();
        assert(msg.is_valid());
        
        msg = {10.0f, 20.0f, 30.0f};
        assert(msg.size() == 3);
        
        channel->send(msg);
    }
    
    // Receive as specific type
    {
        auto received = channel->receive_as<FloatVector>();
        assert(received.has_value());
        
        auto& msg = received.value();
        assert(msg.size() == 3);
        assert(msg[0] == 10.0f);
        assert(msg[1] == 20.0f);
        assert(msg[2] == 30.0f);
    }
    
    std::cout << "All multi-type channel tests passed!" << std::endl;
}

void test_memory_efficiency() {
    std::cout << "Testing memory efficiency..." << std::endl;
    
    // Check that FloatVector size calculation doesn't include VariantHdr
    size_t expected_size = sizeof(float) * FloatVector::max_elements + sizeof(size_t);
    size_t actual_size = FloatVector::calculate_size();
    
    assert(actual_size == expected_size);
    std::cout << "FloatVector size: " << actual_size << " bytes (no VariantHdr overhead)" << std::endl;
    
    // Original size with VariantHdr would have been:
    size_t old_size = sizeof(VariantHdr) + sizeof(float) * 1024;
    std::cout << "Old size with VariantHdr: " << old_size << " bytes" << std::endl;
    std::cout << "Saved: " << (old_size - actual_size) << " bytes per message" << std::endl;
}

int main() {
    try {
        test_single_type_channel();
        test_multi_type_channel();
        test_memory_efficiency();
        
        std::cout << "\nAll tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}