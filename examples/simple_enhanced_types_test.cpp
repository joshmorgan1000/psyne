/**
 * @file simple_enhanced_types_test.cpp
 * @brief Simple test for basic message types in Psyne v1.3.0
 * 
 * Tests the core message types available in the current implementation.
 */

#include <iostream>
#include <psyne/psyne.hpp>

using namespace psyne;

// Simple test message
class TestMessage : public Message<TestMessage> {
public:
    static constexpr uint32_t message_type = 999;
    using Message<TestMessage>::Message;

    static size_t calculate_size() {
        return 64; // 64 bytes
    }

    void set_value(int value) {
        *reinterpret_cast<int*>(data()) = value;
    }

    int get_value() const {
        return *reinterpret_cast<const int*>(data());
    }
};

int main() {
    try {
        std::cout << "Testing Basic Message Types - v1.3.0" << std::endl;
        std::cout << "=====================================" << std::endl;

        // Create a channel
        auto channel = create_channel("memory://enhanced_test", 1024 * 1024);

        std::cout << "\nTesting basic Message<T> functionality..." << std::endl;
        
        // Test creating and sending a message
        TestMessage msg(*channel);
        msg.set_value(42);
        std::cout << "Created message with value: " << msg.get_value() << std::endl;
        
        msg.send();
        std::cout << "Message sent successfully" << std::endl;

        // Test receiving the message
        size_t msg_size;
        uint32_t msg_type;
        void* msg_data = channel->receive_raw_message(msg_size, msg_type);
        
        if (msg_data && msg_type == TestMessage::message_type) {
            TestMessage received(msg_data, msg_size);
            std::cout << "Received message with value: " << received.get_value() << std::endl;
            channel->release_raw_message(msg_data);
        } else {
            std::cout << "No message received" << std::endl;
        }

        // Test FloatVector (available from types.hpp)
        std::cout << "\nTesting FloatVector..." << std::endl;
        FloatVector fvec(*channel);
        fvec.resize(5);
        for (size_t i = 0; i < 5; ++i) {
            fvec[i] = static_cast<float>(i * 2.5f);
        }
        
        std::cout << "FloatVector size: " << fvec.size() << std::endl;
        std::cout << "Values: ";
        for (size_t i = 0; i < fvec.size(); ++i) {
            std::cout << fvec[i] << " ";
        }
        std::cout << std::endl;

        // Test DoubleMatrix (available from types.hpp) 
        std::cout << "\nTesting DoubleMatrix..." << std::endl;
        DoubleMatrix dmat(*channel);
        dmat.set_dimensions(2, 3);
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                dmat.at(i, j) = static_cast<double>(i * 3 + j + 1);
            }
        }
        
        std::cout << "DoubleMatrix " << dmat.rows() << "x" << dmat.cols() << std::endl;
        std::cout << "Matrix values:" << std::endl;
        for (size_t i = 0; i < dmat.rows(); ++i) {
            for (size_t j = 0; j < dmat.cols(); ++j) {
                std::cout << dmat.at(i, j) << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\n✅ Basic message types working!" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}