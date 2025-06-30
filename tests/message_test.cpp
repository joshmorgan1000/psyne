#include <cassert>
#include <iostream>
#include <psyne/psyne.hpp>
#include <string>
#include <vector>

// Test message types and operations
int main() {
    std::cout << "Running Message Tests..." << std::endl;

    try {
        // Test 1: Memory channel creation
        auto channel =
            psyne::create_channel("memory://test_messages", 1024 * 1024);
        assert(channel != nullptr);
        std::cout << "✓ Channel creation successful" << std::endl;

        // Test 2: FloatVector message
        {
            auto msg = psyne::FloatVector(*channel);
            msg.resize(5);
            for (size_t i = 0; i < 5; ++i) {
                msg[i] = static_cast<float>(i * 1.5f);
            }

            assert(msg.size() == 5);
            assert(msg[0] == 0.0f);
            assert(msg[4] == 6.0f);
            std::cout << "✓ FloatVector message operations" << std::endl;
        }

        // Test 3: ByteVector message
        {
            auto msg = psyne::ByteVector(*channel);
            msg.resize(10);

            for (size_t i = 0; i < 10; ++i) {
                msg[i] = static_cast<uint8_t>(i + 100);
            }

            assert(msg.size() == 10);
            assert(msg[0] == 100);
            assert(msg[9] == 109);
            std::cout << "✓ ByteVector message operations" << std::endl;
        }

        // Test 4: DoubleMatrix message
        {
            auto msg = psyne::DoubleMatrix(*channel);
            msg.set_dimensions(3, 4);

            assert(msg.rows() == 3);
            assert(msg.cols() == 4);

            // Set some values
            msg.at(0, 0) = 1.5;
            msg.at(2, 3) = 9.7;

            assert(msg.at(0, 0) == 1.5);
            assert(msg.at(2, 3) == 9.7);
            std::cout << "✓ DoubleMatrix message operations" << std::endl;
        }

        // Test 5: Message type IDs
        assert(psyne::FloatVector::message_type == 1);
        assert(psyne::DoubleMatrix::message_type == 2);
        assert(psyne::ByteVector::message_type == 10);
        std::cout << "✓ Message type IDs correct" << std::endl;

        std::cout << "All Message Tests Passed! ✅" << std::endl;
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}