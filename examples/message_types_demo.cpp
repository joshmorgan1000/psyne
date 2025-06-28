#include <psyne/psyne.hpp>
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace psyne;

void test_float_vector() {
    std::cout << "=== FloatVector Tests ===\n";
    
    auto channel = create_channel("memory://float_test", 1024 * 1024);
    
    // Test basic operations
    FloatVector msg(*channel);
    msg.resize(10);
    
    // Fill with test data
    for (size_t i = 0; i < 10; ++i) {
        msg[i] = static_cast<float>(i) * 1.5f;
    }
    
    std::cout << "Created FloatVector with " << msg.size() << " elements:\n";
    std::cout << "Values: ";
    for (size_t i = 0; i < msg.size(); ++i) {
        std::cout << msg[i] << " ";
    }
    std::cout << "\n";
    
    // Test Eigen integration
    auto eigen_view = msg.as_eigen();
    std::cout << "Eigen view sum: " << eigen_view.sum() << "\n";
    std::cout << "Eigen view mean: " << eigen_view.mean() << "\n";
    
    // Test send/receive
    msg.send();
    
    auto received = channel->receive_single<FloatVector>();
    assert(received.has_value());
    assert(received->size() == 10);
    
    std::cout << "Successfully sent and received FloatVector\n";
    
    // Test initializer list
    FloatVector msg2(*channel);
    msg2 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::cout << "Initializer list test: ";
    for (float val : msg2) {
        std::cout << val << " ";
    }
    std::cout << "\n\n";
}

void test_byte_vector() {
    std::cout << "=== ByteVector Tests ===\n";
    
    auto channel = create_channel("memory://byte_test", 1024 * 1024);
    
    ByteVector msg(*channel);
    msg.resize(16);
    
    // Fill with binary data
    for (size_t i = 0; i < msg.size(); ++i) {
        msg[i] = static_cast<uint8_t>(i * 16 + 7); // Some pattern
    }
    
    std::cout << "Created ByteVector with " << msg.size() << " bytes:\n";
    std::cout << "Values (hex): ";
    for (size_t i = 0; i < msg.size(); ++i) {
        printf("%02x ", msg[i]);
    }
    std::cout << "\n";
    
    // Test iterators
    std::cout << "Using iterators: ";
    for (auto it = msg.begin(); it != msg.end(); ++it) {
        printf("%02x ", *it);
    }
    std::cout << "\n";
    
    // Test send/receive
    msg.send();
    
    auto received = channel->receive_single<ByteVector>();
    assert(received.has_value());
    assert(received->size() == 16);
    
    std::cout << "Successfully sent and received ByteVector\n\n";
}

void test_double_matrix() {
    std::cout << "=== DoubleMatrix Tests ===\n";
    
    auto channel = create_channel("memory://matrix_test", 1024 * 1024);
    
    DoubleMatrix msg(*channel);
    msg.set_dimensions(3, 4); // 3 rows, 4 columns
    
    // Fill with test data
    for (size_t i = 0; i < msg.rows(); ++i) {
        for (size_t j = 0; j < msg.cols(); ++j) {
            msg.at(i, j) = static_cast<double>(i * 10 + j) + 0.5;
        }
    }
    
    std::cout << "Created " << msg.rows() << "x" << msg.cols() << " DoubleMatrix:\n";
    for (size_t i = 0; i < msg.rows(); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < msg.cols(); ++j) {
            printf("%6.1f", msg.at(i, j));
            if (j < msg.cols() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    
    // Test some matrix operations
    double sum = 0.0;
    for (size_t i = 0; i < msg.rows(); ++i) {
        for (size_t j = 0; j < msg.cols(); ++j) {
            sum += msg.at(i, j);
        }
    }
    std::cout << "Matrix sum: " << sum << "\n";
    
    // Test send/receive
    msg.send();
    
    auto received = channel->receive_single<DoubleMatrix>();
    assert(received.has_value());
    assert(received->rows() == 3);
    assert(received->cols() == 4);
    
    std::cout << "Successfully sent and received DoubleMatrix\n\n";
}

void test_multi_type_channel() {
    std::cout << "=== Multi-Type Channel Test ===\n";
    
    auto channel = create_channel("memory://multi_test", 1024 * 1024,
                                  ChannelMode::SPSC, ChannelType::MultiType);
    
    // Send different types of messages
    {
        FloatVector float_msg(*channel);
        float_msg.resize(3);
        float_msg[0] = 1.0f;
        float_msg[1] = 2.0f;
        float_msg[2] = 3.0f;
        float_msg.send();
        std::cout << "Sent FloatVector\n";
    }
    
    {
        ByteVector byte_msg(*channel);
        byte_msg.resize(4);
        for (size_t i = 0; i < 4; ++i) {
            byte_msg[i] = static_cast<uint8_t>(i + 100);
        }
        byte_msg.send();
        std::cout << "Sent ByteVector\n";
    }
    
    {
        DoubleMatrix matrix_msg(*channel);
        matrix_msg.set_dimensions(2, 2);
        matrix_msg.at(0, 0) = 10.5;
        matrix_msg.at(0, 1) = 20.5;
        matrix_msg.at(1, 0) = 30.5;
        matrix_msg.at(1, 1) = 40.5;
        matrix_msg.send();
        std::cout << "Sent DoubleMatrix\n";
    }
    
    // Receive them back (order should be preserved)
    {
        auto msg = channel->receive_single<FloatVector>();
        assert(msg.has_value());
        std::cout << "Received FloatVector: ";
        for (float val : *msg) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    
    {
        auto msg = channel->receive_single<ByteVector>();
        assert(msg.has_value());
        std::cout << "Received ByteVector: ";
        for (uint8_t val : *msg) {
            std::cout << static_cast<int>(val) << " ";
        }
        std::cout << "\n";
    }
    
    {
        auto msg = channel->receive_single<DoubleMatrix>();
        assert(msg.has_value());
        std::cout << "Received DoubleMatrix:\n";
        for (size_t i = 0; i < msg->rows(); ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < msg->cols(); ++j) {
                std::cout << msg->at(i, j);
                if (j < msg->cols() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }
    
    std::cout << "Multi-type channel test completed successfully!\n\n";
}

void test_performance() {
    std::cout << "=== Performance Test ===\n";
    
    auto channel = create_channel("memory://perf_test", 8 * 1024 * 1024);
    
    const int num_messages = 10000;
    const size_t vector_size = 1000;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Send messages
    for (int i = 0; i < num_messages; ++i) {
        FloatVector msg(*channel);
        msg.resize(vector_size);
        
        // Fill with computed values (to prevent optimization)
        for (size_t j = 0; j < vector_size; ++j) {
            msg[j] = std::sin(static_cast<float>(i + j)) * 100.0f;
        }
        
        msg.send();
        
        // Receive immediately (single-threaded test)
        auto received = channel->receive_single<FloatVector>();
        assert(received.has_value());
        assert(received->size() == vector_size);
        
        // Verify first element
        float expected = std::sin(static_cast<float>(i)) * 100.0f;
        assert(std::abs((*received)[0] - expected) < 0.001f);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    double messages_per_second = static_cast<double>(num_messages) / 
                               (static_cast<double>(duration.count()) / 1e6);
    double mb_per_second = (static_cast<double>(num_messages) * vector_size * sizeof(float)) / 
                          (1024.0 * 1024.0 * static_cast<double>(duration.count()) / 1e6);
    
    std::cout << "Performance Results:\n";
    std::cout << "  Messages: " << num_messages << "\n";
    std::cout << "  Vector size: " << vector_size << " floats (" << (vector_size * sizeof(float)) << " bytes)\n";
    std::cout << "  Total time: " << (duration.count() / 1000.0) << " ms\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << messages_per_second << " msg/s\n";
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(1) << mb_per_second << " MB/s\n";
    std::cout << "  Avg latency: " << std::fixed << std::setprecision(2) 
              << (static_cast<double>(duration.count()) / num_messages / 2.0) << " μs per message\n\n";
}

int main() {
    std::cout << "Psyne Message Types Demo\n";
    std::cout << "========================\n\n";
    
    try {
        test_float_vector();
        test_byte_vector();
        test_double_matrix();
        test_multi_type_channel();
        test_performance();
        
        std::cout << "All message type tests completed successfully! ✅\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}