#include <psyne/psyne.hpp>
#include <cassert>
#include <iostream>
#include <thread>
#include <cstring>

using namespace psyne;

// Test message type
class SimpleMessage : public Message<SimpleMessage> {
public:
    static constexpr uint32_t message_type = 400;
    static constexpr size_t size = 64;
    
    template<typename Channel>
    explicit SimpleMessage(Channel& channel) : Message<SimpleMessage>(channel) {
        if (this->data_) {
            std::memset(this->data_, 0, size);
        }
    }
    
    explicit SimpleMessage(const void* data, size_t sz) 
        : Message<SimpleMessage>(data, sz) {}
    
    static constexpr size_t calculate_size() { 
        return size; 
    }
    
    void set_value(uint64_t val) {
        if (data_) {
            *reinterpret_cast<uint64_t*>(data_) = val;
        }
    }
    
    uint64_t get_value() const {
        if (!data_) return 0;
        return *reinterpret_cast<const uint64_t*>(data_);
    }
    
    void before_send() {}
};

// Explicit template instantiation for SimpleMessage
template class psyne::Message<SimpleMessage>;

// Test basic memory channel
void test_memory_channel() {
    std::cout << "Testing memory channel..." << std::endl;
    
    SPSCChannel channel("memory://test", 64 * 1024, ChannelType::SingleType);
    
    // Test single message
    {
        SimpleMessage msg(channel);
        msg.set_value(42);
        msg.send();
    }
    
    auto received = channel.receive_single<SimpleMessage>();
    assert(received.has_value());
    assert(received->get_value() == 42);
    
    std::cout << "✓ Memory channel test passed!" << std::endl;
}

// Test ring buffer operations
void test_ring_buffer() {
    std::cout << "Testing ring buffer..." << std::endl;
    
    SPSCRingBuffer rb(1024);
    
    // Test write and read
    auto write_handle = rb.reserve(64);
    assert(write_handle.has_value());
    
    std::memset(write_handle->data, 0xAB, 64);
    write_handle->commit();
    
    auto read_handle = rb.read();
    assert(read_handle.has_value());
    assert(read_handle->size == 64);
    assert(*static_cast<const uint8_t*>(read_handle->data) == 0xAB);
    (void)read_handle; // Suppress unused variable warning
    
    std::cout << "✓ Ring buffer test passed!" << std::endl;
}

// Test message creation
void test_message_creation() {
    std::cout << "Testing message creation..." << std::endl;
    
    SPSCChannel channel("memory://test", 64 * 1024);
    
    // Test FloatVector
    {
        FloatVector fv(channel);
        fv.resize(10);
        for (size_t i = 0; i < 10; ++i) {
            fv[i] = static_cast<float>(i) * 1.5f;
        }
        
        assert(fv.size() == 10);
        assert(fv[5] == 7.5f);
        
        // Test Eigen view
        auto eigen_view = fv.as_eigen();
        assert(eigen_view.size() == 10);
        assert(eigen_view(5) == 7.5f);
        (void)eigen_view; // Suppress unused variable warning
    }
    
    std::cout << "✓ Message creation test passed!" << std::endl;
}

// Test channel factory
void test_channel_factory() {
    std::cout << "Testing channel factory..." << std::endl;
    
    // Test URI parsing
    assert(ChannelFactory::is_memory_uri("memory://test"));
    assert(ChannelFactory::is_ipc_uri("ipc://test"));
    assert(ChannelFactory::is_tcp_uri("tcp://localhost:8080"));
    
    // Test memory channel creation
    auto channel = ChannelFactory::create<SPSCRingBuffer>(
        "memory://factory_test", 
        64 * 1024,
        ChannelType::SingleType
    );
    
    assert(channel != nullptr);
    assert(channel->uri() == "memory://factory_test");
    
    std::cout << "✓ Channel factory test passed!" << std::endl;
}

// Test producer-consumer pattern
void test_producer_consumer() {
    std::cout << "Testing producer-consumer..." << std::endl;
    
    SPSCChannel channel("memory://pc_test", 64 * 1024);
    const int num_messages = 100;
    std::atomic<int> received_count{0};
    
    std::thread producer([&]() {
        for (int i = 0; i < num_messages; ++i) {
            SimpleMessage msg(channel);
            msg.set_value(i);
            // Debug: verify value was set
            assert(msg.get_value() == static_cast<uint64_t>(i));
            msg.send();
            
            if (i % 10 == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    });
    
    std::thread consumer([&]() {
        int expected = 0;
        while (expected < num_messages) {
            auto msg = channel.receive_single<SimpleMessage>(
                std::chrono::milliseconds(10)
            );
            if (msg) {
                uint64_t value = msg->get_value();
                if (value != static_cast<uint64_t>(expected)) {
                    std::cerr << "Expected: " << expected << ", Got: " << value << std::endl;
                    assert(false);
                }
                expected++;
                received_count++;
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    assert(received_count == num_messages);
    std::cout << "✓ Producer-consumer test passed!" << std::endl;
}

int main() {
    std::cout << "Running basic Psyne tests..." << std::endl;
    std::cout << "===========================" << std::endl;
    
    try {
        test_ring_buffer();
        test_memory_channel();
        test_message_creation();
        test_channel_factory();
        test_producer_consumer();
        
        std::cout << "\nAll tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}