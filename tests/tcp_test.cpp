#include <atomic>
#include <cassert>
#include <cstring>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

using namespace psyne;

// Test message type
class TestMessage : public Message<TestMessage> {
public:
    static constexpr uint32_t message_type = 300;
    static constexpr size_t payload_size = 1024;

    template <typename Channel>
    explicit TestMessage(Channel &channel) : Message<TestMessage>(channel) {
        if (this->data_) {
            std::memset(this->data_, 0, payload_size);
        }
    }

    explicit TestMessage(const void *data, size_t size)
        : Message<TestMessage>(data, size) {}

    static constexpr size_t calculate_size() {
        return payload_size;
    }

    void set_data(const std::string &text, uint64_t counter) {
        if (!data_)
            return;

        // Store counter at beginning
        *reinterpret_cast<uint64_t *>(data_) = counter;

        // Store text after counter
        size_t text_len =
            std::min(text.size(), payload_size - sizeof(uint64_t) - 1);
        std::memcpy(data_ + sizeof(uint64_t), text.c_str(), text_len);
        data_[sizeof(uint64_t) + text_len] = '\0';
    }

    uint64_t get_counter() const {
        if (!data_)
            return 0;
        return *reinterpret_cast<const uint64_t *>(data_);
    }

    std::string get_text() const {
        if (!data_)
            return "";
        return std::string(
            reinterpret_cast<const char *>(data_ + sizeof(uint64_t)));
    }

    void before_send() {}
};

// Test basic TCP functionality
void test_tcp_basic() {
    std::cout << "Testing basic TCP functionality..." << std::endl;

    const size_t buffer_size = 64 * 1024;
    std::atomic<bool> server_ready{false};
    std::atomic<uint64_t> messages_received{0};
    const uint64_t num_messages = 10;

    // Server thread
    std::thread server_thread([&]() {
        try {
            TCPChannel<SPSCRingBuffer> server("0.0.0.0", 9997, buffer_size,
                                              true);
            server_ready = true;

            while (messages_received < num_messages) {
                auto *rb = server.ring_buffer();
                if (rb) {
                    auto read_handle = rb->read();
                    if (read_handle) {
                        TestMessage msg(read_handle->data, read_handle->size);

                        uint64_t counter = msg.get_counter();
                        std::string text = msg.get_text();

                        // Verify message
                        assert(text ==
                               "Test message " + std::to_string(counter));

                        messages_received++;
                        read_handle.reset();
                    } else {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(10));
                    }
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "Server error: " << e.what() << std::endl;
            assert(false);
        }
    });

    // Wait for server to start
    while (!server_ready) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Client thread
    std::thread client_thread([&]() {
        try {
            // Give server time to fully initialize
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            TCPChannel<SPSCRingBuffer> client("localhost", 9997, buffer_size,
                                              false);

            // Send messages
            for (uint64_t i = 0; i < num_messages; ++i) {
                TestMessage msg(client);
                msg.set_data("Test message " + std::to_string(i), i);
                msg.send();

                // Small delay between messages
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            // Give time for all messages to be received
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

        } catch (const std::exception &e) {
            std::cerr << "Client error: " << e.what() << std::endl;
            assert(false);
        }
    });

    client_thread.join();
    server_thread.join();

    assert(messages_received == num_messages);
    std::cout << "✓ Basic TCP test passed!" << std::endl;
}

// Test TCP framing and checksums
void test_tcp_framing() {
    std::cout << "Testing TCP framing and checksums..." << std::endl;

    // Test frame header creation
    const char *test_data = "Hello, TCP framing!";
    size_t test_size = strlen(test_data);

    auto header = TCPFramer::create_header(test_data, test_size);

    // Verify checksum
    assert(TCPFramer::verify_frame(header, test_data, test_size));

    // Test with corrupted data
    char corrupted_data[32];
    strcpy(corrupted_data, test_data);
    corrupted_data[5] = 'X'; // Corrupt one byte

    assert(!TCPFramer::verify_frame(header, corrupted_data, test_size));

    std::cout << "✓ TCP framing test passed!" << std::endl;
}

// Test channel factory TCP creation
void test_channel_factory() {
    std::cout << "Testing channel factory TCP creation..." << std::endl;

    // Test TCP URI parsing
    auto [host1, port1] =
        ChannelFactory::extract_tcp_endpoint("tcp://localhost:8080");
    assert(host1 == "localhost");
    assert(port1 == 8080);

    auto [host2, port2] =
        ChannelFactory::extract_tcp_endpoint("tcp://192.168.1.1:9999");
    assert(host2 == "192.168.1.1");
    assert(port2 == 9999);

    auto [host3, port3] =
        ChannelFactory::extract_tcp_endpoint("tcp://example.com");
    assert(host3 == "example.com");
    assert(port3 == 9999); // Default port

    // Test URI recognition
    assert(ChannelFactory::is_tcp_uri("tcp://localhost:8080"));
    assert(!ChannelFactory::is_tcp_uri("ipc://test"));
    assert(!ChannelFactory::is_tcp_uri("memory://test"));

    std::cout << "✓ Channel factory test passed!" << std::endl;
}

int main() {
    std::cout << "Running TCP messaging tests..." << std::endl;
    std::cout << "==============================" << std::endl;

    try {
        test_tcp_framing();
        test_channel_factory();
        test_tcp_basic();

        std::cout << "\nAll tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}