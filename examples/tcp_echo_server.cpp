#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>

using namespace psyne;

// Simple text message
class TextMessage : public Message<TextMessage> {
public:
    static constexpr uint32_t message_type = 100;  // Custom message type
    static constexpr size_t max_text_size = 256;   // Max text size
    
    template<typename Channel>
    explicit TextMessage(Channel& channel) : Message<TextMessage>(channel) {
        if (this->data_) {
            // Initialize the buffer with zeros
            std::memset(this->data_, 0, max_text_size);
        }
    }
    
    // Constructor for incoming messages
    explicit TextMessage(const void* data, size_t size) 
        : Message<TextMessage>(data, size) {}
    
    static constexpr size_t calculate_size() { 
        return max_text_size; 
    }
    
    void set_text(const std::string& text) {
        if (data_ && text.size() < max_text_size - 1) {
            std::memcpy(data_, text.c_str(), text.size() + 1);
        }
    }
    
    std::string get_text() const {
        if (!data_) return "";
        return std::string(reinterpret_cast<const char*>(data_));
    }
    
    void before_send() {
        // Optional: could add any last-minute processing here
    }
};

int main() {
    // Create server channel
    const size_t buffer_size = 64 * 1024;  // 64KB buffer
    TCPChannel<RingBuffer<SingleProducer, SingleConsumer>> server_channel(
        "0.0.0.0", 9999, buffer_size, true);
    
    std::cout << "TCP Echo Server listening on port 9999..." << std::endl;
    
    // Server thread - echo received messages back
    std::thread server_thread([&server_channel]() {
        while (true) {
            // Wait for incoming messages
            auto* rb = server_channel.ring_buffer();
            if (rb) {
                auto read_handle = rb->read();
                if (read_handle) {
                    // Create a view of the received message
                    TextMessage received(read_handle->data, read_handle->size);
                    
                    std::string text = received.get_text();
                    std::cout << "Server received: " << text << std::endl;
                    
                    // Commit the read
                    read_handle.reset();  // This commits the read
                    
                    // Send echo response
                    TextMessage echo(server_channel);
                    echo.set_text("Echo: " + text);
                    echo.send();
                } else {
                    // No data available, sleep a bit
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }
    });
    
    // Keep server running
    server_thread.join();
    
    return 0;
}