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
    // Create client channel
    const size_t buffer_size = 64 * 1024;  // 64KB buffer
    TCPChannel<RingBuffer<SingleProducer, SingleConsumer>> client_channel(
        "localhost", 9999, buffer_size, false);
    
    std::cout << "TCP Echo Client connected to localhost:9999" << std::endl;
    
    // Receiver thread
    std::thread receiver_thread([&client_channel]() {
        while (true) {
            auto* rb = client_channel.ring_buffer();
            if (rb) {
                auto read_handle = rb->read();
                if (read_handle) {
                    TextMessage received(read_handle->data, read_handle->size);
                    std::cout << "Client received: " << received.get_text() << std::endl;
                    read_handle.reset();  // This commits the read
                } else {
                    // No data available, sleep a bit
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }
    });
    
    // Send messages from stdin
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit") break;
        
        TextMessage msg(client_channel);
        msg.set_text(line);
        msg.send();
        
        std::cout << "Client sent: " << line << std::endl;
        
        // Give some time for echo
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Clean shutdown
    receiver_thread.detach();
    
    return 0;
}