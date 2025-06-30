/**
 * @file request_reply_demo.cpp
 * @brief Demonstrates request/reply messaging pattern with Psyne
 * 
 * Request/reply is a fundamental messaging pattern where a client sends
 * a request and waits for a response. This example shows how to implement
 * it efficiently with Psyne's zero-copy architecture.
 */

#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <future>

using namespace psyne;

// Request message type
class Request : public Message<Request> {
public:
    static consteval size_t calculate_size() noexcept {
        return sizeof(uint32_t) + 256; // ID + payload
    }
    
    Request(Channel& channel) : Message<Request>(channel) {}
    
    void set_request_id(uint32_t id) {
        *reinterpret_cast<uint32_t*>(data()) = id;
    }
    
    uint32_t get_request_id() const {
        return *reinterpret_cast<const uint32_t*>(data());
    }
    
    std::span<char> payload() {
        return std::span<char>(reinterpret_cast<char*>(data() + sizeof(uint32_t)), 256);
    }
};

// Reply message type
class Reply : public Message<Reply> {
public:
    static consteval size_t calculate_size() noexcept {
        return sizeof(uint32_t) + sizeof(uint32_t) + 512; // ID + status + payload
    }
    
    Reply(Channel& channel) : Message<Reply>(channel) {}
    
    void set_request_id(uint32_t id) {
        *reinterpret_cast<uint32_t*>(data()) = id;
    }
    
    uint32_t get_request_id() const {
        return *reinterpret_cast<const uint32_t*>(data());
    }
    
    void set_status(uint32_t status) {
        *reinterpret_cast<uint32_t*>(data() + sizeof(uint32_t)) = status;
    }
    
    uint32_t get_status() const {
        return *reinterpret_cast<const uint32_t*>(data() + sizeof(uint32_t));
    }
    
    std::span<char> payload() {
        return std::span<char>(reinterpret_cast<char*>(data() + 2 * sizeof(uint32_t)), 512);
    }
};

// Simple request/reply server
void run_server(Channel& request_channel, Channel& reply_channel) {
    std::cout << "🖥️  Server: Listening for requests...\n";
    
    while (true) {
        // Wait for request
        auto req_span = request_channel.buffer_span();
        if (req_span.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Process request (zero-copy view)
        Request req(request_channel);
        req.set_data(req_span.data(), req_span.size());
        
        uint32_t req_id = req.get_request_id();
        std::cout << "📨 Server: Got request #" << req_id << "\n";
        
        // Create reply
        Reply reply(reply_channel);
        reply.set_request_id(req_id);
        reply.set_status(200); // Success
        
        // Process and fill reply
        auto req_payload = req.payload();
        auto reply_payload = reply.payload();
        std::string response = "Processed: " + std::string(req_payload.data());
        std::copy_n(response.begin(), std::min(response.size(), reply_payload.size()), 
                    reply_payload.begin());
        
        // Send reply
        reply.send();
        std::cout << "📤 Server: Sent reply for request #" << req_id << "\n";
        
        // Advance read pointer
        request_channel.advance_read_pointer(Request::calculate_size());
    }
}

// Client that sends requests and waits for replies
void run_client(Channel& request_channel, Channel& reply_channel) {
    std::cout << "💻 Client: Sending requests...\n";
    
    for (uint32_t i = 1; i <= 5; ++i) {
        // Create request
        Request req(request_channel);
        req.set_request_id(i);
        
        // Fill request payload
        auto payload = req.payload();
        std::string msg = "Hello from request " + std::to_string(i);
        std::copy_n(msg.begin(), std::min(msg.size(), payload.size()), payload.begin());
        
        std::cout << "📮 Client: Sending request #" << i << "\n";
        req.send();
        
        // Wait for reply
        while (true) {
            auto reply_span = reply_channel.buffer_span();
            if (!reply_span.empty()) {
                Reply reply(reply_channel);
                reply.set_data(reply_span.data(), reply_span.size());
                
                if (reply.get_request_id() == i) {
                    std::cout << "✅ Client: Got reply for request #" << i 
                              << " (status: " << reply.get_status() << ")\n";
                    reply_channel.advance_read_pointer(Reply::calculate_size());
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "🎉 Client: All requests completed!\n";
}

int main() {
    std::cout << "🔄 Request/Reply Pattern Demo\n";
    std::cout << "=============================\n\n";
    
    // Create channels for request/reply pattern
    // In real use, these might be TCP channels for network communication
    auto request_channel = Channel::create("memory://requests", 
                                         1024 * 1024,
                                         ChannelMode::SPSC,
                                         ChannelType::SingleType);
    
    auto reply_channel = Channel::create("memory://replies",
                                       1024 * 1024,
                                       ChannelMode::SPSC,
                                       ChannelType::SingleType);
    
    std::cout << "📡 Created request and reply channels\n\n";
    
    // Start server in background
    std::thread server_thread(run_server, 
                             std::ref(*request_channel), 
                             std::ref(*reply_channel));
    
    // Give server time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Run client
    run_client(*request_channel, *reply_channel);
    
    // In real app, would have proper shutdown mechanism
    server_thread.detach();
    
    std::cout << "\n✨ Request/Reply pattern demonstrated successfully!\n";
    std::cout << "\n📝 Key Points:\n";
    std::cout << "   - Requests and replies are separate message types\n";
    std::cout << "   - Each has a unique ID for correlation\n";
    std::cout << "   - Zero-copy throughout the entire flow\n";
    std::cout << "   - Can work over any transport (memory, TCP, etc.)\n";
    
    return 0;
}