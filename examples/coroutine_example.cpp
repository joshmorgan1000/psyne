#include <boost/asio.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <cstring>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

using namespace psyne;
namespace asio = boost::asio;

// Simple text message
class TextMessage : public Message<TextMessage> {
public:
    static constexpr uint32_t message_type = 100;
    static constexpr size_t max_text_size = 256;

    using Message<TextMessage>::Message;

    static constexpr size_t calculate_size() {
        return max_text_size;
    }

    void set_text(const std::string &text) {
        if (data_ && text.size() < max_text_size - 1) {
            std::memset(data_, 0, max_text_size);
            std::memcpy(data_, text.c_str(), text.size() + 1);
        }
    }

    std::string get_text() const {
        if (!data_)
            return "";
        return std::string(reinterpret_cast<const char *>(data_));
    }
};

// Coroutine that processes messages
asio::awaitable<void> message_processor(Channel* channel, asio::io_context& io) {
    std::cout << "[Processor] Starting message processor coroutine..." << std::endl;

    while (true) {
        // Poll for messages with async timer
        asio::steady_timer timer(io);
        timer.expires_after(std::chrono::milliseconds(10));
        co_await timer.async_wait(asio::use_awaitable);

        // Try to receive message
        size_t msg_size;
        uint32_t msg_type;
        void* msg_data = channel->receive_raw_message(msg_size, msg_type);
        
        if (msg_data && msg_type == TextMessage::message_type) {
            TextMessage msg(msg_data, msg_size);
            std::string text = msg.get_text();
            std::cout << "[Processor] Coroutine received: " << text << std::endl;

            // Simulate some async processing
            timer.expires_after(std::chrono::milliseconds(50));
            co_await timer.async_wait(asio::use_awaitable);

            std::cout << "[Processor] Processed: " << text << std::endl;
            
            // Release the message
            channel->release_raw_message(msg_data);

            if (text == "quit") {
                break;
            }
        }
    }

    std::cout << "[Processor] Message processor coroutine finished." << std::endl;
}

// Coroutine that sends messages  
asio::awaitable<void> message_sender(Channel* channel, asio::io_context& io) {
    std::cout << "[Sender] Starting message sender coroutine..." << std::endl;
    
    std::vector<std::string> messages = {
        "Hello from coroutine!", 
        "This is an async message",
        "C++20 coroutines with zero-copy messaging", 
        "Boost.Asio integration works great",
        "quit"
    };

    for (const auto &text : messages) {
        // Wait a bit between messages
        asio::steady_timer timer(io);
        timer.expires_after(std::chrono::milliseconds(200));
        co_await timer.async_wait(asio::use_awaitable);

        bool sent = false;
        try {
            // Create and send message
            TextMessage msg(*channel);
            msg.set_text(text);
            msg.send();
            std::cout << "[Sender] Sent: " << text << std::endl;
            sent = true;
        } catch (const std::runtime_error& e) {
            std::cout << "[Sender] Buffer full, retrying..." << std::endl;
            // In production, implement proper backpressure handling
        }
        
        if (!sent) {
            timer.expires_after(std::chrono::milliseconds(50));
            co_await timer.async_wait(asio::use_awaitable);
        }
    }

    std::cout << "[Sender] Message sender coroutine finished." << std::endl;
}

// Coroutine that monitors channel metrics
asio::awaitable<void> metrics_monitor(Channel* channel, asio::io_context& io) {
    std::cout << "[Monitor] Starting metrics monitor..." << std::endl;
    
    for (int i = 0; i < 10; ++i) {
        asio::steady_timer timer(io);
        timer.expires_after(std::chrono::seconds(1));
        co_await timer.async_wait(asio::use_awaitable);
        
        if (channel->has_metrics()) {
            auto metrics = channel->get_metrics();
            std::cout << "[Monitor] Messages sent: " << metrics.messages_sent 
                      << ", received: " << metrics.messages_received << std::endl;
        }
    }
    
    std::cout << "[Monitor] Metrics monitor finished." << std::endl;
}

int main() {
    std::cout << "Psyne Coroutine Example\n";
    std::cout << "=======================\n\n";
    std::cout << "Demonstrating C++20 coroutines with zero-copy messaging\n\n";

    try {
        // Create channel with metrics enabled
        auto channel = create_channel("memory://coroutine_demo", 
                                      10 * 1024 * 1024,
                                      ChannelMode::SPSC, 
                                      ChannelType::SingleType,
                                      true); // Enable metrics

        // Create io_context for coroutines
        asio::io_context io_context;

        // Spawn coroutines
        asio::co_spawn(io_context, 
                       message_processor(channel.get(), io_context), 
                       asio::detached);
        
        asio::co_spawn(io_context, 
                       message_sender(channel.get(), io_context), 
                       asio::detached);
        
        asio::co_spawn(io_context, 
                       metrics_monitor(channel.get(), io_context), 
                       asio::detached);

        // Run the io_context
        std::cout << "Running coroutines...\n\n";
        io_context.run();

        std::cout << "\nAll coroutines completed!\n";
        
        // Final metrics
        if (channel->has_metrics()) {
            auto metrics = channel->get_metrics();
            std::cout << "\nFinal metrics:\n";
            std::cout << "  Total messages sent: " << metrics.messages_sent << "\n";
            std::cout << "  Total messages received: " << metrics.messages_received << "\n";
            std::cout << "  Send blocks: " << metrics.send_blocks << "\n";
            std::cout << "  Receive blocks: " << metrics.receive_blocks << "\n";
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}