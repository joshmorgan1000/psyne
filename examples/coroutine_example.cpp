#include <psyne/psyne.hpp>
#include <boost/asio.hpp>
#include <boost/asio/co_spawn.hpp>
#include <iostream>
#include <cstring>

using namespace psyne;
namespace asio = boost::asio;

// Simple text message
class TextMessage : public Message<TextMessage> {
public:
    static constexpr uint32_t message_type = 100;
    static constexpr size_t max_text_size = 256;
    
    template<typename Channel>
    explicit TextMessage(Channel& channel) : Message<TextMessage>(channel) {
        if (this->data_) {
            std::memset(this->data_, 0, max_text_size);
        }
    }
    
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
    
    void before_send() {}
};

// Coroutine that processes messages
asio::awaitable<void> message_processor(SPSCChannel& channel) {
    std::cout << "Starting message processor coroutine..." << std::endl;
    
    while (true) {
        // Use async receive - this will yield control while waiting
        auto msg_opt = co_await channel.async_receive_single<TextMessage>();
        
        if (msg_opt) {
            auto& msg = *msg_opt;
            std::string text = msg.get_text();
            std::cout << "Coroutine received: " << text << std::endl;
            
            // Simulate some async processing
            asio::steady_timer timer(co_await asio::this_coro::executor);
            timer.expires_after(std::chrono::milliseconds(100));
            co_await timer.async_wait(asio::use_awaitable);
            
            std::cout << "Processed: " << text << std::endl;
            
            if (text == "quit") {
                break;
            }
        }
    }
    
    std::cout << "Message processor coroutine finished." << std::endl;
}

// Coroutine that sends messages
asio::awaitable<void> message_sender(SPSCChannel& channel) {
    std::vector<std::string> messages = {
        "Hello from coroutine!",
        "This is an async message",
        "Boost.Asio coroutines are powerful",
        "quit"
    };
    
    for (const auto& text : messages) {
        // Create and send message
        TextMessage msg(channel);
        msg.set_text(text);
        msg.send();
        
        std::cout << "Sent: " << text << std::endl;
        
        // Wait a bit between messages
        asio::steady_timer timer(co_await asio::this_coro::executor);
        timer.expires_after(std::chrono::milliseconds(200));
        co_await timer.async_wait(asio::use_awaitable);
    }
    
    std::cout << "Message sender coroutine finished." << std::endl;
}

int main() {
    try {
        // Create io_context for coroutines
        asio::io_context io_context;
        
        // Create channel
        const size_t buffer_size = 64 * 1024;
        SPSCChannel channel("memory://coroutine_demo", buffer_size, ChannelType::SingleType);
        
        std::cout << "Starting coroutine example..." << std::endl;
        
        // Spawn coroutines
        asio::co_spawn(io_context, message_processor(channel), asio::detached);
        asio::co_spawn(io_context, message_sender(channel), asio::detached);
        
        // Run the io_context
        io_context.run();
        
        std::cout << "Coroutine example completed." << std::endl;
        
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    
    return 0;
}