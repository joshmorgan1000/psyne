#pragma once

#include "../core/message.hpp"
#include "../memory/ring_buffer.hpp"
#include <memory>
#include <string>
#include <functional>
#include <thread>
#include <chrono>
#include <atomic>
#include <variant>
#include <unordered_map>
#include <optional>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/use_awaitable.hpp>

namespace psyne {

// Channel modes
enum class ChannelMode {
    SPSC,
    SPMC, 
    MPSC,
    MPMC
};

enum class ChannelType {
    SingleType,   // Optimized for single message type (no metadata)
    MultiType     // Supports multiple types (small overhead)
};

// Message envelope for multi-type channels
struct MessageEnvelope {
    uint32_t type;
    uint32_t size;
    // Data follows immediately after
    
    void* data() { return reinterpret_cast<uint8_t*>(this) + sizeof(MessageEnvelope); }
    const void* data() const { return reinterpret_cast<const uint8_t*>(this) + sizeof(MessageEnvelope); }
};

template<typename RingBufferType>
class Channel {
public:
    Channel(const std::string& uri, size_t buffer_size, ChannelType type = ChannelType::MultiType)
        : uri_(uri)
        , channel_type_(type)
        , ring_buffer_(buffer_size > 0 ? buffer_size : 1024) {}
    
    virtual ~Channel() = default;
    
    // For single-type channels - create message directly
    template<typename MessageType>
    MessageType create_message() {
        static_assert(std::is_base_of_v<Message<MessageType>, MessageType>);
        
        if (channel_type_ == ChannelType::SingleType) {
            // Direct allocation, no envelope
            return MessageType(*this);
        } else {
            // With envelope for type info
            size_t total_size = sizeof(MessageEnvelope) + MessageType::calculate_size();
            auto handle = ring_buffer_.reserve(total_size);
            if (!handle) return MessageType(nullptr, 0);
            
            auto* envelope = reinterpret_cast<const MessageEnvelope*>(handle->data);
            envelope->type = MessageType::message_type;
            envelope->size = MessageType::calculate_size();
            
            // Create message in the data portion
            return MessageType(envelope->data(), envelope->size);
        }
    }
    
    // Send for messages created on this channel
    template<typename MessageType>
    void send(MessageType& msg) {
        msg.send();
        notify();
    }
    
    // Direct send (copies data - avoid when possible)
    template<typename MessageType>
    [[deprecated("Violates zero-copy principle. Create messages directly in channel instead.")]]
    bool send_copy(const MessageType& msg) {
        size_t required_size = channel_type_ == ChannelType::SingleType 
            ? MessageType::calculate_size()
            : sizeof(MessageEnvelope) + MessageType::calculate_size();
            
        auto handle = ring_buffer_.reserve(required_size);
        if (!handle) return false;
        
        if (channel_type_ == ChannelType::MultiType) {
            auto* envelope = reinterpret_cast<const MessageEnvelope*>(handle->data);
            envelope->type = MessageType::message_type;
            envelope->size = MessageType::calculate_size();
            // Copy message data
            msg.copy_to(envelope->data());
        } else {
            // Copy message data directly
            msg.copy_to(handle->data);
        }
        
        handle->commit();
        notify();
        return true;
    }
    
    // Receive for single-type channels
    template<typename MessageType>
    std::optional<MessageType> receive_single(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        static_assert(std::is_base_of_v<Message<MessageType>, MessageType>);
        
        if (!wait_for_data(timeout)) return std::nullopt;
        
        auto handle = ring_buffer_.read();
        if (!handle) return std::nullopt;
        
        return MessageType(handle->data, handle->size);
    }
    
    // Receive for multi-type channels - returns type ID and message
    std::optional<std::pair<uint32_t, const void*>> receive_multi(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        if (!wait_for_data(timeout)) return std::nullopt;
        
        auto handle = ring_buffer_.read();
        if (!handle) return std::nullopt;
        
        auto* envelope = reinterpret_cast<const MessageEnvelope*>(handle->data);
        return std::make_pair(envelope->type, envelope->data());
    }
    
    // Type-safe receive for multi-type channels
    template<typename MessageType>
    std::optional<MessageType> receive_as(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        auto result = receive_multi(timeout);
        if (!result) return std::nullopt;
        
        auto [type, data] = *result;
        if (type != MessageType::message_type) return std::nullopt;
        
        return MessageType(data, MessageType::calculate_size());
    }
    
    // Listen with single handler (for single-type channels)
    template<typename MessageType>
    std::unique_ptr<std::thread> listen(std::function<void(MessageType&&)> handler) {
        return std::make_unique<std::thread>([this, handler]() {
            while (!stop_flag_.load(std::memory_order_acquire)) {
                auto msg = receive_single<MessageType>(std::chrono::milliseconds(100));
                if (msg) {
                    handler(std::move(*msg));
                }
            }
        });
    }
    
    // Listen with multiple handlers (for multi-type channels)
    std::unique_ptr<std::thread> listen(std::unordered_map<uint32_t, std::function<void(const void*)>> handlers) {
        return std::make_unique<std::thread>([this, handlers = std::move(handlers)]() {
            while (!stop_flag_.load(std::memory_order_acquire)) {
                auto result = receive_multi(std::chrono::milliseconds(100));
                if (result) {
                    auto [type, data] = *result;
                    auto it = handlers.find(type);
                    if (it != handlers.end()) {
                        it->second(data);
                    }
                }
            }
        });
    }
    
    // Helper to register typed handlers
    template<typename MessageType>
    static auto make_handler(std::function<void(MessageType&&)> func) {
        return std::make_pair(
            MessageType::message_type,
            std::function<void(const void*)>([func](const void* data) {
                MessageType msg(data, MessageType::calculate_size());
                func(std::move(msg));
            })
        );
    }
    
    void stop() {
        stop_flag_.store(true, std::memory_order_release);
    }
    
    virtual RingBufferType* ring_buffer() { return &ring_buffer_; }
    const std::string& uri() const { return uri_; }
    ChannelType type() const { return channel_type_; }
    
    // Coroutine support for async operations
    template<typename MessageType>
    boost::asio::awaitable<std::optional<MessageType>> async_receive_single() {
        // This is a simplified version - in a real implementation you'd want to
        // integrate with the io_context properly
        while (true) {
            auto msg = receive_single<MessageType>(std::chrono::milliseconds(10));
            if (msg) {
                co_return msg;
            }
            // Yield to allow other coroutines to run
            co_await boost::asio::this_coro::executor;
        }
    }
    
    // Async receive for multi-type channels
    boost::asio::awaitable<std::optional<std::pair<uint32_t, const void*>>> async_receive_multi() {
        while (true) {
            auto result = receive_multi(std::chrono::milliseconds(10));
            if (result) {
                co_return result;
            }
            // Yield to allow other coroutines to run
            co_await boost::asio::this_coro::executor;
        }
    }
    
    // Type-safe async receive
    template<typename MessageType>
    boost::asio::awaitable<std::optional<MessageType>> async_receive_as() {
        while (true) {
            auto msg = receive_as<MessageType>(std::chrono::milliseconds(10));
            if (msg) {
                co_return msg;
            }
            // Yield to allow other coroutines to run
            co_await boost::asio::this_coro::executor;
        }
    }
    
    virtual void notify() {
        // Subclasses override for IPC signaling
    }
    
protected:
    virtual bool wait_for_data(std::chrono::milliseconds timeout) {
        // Default implementation - can be overridden for IPC
        using namespace std::chrono;
        auto start = steady_clock::now();
        
        while (ring_buffer_.empty()) {
            if (timeout != milliseconds::zero() && 
                duration_cast<milliseconds>(steady_clock::now() - start) > timeout) {
                return false;
            }
            std::this_thread::yield();
        }
        return true;
    }
    
private:
    std::string uri_;
    ChannelType channel_type_;
    RingBufferType ring_buffer_;
    std::atomic<bool> stop_flag_{false};
};

// Convenience aliases
using SPSCChannel = Channel<SPSCRingBuffer>;
using SPMCChannel = Channel<SPMCRingBuffer>;
using MPSCChannel = Channel<MPSCRingBuffer>;
using MPMCChannel = Channel<MPMCRingBuffer>;

// Forward declarations for IPC and TCP channels
template<typename RingBufferType> class IPCChannel;
template<typename RingBufferType> class TCPChannel;

// Channel factory for URI-based creation
class ChannelFactory {
public:
    static bool is_ipc_uri(const std::string& uri) {
        return uri.find("ipc://") == 0;
    }
    
    static bool is_tcp_uri(const std::string& uri) {
        return uri.find("tcp://") == 0;
    }
    
    static bool is_memory_uri(const std::string& uri) {
        return uri.find("memory://") == 0;
    }
    
    static std::string extract_ipc_name(const std::string& uri) {
        if (!is_ipc_uri(uri)) return "";
        return uri.substr(6);  // Skip "ipc://"
    }
    
    static std::pair<std::string, uint16_t> extract_tcp_endpoint(const std::string& uri) {
        if (!is_tcp_uri(uri)) return {"", 0};
        
        std::string endpoint = uri.substr(6);  // Skip "tcp://"
        size_t colon_pos = endpoint.find(':');
        
        if (colon_pos == std::string::npos) {
            return {endpoint, 9999};  // Default port
        }
        
        std::string host = endpoint.substr(0, colon_pos);
        uint16_t port = static_cast<uint16_t>(std::stoi(endpoint.substr(colon_pos + 1)));
        
        return {host, port};
    }
    
    // Create channel from URI
    template<typename RingBufferType>
    static std::unique_ptr<Channel<RingBufferType>> create(
        const std::string& uri, 
        size_t buffer_size,
        ChannelType type = ChannelType::MultiType,
        bool create_new = true) {
        
        if (is_ipc_uri(uri)) {
            // Need to include IPC channel header
            auto name = extract_ipc_name(uri);
            return std::make_unique<IPCChannel<RingBufferType>>(name, buffer_size, create_new);
        } else if (is_tcp_uri(uri)) {
            // Need to include TCP channel header
            auto [host, port] = extract_tcp_endpoint(uri);
            bool is_server = (host == "0.0.0.0" || host == "::" || host == "*");
            return std::make_unique<TCPChannel<RingBufferType>>(host, port, buffer_size, is_server);
        } else if (is_memory_uri(uri)) {
            // In-memory channel
            return std::make_unique<Channel<RingBufferType>>(uri, buffer_size, type);
        } else {
            throw std::invalid_argument("Unknown channel URI scheme: " + uri);
        }
    }
};

}  // namespace psyne