#pragma once

#include <psyne/psyne.hpp>
#include <memory>
#include <string>

namespace psyne {
namespace detail {

/**
 * @brief Simple UDP multicast implementation that delegates to memory channels
 * 
 * This provides a working UDP multicast API while the full zero-copy 
 * implementation is being developed. Uses memory channels internally
 * with UDP multicast for actual networking.
 */
class SimpleUDPMulticastChannel {
public:
    enum class Role {
        Publisher,
        Subscriber
    };

    SimpleUDPMulticastChannel(const std::string& multicast_address, 
                              uint16_t port, 
                              Role role,
                              size_t buffer_size = 1024 * 1024);
    
    ~SimpleUDPMulticastChannel();

    // Forward channel interface to internal memory channel
    std::string uri() const;
    ChannelMode mode() const;
    ChannelType type() const;
    void stop();
    bool is_stopped() const;
    void* receive_message(size_t& size, uint32_t& type);
    void release_message(void* handle);
    bool has_metrics() const;
    debug::ChannelMetrics get_metrics() const;
    void reset_metrics();

    // Simple interface for messages
    template<typename T>
    void send(const T& message) {
        if (role_ == Role::Publisher && memory_channel_) {
            // Send via memory channel (local delivery)
            try {
                // Create message in memory channel
                auto typed_channel = std::dynamic_pointer_cast<Channel<T>>(memory_channel_);
                if (typed_channel) {
                    T msg(*typed_channel);
                    msg = message; // Copy message content
                    msg.send();
                }
            } catch (const std::exception&) {
                // Failed to send via memory channel
            }
            
            // Network thread will handle actual UDP multicast delivery
        }
    }

    template<typename T>
    std::optional<T> receive() {
        if (role_ == Role::Subscriber && memory_channel_) {
            // Receive from memory channel 
            try {
                auto typed_channel = std::dynamic_pointer_cast<Channel<T>>(memory_channel_);
                if (typed_channel) {
                    size_t size;
                    uint32_t type;
                    void* data = typed_channel->receive_message(size, type);
                    if (data) {
                        T result(*typed_channel);
                        std::memcpy(result.data(), data, size);
                        typed_channel->release_message(data);
                        return result;
                    }
                }
            } catch (const std::exception&) {
                // Failed to receive from memory channel
            }
            
            // Network thread handles receiving from UDP multicast
        }
        return std::nullopt;
    }

private:
    std::string multicast_address_;
    uint16_t port_;
    Role role_;
    std::unique_ptr<Channel> memory_channel_;
    
    // Network socket (platform specific)
#ifdef _WIN32
    SOCKET socket_ = INVALID_SOCKET;
#else
    int socket_ = -1;
#endif
    std::thread network_thread_;
    std::atomic<bool> running_{false};
    
    void setup_udp_multicast();
    void network_send_loop();
    void network_receive_loop();
};

} // namespace detail
} // namespace psyne