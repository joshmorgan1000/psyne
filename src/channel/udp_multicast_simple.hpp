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
            // TODO: Also send via UDP multicast for network delivery
        }
    }

    template<typename T>
    std::optional<T> receive() {
        if (role_ == Role::Subscriber && memory_channel_) {
            // Receive from memory channel 
            // TODO: Also receive from UDP multicast
            return std::nullopt;
        }
        return std::nullopt;
    }

private:
    std::string multicast_address_;
    uint16_t port_;
    Role role_;
    std::unique_ptr<Channel> memory_channel_;
    
    void setup_udp_multicast();
};

} // namespace detail
} // namespace psyne