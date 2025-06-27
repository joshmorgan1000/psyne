#include <psyne/psyne.hpp>
#include "channel_impl.hpp"
#include "../memory/ring_buffer_impl.hpp"
#include "ipc_channel.hpp"
#include "tcp_channel.hpp"
#include <stdexcept>
#include <regex>

namespace psyne {

// Internal implementation
namespace detail {

class ChannelWrapper : public Channel {
public:
    explicit ChannelWrapper(std::unique_ptr<ChannelImpl> impl)
        : impl_(std::move(impl)) {}
        
    void stop() override {
        impl_->stop();
    }
    
    bool is_stopped() const override {
        return impl_->is_stopped();
    }
    
    const std::string& uri() const override {
        return impl_->uri();
    }
    
    ChannelType type() const override {
        return impl_->type();
    }
    
    ChannelMode mode() const override {
        return impl_->mode();
    }
    
private:
    detail::ChannelImpl* impl() override { return impl_.get(); }
    const detail::ChannelImpl* impl() const override { return impl_.get(); }
    
    std::unique_ptr<ChannelImpl> impl_;
};

// Factory for creating ring buffer based channels
template<typename RingBufferType>
std::unique_ptr<ChannelImpl> create_ring_buffer_channel(
    const std::string& uri, size_t buffer_size, ChannelMode mode, ChannelType type) {
    return std::make_unique<ChannelImplT<RingBufferType>>(uri, buffer_size, mode, type);
}

} // namespace detail

// Public API implementation
std::unique_ptr<Channel> Channel::create(
    const std::string& uri,
    size_t buffer_size,
    ChannelMode mode,
    ChannelType type) {
    
    // Parse URI to determine channel type
    std::regex uri_regex("^(\\w+)://(.+)$");
    std::smatch match;
    
    if (!std::regex_match(uri, match, uri_regex)) {
        throw std::invalid_argument("Invalid URI format: " + uri);
    }
    
    std::string scheme = match[1];
    std::string path = match[2];
    
    std::unique_ptr<detail::ChannelImpl> impl;
    
    if (scheme == "memory") {
        // Create appropriate ring buffer based on mode
        switch (mode) {
            case ChannelMode::SPSC:
                impl = detail::create_ring_buffer_channel<detail::SPSCRingBuffer>(
                    uri, buffer_size, mode, type);
                break;
            case ChannelMode::SPMC:
                impl = detail::create_ring_buffer_channel<detail::SPMCRingBuffer>(
                    uri, buffer_size, mode, type);
                break;
            case ChannelMode::MPSC:
                impl = detail::create_ring_buffer_channel<detail::MPSCRingBuffer>(
                    uri, buffer_size, mode, type);
                break;
            case ChannelMode::MPMC:
                impl = detail::create_ring_buffer_channel<detail::MPMCRingBuffer>(
                    uri, buffer_size, mode, type);
                break;
        }
    } else if (scheme == "ipc") {
        // Create IPC channel
        impl = std::make_unique<detail::IPCChannel>(uri, buffer_size, mode, type);
    } else if (scheme == "tcp") {
        // Create TCP channel
        impl = detail::create_tcp_channel(uri, buffer_size, mode, type);
    } else {
        throw std::invalid_argument("Unknown URI scheme: " + scheme);
    }
    
    return std::make_unique<detail::ChannelWrapper>(std::move(impl));
}

// Factory function is now inline in header

} // namespace psyne