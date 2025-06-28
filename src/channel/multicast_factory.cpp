#include <psyne/psyne.hpp>
#include "channel_impl.hpp"
#include "udp_multicast_channel.hpp"

namespace psyne {

// Wrapper for multicast channels
namespace detail {

class MulticastChannelWrapper : public Channel {
public:
    explicit MulticastChannelWrapper(std::unique_ptr<UDPMulticastChannel> impl)
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
    
    void* receive_raw_message(size_t& size, uint32_t& type) override {
        return impl_->receive_message(size, type);
    }
    
    void release_raw_message(void* handle) override {
        impl_->release_message(handle);
    }
    
    bool has_metrics() const override {
        return impl_->has_metrics();
    }
    
    debug::ChannelMetrics get_metrics() const override {
        return impl_->get_metrics();
    }
    
    void reset_metrics() override {
        impl_->reset_metrics();
    }
    
    // Multicast-specific methods
    UDPMulticastChannel::MulticastStats get_multicast_stats() const {
        return impl_->get_multicast_stats();
    }
    
    void set_ttl(int ttl) {
        impl_->set_ttl(ttl);
    }
    
    void set_loopback(bool enable) {
        impl_->set_loopback(enable);
    }
    
private:
    detail::ChannelImpl* impl() override { return impl_.get(); }
    const detail::ChannelImpl* impl() const override { return impl_.get(); }
    
    std::unique_ptr<UDPMulticastChannel> impl_;
};

} // namespace detail

namespace multicast {

std::unique_ptr<Channel> create_publisher(
    const std::string& multicast_address, uint16_t port,
    size_t buffer_size,
    const compression::CompressionConfig& compression_config) {
    
    std::string uri = "udp://" + multicast_address + ":" + std::to_string(port);
    
    auto impl = std::make_unique<detail::UDPMulticastChannel>(
        uri, buffer_size, ChannelMode::SPMC, ChannelType::MultiType,
        detail::MulticastRole::Publisher, compression_config, "");
    
    return std::make_unique<detail::MulticastChannelWrapper>(std::move(impl));
}

std::unique_ptr<Channel> create_subscriber(
    const std::string& multicast_address, uint16_t port,
    size_t buffer_size,
    const std::string& interface_address) {
    
    std::string uri = "udp://" + multicast_address + ":" + std::to_string(port);
    
    compression::CompressionConfig empty_config{};
    auto impl = std::make_unique<detail::UDPMulticastChannel>(
        uri, buffer_size, ChannelMode::SPMC, ChannelType::MultiType,
        detail::MulticastRole::Subscriber, empty_config, interface_address);
    
    return std::make_unique<detail::MulticastChannelWrapper>(std::move(impl));
}

std::unique_ptr<Channel> create_multicast_channel(
    const std::string& multicast_address, uint16_t port,
    Role role,
    size_t buffer_size,
    const compression::CompressionConfig& compression_config,
    const std::string& interface_address) {
    
    if (role == Role::Publisher) {
        return create_publisher(multicast_address, port, buffer_size, compression_config);
    } else {
        return create_subscriber(multicast_address, port, buffer_size, interface_address);
    }
}

} // namespace multicast

} // namespace psyne