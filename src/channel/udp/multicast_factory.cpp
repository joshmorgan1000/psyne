#include <psyne/psyne.hpp>
#include "udp_multicast_channel.hpp"
#include "../channel_impl.hpp"

namespace psyne {

namespace detail {

/**
 * @brief Wrapper class to adapt UDPMulticastChannel to the Channel interface
 */
class MulticastChannelWrapper : public Channel {
public:
    MulticastChannelWrapper(std::unique_ptr<detail::UDPMulticastChannel> impl)
        : impl_(std::move(impl)) {}

    // Zero-copy interface
    uint32_t reserve_write_slot(size_t size) noexcept override {
        return impl_->reserve_write_slot(size);
    }

    void notify_message_ready(uint32_t offset, size_t size) noexcept override {
        impl_->notify_message_ready(offset, size);
    }

    RingBuffer& get_ring_buffer() noexcept override {
        return impl_->get_ring_buffer();
    }

    const RingBuffer& get_ring_buffer() const noexcept override {
        return impl_->get_ring_buffer();
    }

    void advance_read_pointer(size_t size) noexcept override {
        impl_->advance_read_pointer(size);
    }

    // Channel properties
    const std::string& uri() const override {
        return impl_->uri();
    }

    ChannelMode mode() const override {
        return impl_->mode();
    }

    ChannelType type() const override {
        return impl_->type();
    }

    void stop() override {
        impl_->stop();
    }

    bool is_stopped() const override {
        return impl_->is_stopped();
    }

    void *receive_raw_message(size_t &size, uint32_t &type) override {
        return impl_->receive_message(size, type);
    }

    void release_raw_message(void *handle) override {
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

private:
    std::unique_ptr<detail::UDPMulticastChannel> impl_;
};

} // namespace detail

namespace multicast {

std::unique_ptr<Channel>
create_publisher(const std::string &multicast_address, uint16_t port,
                 size_t buffer_size,
                 const compression::CompressionConfig &compression_config) {
    std::string uri = "udp://" + multicast_address + ":" + std::to_string(port);
    auto impl = std::make_unique<detail::UDPMulticastChannel>(
        uri, buffer_size, ChannelMode::SPMC, ChannelType::MultiType,
        detail::MulticastRole::Publisher, compression_config, "");
    return std::make_unique<detail::MulticastChannelWrapper>(std::move(impl));
}

std::unique_ptr<Channel>
create_subscriber(const std::string &multicast_address, uint16_t port,
                  size_t buffer_size, const std::string &interface_address) {
    std::string uri = "udp://" + multicast_address + ":" + std::to_string(port);
    compression::CompressionConfig empty_config{};
    auto impl = std::make_unique<detail::UDPMulticastChannel>(
        uri, buffer_size, ChannelMode::SPMC, ChannelType::MultiType,
        detail::MulticastRole::Subscriber, empty_config, interface_address);
    return std::make_unique<detail::MulticastChannelWrapper>(std::move(impl));
}

std::unique_ptr<Channel> create_multicast_channel(
    const std::string &multicast_address, uint16_t port, Role role,
    size_t buffer_size,
    const compression::CompressionConfig &compression_config,
    const std::string &interface_address) {
    if (role == Role::Publisher) {
        return create_publisher(multicast_address, port, buffer_size,
                                compression_config);
    } else {
        return create_subscriber(multicast_address, port, buffer_size,
                                 interface_address);
    }
}

} // namespace multicast

} // namespace psyne