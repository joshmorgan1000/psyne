#include <psyne/psyne.hpp>
#include "channel/channel_impl.hpp"
#include "memory/ring_buffer_impl.hpp"
#include "channel/ipc_channel.hpp"
#include "channel/tcp_channel.hpp"
#include "channel/unix_channel.hpp"
#include "channel/udp_multicast_channel.hpp"
#include "channel/websocket_channel.hpp"
#include <stdexcept>
#include <regex>

namespace psyne {

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
    
private:
    detail::ChannelImpl* impl() override { return impl_.get(); }
    const detail::ChannelImpl* impl() const override { return impl_.get(); }
    
    std::unique_ptr<ChannelImpl> impl_;
};

} // namespace detail


std::string get_performance_summary() {
    return "Performance monitoring not yet implemented";
}

std::vector<std::string> get_performance_recommendations() {
    return {"Enable zero-copy mode", "Use appropriate buffer sizes", "Consider RDMA for ultra-low latency"};
}

} // namespace psyne