#include "channel/channel_impl.hpp"
#include "channel/ipc_channel.hpp"
#include "channel/tcp_channel.hpp"
#include "channel/udp_multicast_channel.hpp"
#include "channel/unix_channel.hpp"
#include "channel/websocket_channel.hpp"
#include "memory/ring_buffer_impl.hpp"
#include <psyne/psyne.hpp>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>

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

    const std::string &uri() const override {
        return impl_->uri();
    }

    ChannelType type() const override {
        return impl_->type();
    }

    ChannelMode mode() const override {
        return impl_->mode();
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
    detail::ChannelImpl *impl() override {
        return impl_.get();
    }
    const detail::ChannelImpl *impl() const override {
        return impl_.get();
    }

    std::unique_ptr<ChannelImpl> impl_;
};

} // namespace detail

std::string get_performance_summary() {
    // Basic performance summary with runtime information
    std::ostringstream summary;
    
    summary << "=== Psyne Performance Summary ===\n";
    summary << "Version: " << version() << "\n";
    summary << "Build Configuration:\n";
    
    // Check compile-time features
    #ifdef PSYNE_CUDA_SUPPORT
    summary << "  - CUDA support: ENABLED\n";
    #else
    summary << "  - CUDA support: DISABLED\n";
    #endif
    
    #ifdef PSYNE_ASYNC_SUPPORT
    summary << "  - Async support: ENABLED\n";
    #else
    summary << "  - Async support: DISABLED\n";
    #endif
    
    #ifdef NDEBUG
    summary << "  - Build type: RELEASE\n";
    #else
    summary << "  - Build type: DEBUG\n";
    #endif
    
    // System information
    summary << "System Information:\n";
    summary << "  - Hardware concurrency: " << std::thread::hardware_concurrency() << " threads\n";
    
    // Memory alignment for zero-copy operations
    summary << "Memory Configuration:\n";
    summary << "  - Cache line size: " << std::hardware_destructive_interference_size << " bytes\n";
    summary << "  - Zero-copy aligned: " << (std::hardware_destructive_interference_size >= 64 ? "YES" : "NO") << "\n";
    
    // Transport availability
    summary << "Transport Support:\n";
    summary << "  - Memory channels: AVAILABLE\n";
    summary << "  - IPC channels: AVAILABLE\n";
    summary << "  - TCP channels: AVAILABLE (Boost ASIO)\n";
    summary << "  - UDP multicast: AVAILABLE\n";
    summary << "  - WebRTC: AVAILABLE\n";
    summary << "  - QUIC: AVAILABLE\n";
    summary << "  - WebSocket: AVAILABLE\n";
    
    summary << "\nFor detailed benchmarks, run psyne performance tests.";
    
    return summary.str();
}

std::vector<std::string> get_performance_recommendations() {
    return {"Enable zero-copy mode", "Use appropriate buffer sizes",
            "Consider RDMA for ultra-low latency"};
}

} // namespace psyne