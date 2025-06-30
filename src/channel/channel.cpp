#include "../memory/ring_buffer_impl.hpp"
#include "channel_impl.hpp"
#include "ipc_channel.hpp"
#include "tcp_channel.hpp"
#include "udp_multicast_channel.hpp"
#include "unix_channel.hpp"
#include "webrtc_channel.hpp"
#include "websocket_channel.hpp"
#include <psyne/psyne.hpp>
#include <regex>
#include <stdexcept>

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

// Factory for creating ring buffer based channels
template <typename RingBufferType>
std::unique_ptr<ChannelImpl>
create_ring_buffer_channel(const std::string &uri, size_t buffer_size,
                           ChannelMode mode, ChannelType type,
                           bool enable_metrics) {
    return std::make_unique<ChannelImplT<RingBufferType>>(
        uri, buffer_size, mode, type, enable_metrics);
}

} // namespace detail

// Public API implementation
std::unique_ptr<Channel>
Channel::create(const std::string &uri, size_t buffer_size, ChannelMode mode,
                ChannelType type, bool enable_metrics,
                const compression::CompressionConfig &compression_config) {
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
                uri, buffer_size, mode, type, enable_metrics);
            break;
        case ChannelMode::SPMC:
            impl = detail::create_ring_buffer_channel<detail::SPMCRingBuffer>(
                uri, buffer_size, mode, type, enable_metrics);
            break;
        case ChannelMode::MPSC:
            impl = detail::create_ring_buffer_channel<detail::MPSCRingBuffer>(
                uri, buffer_size, mode, type, enable_metrics);
            break;
        case ChannelMode::MPMC:
            impl = detail::create_ring_buffer_channel<detail::MPMCRingBuffer>(
                uri, buffer_size, mode, type, enable_metrics);
            break;
        }
    } else if (scheme == "ipc") {
        // Create IPC channel (metrics not yet supported)
        impl =
            std::make_unique<detail::IPCChannel>(uri, buffer_size, mode, type);
    } else if (scheme == "tcp") {
        // Create TCP channel with compression support
        impl = detail::create_tcp_channel(uri, buffer_size, mode, type,
                                          compression_config);
    } else if (scheme == "unix") {
        // Create Unix domain socket channel (metrics not yet supported)
        // For unix:///path/to/socket or unix://path/to/socket
        std::string socket_path = path;
        if (socket_path.empty()) {
            throw std::invalid_argument("Unix socket path cannot be empty");
        }

        // If path starts with /, it's already absolute
        // Otherwise, make it relative to current directory
        if (socket_path[0] != '/' && socket_path.substr(0, 2) != "./") {
            socket_path = "./" + socket_path;
        }

        // Determine if this is server or client based on presence of '@' prefix
        // unix://@/path/to/socket for server (listening)
        // unix:///path/to/socket for client (connecting)
        bool is_server = false;
        if (socket_path[0] == '@') {
            is_server = true;
            socket_path = socket_path.substr(1); // Remove @ prefix
        }

        if (is_server) {
            impl = std::make_unique<detail::UnixChannel>(uri, buffer_size, mode,
                                                         type, socket_path);
        } else {
            impl = std::make_unique<detail::UnixChannel>(
                uri, buffer_size, mode, type, socket_path, true);
        }
    } else if (scheme == "ws" || scheme == "wss") {
        // Create WebSocket channel
        // ws://host:port for client, ws://:port for server
        bool is_server = path[0] == ':';
        impl = std::make_unique<detail::WebSocketChannel>(uri, buffer_size,
                                                          is_server);
    } else if (scheme == "webrtc") {
        // Create WebRTC channel
        // webrtc://peer-id or webrtc://signaling-server/room-id
        detail::WebRTCConfig webrtc_config;

        // Add default STUN servers
        webrtc_config.stun_servers.push_back(
            {"stun.l.google.com", 19302, "", ""});
        webrtc_config.stun_servers.push_back(
            {"stun1.l.google.com", 19302, "", ""});

        // Default signaling server (can be overridden)
        std::string signaling_server = "ws://localhost:8080";

        impl = detail::create_webrtc_channel(uri, buffer_size, mode, type,
                                             webrtc_config, signaling_server,
                                             compression_config);
    } else if (scheme == "quic") {
        // Create QUIC channel
        // quic://host:port for client, quic://:port for server
        bool is_server = path[0] == ':';
        impl = detail::create_quic_channel(uri, buffer_size, mode, type,
                                           compression_config);
    } else {
        throw std::invalid_argument("Unknown URI scheme: " + scheme);
    }

    return std::make_unique<detail::ChannelWrapper>(std::move(impl));
}

// Factory function is now inline in header

} // namespace psyne