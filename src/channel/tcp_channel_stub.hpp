#pragma once

#include <psyne/psyne.hpp>
#include <stdexcept>

namespace psyne {
namespace detail {

/**
 * @brief Stub TCP Channel implementation
 *
 * This is a placeholder implementation that throws runtime errors.
 * The full TCP implementation requires Boost ASIO which was removed
 * to minimize dependencies per project requirements.
 *
 * To enable TCP support:
 * 1. Install Boost development libraries
 * 2. Uncomment Boost in CMakeLists.txt
 * 3. Replace this file with the full tcp_channel.hpp implementation
 */
class TCPChannel : public Channel {
public:
    TCPChannel(const std::string &uri, size_t buffer_size, ChannelMode mode,
               ChannelType type, uint16_t port,
               const compression::CompressionConfig &compression_config = {})
        : Channel(uri, buffer_size, type) {
        throw std::runtime_error(
            "TCP Channel not implemented. Requires Boost ASIO.");
    }

    TCPChannel(const std::string &uri, size_t buffer_size, ChannelMode mode,
               ChannelType type, const std::string &host, uint16_t port,
               const compression::CompressionConfig &compression_config = {})
        : Channel(uri, buffer_size, type) {
        throw std::runtime_error(
            "TCP Channel not implemented. Requires Boost ASIO.");
    }

    ~TCPChannel() = default;

    // These methods don't exist in Channel base class - remove them
};

// Factory function to create TCP channels from URI
std::unique_ptr<Channel> create_tcp_channel(
    const std::string &uri, size_t buffer_size, ChannelMode mode,
    ChannelType type,
    const compression::CompressionConfig &compression_config = {}) {
    throw std::runtime_error(
        "TCP Channel not implemented. Requires Boost ASIO. "
        "Install Boost and rebuild with TCP support enabled.");
}

} // namespace detail
} // namespace psyne