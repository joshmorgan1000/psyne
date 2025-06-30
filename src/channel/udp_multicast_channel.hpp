#pragma once

#include "../compression/compression.hpp"
#include "channel_impl.hpp"
#include <atomic>
#include <boost/asio.hpp>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <span>

namespace psyne {
namespace detail {

namespace asio = boost::asio;
using udp = asio::ip::udp;

/**
 * @struct UDPMulticastHeader
 * @brief Header for UDP multicast messages
 */
struct UDPMulticastHeader {
    uint32_t magic; ///< Magic number for validation (0x50594E45 = "PYNE")
    uint32_t sequence_number; ///< Sequence number for ordering
    uint32_t message_length;  ///< Total message length including header
    uint32_t message_type;    ///< Message type identifier
    uint64_t timestamp;       ///< Timestamp (microseconds since epoch)
    uint64_t checksum;        ///< xxHash64 checksum of payload
    uint32_t original_size;   ///< Original size before compression (0 if not
                              ///< compressed)
    uint8_t compression_type; ///< Compression algorithm used
    uint8_t flags;            ///< Additional flags
    uint16_t reserved;        ///< Reserved for future use
};

static_assert(sizeof(UDPMulticastHeader) == 40,
              "UDPMulticastHeader must be 40 bytes");

/**
 * @enum MulticastRole
 * @brief Role of the multicast endpoint
 */
enum class MulticastRole {
    Publisher, ///< Sends multicast messages
    Subscriber ///< Receives multicast messages
};

/**
 * @class UDPMulticastChannel
 * @brief UDP multicast channel implementation
 *
 * Supports one-to-many broadcasting using IP multicast.
 * Publishers send to a multicast group, subscribers join the group to receive.
 */
class UDPMulticastChannel : public ChannelImpl {
public:
    /**
     * @brief Constructor for UDP multicast channel
     * @param uri Channel URI (e.g., "udp://239.255.0.1:12345")
     * @param buffer_size Buffer size for messages
     * @param mode Channel mode (should be SPMC for multicast)
     * @param type Channel type
     * @param role Publisher or Subscriber role
     * @param compression_config Compression configuration
     * @param interface_address Local interface to bind to (optional)
     */
    UDPMulticastChannel(
        const std::string &uri, size_t buffer_size, ChannelMode mode,
        ChannelType type, MulticastRole role,
        const compression::CompressionConfig &compression_config = {},
        const std::string &interface_address = "");

    ~UDPMulticastChannel();

    // Zero-copy interface
    uint32_t reserve_write_slot(size_t size) noexcept override;
    void notify_message_ready(uint32_t offset, size_t size) noexcept override;
    std::span<uint8_t> get_write_span(size_t size) noexcept;
    std::span<const uint8_t> buffer_span() const noexcept;
    void advance_read_pointer(size_t size) noexcept override;
    RingBuffer& get_ring_buffer() override;
    const RingBuffer& get_ring_buffer() const;
    
    // Legacy interface (deprecated)
    [[deprecated("Use reserve_write_slot() instead")]]
    void *reserve_space(size_t size) override;
    [[deprecated("Data is committed when written")]]
    void commit_message(void *handle) override;
    void *receive_message(size_t &size, uint32_t &type) override;
    void release_message(void *handle) override;

private:
    /**
     * @brief Zero-copy UDP multicast using scatter-gather I/O
     */
    void start_zero_copy_multicast(const UDPMulticastHeader& header, uint32_t offset, size_t size);

public:
    /**
     * @brief Join a multicast group (for subscribers)
     */
    void join_group();

    /**
     * @brief Leave a multicast group (for subscribers)
     */
    void leave_group();

    /**
     * @brief Set Time-To-Live for multicast packets (for publishers)
     */
    void set_ttl(int ttl);

    /**
     * @brief Enable/disable loopback (receiving own messages)
     */
    void set_loopback(bool enable);

    /**
     * @brief Get multicast statistics
     */
    struct MulticastStats {
        uint64_t packets_sent = 0;
        uint64_t packets_received = 0;
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        uint64_t packets_dropped = 0;
        uint64_t sequence_errors = 0;
        uint32_t last_sequence_number = 0;
    };

    MulticastStats get_multicast_stats() const {
        return stats_;
    }

private:
    // Network components
    asio::io_context io_context_;
    std::unique_ptr<udp::socket> socket_;
    udp::endpoint multicast_endpoint_;
    udp::endpoint local_endpoint_;
    udp::endpoint sender_endpoint_tmp_; // Temporary for async_receive_from
    std::thread io_thread_;

    // Message handling
    struct PendingMessage {
        std::vector<uint8_t> data;
        uint32_t type;
        uint32_t sequence_number;
        uint64_t timestamp;
    };

    std::queue<PendingMessage> send_queue_;
    std::queue<PendingMessage> recv_queue_;
    std::mutex send_mutex_;
    std::mutex recv_mutex_;
    std::condition_variable recv_cv_;

    // Buffers
    std::vector<uint8_t> send_buffer_;
    std::vector<uint8_t> recv_buffer_;
    std::vector<uint8_t> temp_recv_buffer_;

    // State
    MulticastRole role_;
    bool joined_group_;
    std::atomic<bool> stopping_;
    std::atomic<uint32_t> sequence_number_;

    // Configuration
    std::string interface_address_;
    int ttl_;
    bool loopback_enabled_;

    // Compression support
    compression::CompressionManager compression_manager_;
    std::vector<uint8_t> compression_buffer_;
    
    // Delegate to memory channel for ring buffer operations
    std::unique_ptr<Channel> memory_channel_;

    // Statistics
    mutable std::mutex stats_mutex_;
    mutable MulticastStats stats_;

    // Async operations
    void start_receive();
    void handle_receive(const boost::system::error_code &error,
                        size_t bytes_transferred,
                        udp::endpoint sender_endpoint);
    void start_send();
    void handle_send(const boost::system::error_code &error,
                     size_t bytes_transferred);
    void run_io_service();

    // Helper functions
    void setup_socket();
    void parse_uri(const std::string &uri);
    uint64_t calculate_checksum(const uint8_t *data, size_t size);
    bool validate_message(const UDPMulticastHeader &header,
                          const uint8_t *payload, size_t payload_size);
    void update_stats_sent(size_t bytes);
    void update_stats_received(size_t bytes);
    void update_stats_dropped();
    void update_stats_sequence_error();
};

/**
 * @brief Factory function to create UDP multicast channels
 */
std::unique_ptr<ChannelImpl> create_udp_multicast_channel(
    const std::string &uri, size_t buffer_size, ChannelMode mode,
    ChannelType type, MulticastRole role,
    const compression::CompressionConfig &compression_config = {},
    const std::string &interface_address = "");

/**
 * @brief Create a UDP multicast publisher
 */
inline std::unique_ptr<ChannelImpl> create_multicast_publisher(
    const std::string &multicast_address, uint16_t port,
    size_t buffer_size = 1024 * 1024,
    const compression::CompressionConfig &compression_config = {}) {
    std::string uri = "udp://" + multicast_address + ":" + std::to_string(port);
    return create_udp_multicast_channel(
        uri, buffer_size, ChannelMode::SPMC, ChannelType::MultiType,
        MulticastRole::Publisher, compression_config);
}

/**
 * @brief Create a UDP multicast subscriber
 */
inline std::unique_ptr<ChannelImpl>
create_multicast_subscriber(const std::string &multicast_address, uint16_t port,
                            size_t buffer_size = 1024 * 1024,
                            const std::string &interface_address = "") {
    std::string uri = "udp://" + multicast_address + ":" + std::to_string(port);
    return create_udp_multicast_channel(
        uri, buffer_size, ChannelMode::SPMC, ChannelType::MultiType,
        MulticastRole::Subscriber, {}, interface_address);
}

} // namespace detail
} // namespace psyne