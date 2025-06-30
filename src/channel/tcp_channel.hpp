#pragma once

#include "../compression/compression.hpp"
#include "channel_impl.hpp"
#include <boost/asio.hpp>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <span>

namespace psyne {
namespace detail {

namespace asio = boost::asio;
using tcp = asio::ip::tcp;

// TCP message frame header (16-byte aligned)
struct TCPFrameHeader {
    uint32_t length;   // Total message length (including this header)
    uint32_t type;     // Message type
    uint64_t checksum; // xxHash64 checksum of the payload
    uint32_t
        original_size; // Original size before compression (0 if not compressed)
    uint8_t compression_type;  // Compression algorithm used
    uint8_t compression_level; // Compression level used
    uint16_t flags;            // Additional flags for future use
};

// TCP Channel implementation
class TCPChannel : public ChannelImpl {
public:
    // Constructor for server (listening) mode
    TCPChannel(const std::string &uri, size_t buffer_size, ChannelMode mode,
               ChannelType type, uint16_t port,
               const compression::CompressionConfig &compression_config = {});

    // Constructor for client (connecting) mode
    TCPChannel(const std::string &uri, size_t buffer_size, ChannelMode mode,
               ChannelType type, const std::string &host, uint16_t port,
               const compression::CompressionConfig &compression_config = {});

    ~TCPChannel();

    // Zero-copy interface
    uint32_t reserve_write_slot(size_t size) noexcept override;
    void notify_message_ready(uint32_t offset, size_t size) noexcept override;
    RingBuffer& get_ring_buffer() noexcept override;
    const RingBuffer& get_ring_buffer() const noexcept override;
    void advance_read_pointer(size_t size) noexcept override;
    
    // TCP-specific zero-copy methods
    std::span<uint8_t> get_write_span(size_t size) noexcept;
    std::span<const uint8_t> buffer_span() const noexcept;

private:
    /**
     * @brief Zero-copy TCP write using scatter-gather I/O
     */
    void start_zero_copy_write(const TCPFrameHeader& header, uint32_t offset, size_t size);
    
public:
    // Legacy interface (deprecated)
    [[deprecated("Use reserve_write_slot() instead")]]
    void *reserve_space(size_t size) override;
    [[deprecated("Data is committed when written")]]
    void commit_message(void *handle) override;
    void *receive_message(size_t &size, uint32_t &type) override;
    void release_message(void *handle) override;

private:
    // Network components
    asio::io_context io_context_;
    std::unique_ptr<tcp::acceptor> acceptor_; // For server mode
    std::unique_ptr<tcp::socket> socket_;
    std::thread io_thread_;

    // Message queues
    struct PendingMessage {
        std::vector<uint8_t> data;
        uint32_t type;
    };

    std::queue<PendingMessage> send_queue_;
    std::queue<PendingMessage> recv_queue_;
    std::mutex send_mutex_;
    std::mutex recv_mutex_;
    std::condition_variable recv_cv_;

    // Buffers for current operations
    std::vector<uint8_t> send_buffer_;
    std::vector<uint8_t> recv_buffer_;
    std::vector<uint8_t> temp_recv_buffer_;

    // Connection state
    bool is_server_;
    bool connected_;
    std::atomic<bool> stopping_;

    // Compression support
    compression::CompressionManager compression_manager_;
    std::vector<uint8_t> compression_buffer_;

    // Async operations
    void start_accept();
    void handle_accept(const boost::system::error_code &error);
    void start_connect(const std::string &host, uint16_t port);
    void handle_connect(const boost::system::error_code &error);

    void start_read();
    void handle_read_header(const boost::system::error_code &error,
                            size_t bytes_transferred);
    void handle_read_body(const boost::system::error_code &error,
                          size_t bytes_transferred);

    void start_write();
    void handle_write(const boost::system::error_code &error,
                      size_t bytes_transferred);

    void run_io_service();

    // Helper to calculate checksum
    uint64_t calculate_checksum(const uint8_t *data, size_t size);
    
    // Get or create ring buffer for TCP operations
    mutable std::unique_ptr<RingBuffer> dummy_ring_buffer_;
};

// Factory function to create TCP channels from URI
std::unique_ptr<ChannelImpl> create_tcp_channel(
    const std::string &uri, size_t buffer_size, ChannelMode mode,
    ChannelType type,
    const compression::CompressionConfig &compression_config = {});

} // namespace detail
} // namespace psyne