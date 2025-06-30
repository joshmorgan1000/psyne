#pragma once

/**
 * @file tcp_channel.hpp
 * @brief TCP/IP network channel implementation
 * @author Psyne Contributors
 * @date 2025
 * 
 * This file implements TCP channels for network communication.
 * TCP channels support both server (listening) and client (connecting) modes,
 * with optional compression and zero-copy scatter-gather I/O.
 */

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

/**
 * @struct TCPFrameHeader
 * @brief Header structure for TCP message frames
 * 
 * This 16-byte aligned header prefixes each message sent over TCP.
 * It contains metadata for message framing, type identification,
 * integrity checking, and compression information.
 */
struct TCPFrameHeader {
    uint32_t length;   ///< Total message length (including this header)
    uint32_t type;     ///< Message type identifier
    uint64_t checksum; ///< xxHash64 checksum of the payload
    uint32_t
        original_size; ///< Original size before compression (0 if not compressed)
    uint8_t compression_type;  ///< Compression algorithm used
    uint8_t compression_level; ///< Compression level used
    uint16_t flags;            ///< Additional flags for future use
};

/**
 * @class TCPChannel
 * @brief TCP/IP channel implementation
 * 
 * This class implements network communication over TCP sockets.
 * It supports both server (accepting connections) and client
 * (initiating connections) modes, with optional compression
 * and zero-copy scatter-gather I/O for maximum performance.
 * 
 * The implementation uses Boost.Asio for cross-platform networking
 * and maintains separate send/receive queues for thread safety.
 */
class TCPChannel : public ChannelImpl {
public:
    /**
     * @brief Constructor for server (listening) mode
     * @param uri Channel URI identifier
     * @param buffer_size Size of the internal buffer in bytes
     * @param mode Synchronization mode (SPSC, MPSC, etc.)
     * @param type Channel type identifier
     * @param port Port number to listen on
     * @param compression_config Optional compression configuration
     */
    TCPChannel(const std::string &uri, size_t buffer_size, ChannelMode mode,
               ChannelType type, uint16_t port,
               const compression::CompressionConfig &compression_config = {});

    /**
     * @brief Constructor for client (connecting) mode
     * @param uri Channel URI identifier
     * @param buffer_size Size of the internal buffer in bytes
     * @param mode Synchronization mode (SPSC, MPSC, etc.)
     * @param type Channel type identifier
     * @param host Hostname or IP address to connect to
     * @param port Port number to connect to
     * @param compression_config Optional compression configuration
     */
    TCPChannel(const std::string &uri, size_t buffer_size, ChannelMode mode,
               ChannelType type, const std::string &host, uint16_t port,
               const compression::CompressionConfig &compression_config = {});

    /**
     * @brief Destructor - closes socket and stops I/O thread
     */
    ~TCPChannel();

    // Zero-copy interface
    /**
     * @brief Reserve a write slot in the ring buffer
     * @param size Size of the message to write
     * @return Offset in the buffer, or BUFFER_FULL if no space
     */
    uint32_t reserve_write_slot(size_t size) noexcept override;
    
    /**
     * @brief Notify that a message is ready to be sent
     * @param offset Offset of the message in the buffer
     * @param size Size of the message
     */
    void notify_message_ready(uint32_t offset, size_t size) noexcept override;
    
    /**
     * @brief Get reference to the underlying ring buffer
     * @return Reference to the ring buffer
     */
    RingBuffer& get_ring_buffer() noexcept override;
    
    /**
     * @brief Get const reference to the underlying ring buffer
     * @return Const reference to the ring buffer
     */
    const RingBuffer& get_ring_buffer() const noexcept override;
    
    /**
     * @brief Advance the read pointer after consuming a message
     * @param size Size of the consumed message
     */
    void advance_read_pointer(size_t size) noexcept override;
    
    // TCP-specific zero-copy methods
    /**
     * @brief Get a writable span for zero-copy writes
     * @param size Size of the span needed
     * @return Writable span of the requested size
     */
    std::span<uint8_t> get_write_span(size_t size) noexcept;
    
    /**
     * @brief Get a const span view of the entire buffer
     * @return Const span of the buffer
     */
    std::span<const uint8_t> buffer_span() const noexcept;

private:
    /**
     * @brief Zero-copy TCP write using scatter-gather I/O
     * @param header TCP frame header to send
     * @param offset Offset of the message data in the ring buffer
     * @param size Size of the message data
     * 
     * Uses Boost.Asio's scatter-gather I/O to send the header and
     * message data in a single system call without copying.
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