#pragma once

/**
 * @file tcp_channel.hpp
 * @brief TCP transport implementation for Psyne channels
 * @author Psyne Contributors
 * @date 2025
 * 
 * This file implements TCP socket-based transport following the zero-copy
 * principles defined in CORE_DESIGN.md. The TCP channel provides reliable,
 * ordered delivery of messages over network connections.
 * 
 * Key Design Principles:
 * - Zero-copy message creation through direct ring buffer access
 * - Network transport streams directly from ring buffer
 * - Receiver writes network data directly to matching buffer
 * - Message framing for reliable stream-based communication
 */

#include "../channel_impl.hpp"
#include "../../utils/logger.hpp"
#include <boost/asio.hpp>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace psyne {
namespace detail {

/**
 * @class TCPChannel
 * @brief TCP transport implementation following zero-copy principles
 * 
 * This implementation provides TCP socket communication while maintaining
 * the zero-copy message interface. Messages are created as direct views
 * into the ring buffer, and network transport streams data without copying.
 */
class TCPChannel : public ChannelImpl {
public:
    /**
     * @brief Construct TCP channel as server (listening mode)
     * @param uri Channel URI (tcp://host:port/path)
     * @param buffer_size Size of ring buffer in bytes
     * @param mode Channel synchronization mode
     * @param type Channel type (single/multi-type)
     * @param port TCP port to listen on
     */
    TCPChannel(const std::string& uri, size_t buffer_size, ChannelMode mode,
               ChannelType type, uint16_t port);
    
    /**
     * @brief Construct TCP channel as client (connecting mode)
     * @param uri Channel URI (tcp://host:port/path)
     * @param buffer_size Size of ring buffer in bytes
     * @param mode Channel synchronization mode
     * @param type Channel type (single/multi-type)
     * @param host Remote host to connect to
     * @param port Remote port to connect to
     */
    TCPChannel(const std::string& uri, size_t buffer_size, ChannelMode mode,
               ChannelType type, const std::string& host, uint16_t port);
    
    /**
     * @brief Destructor - ensures clean shutdown
     */
    ~TCPChannel() override;
    
    // Zero-copy interface implementation
    uint32_t reserve_write_slot(size_t size) override;
    void notify_message_ready(uint32_t offset, size_t size) override;
    RingBuffer& get_ring_buffer() override;
    const RingBuffer& get_ring_buffer() const override;
    void advance_read_pointer(size_t size) override;
    
    // Legacy interface (deprecated)
    void* reserve_space(size_t size) override;
    void commit_message(void* handle) override;
    void* receive_message(size_t& size, uint32_t& type) override;
    void release_message(void* handle) override;
    
    /**
     * @brief Stop the TCP channel and close connections
     */
    void stop();
    
    /**
     * @brief Check if channel is connected
     * @return true if connected, false otherwise
     */
    bool is_connected() const;

private:
    /**
     * @brief Message header for TCP framing
     * 
     * TCP is a stream protocol, so we need framing to delimit messages.
     * This header provides the minimum metadata needed.
     */
    struct MessageHeader {
        uint32_t size;      ///< Message size in bytes (network byte order)
        uint32_t checksum;  ///< Simple checksum for integrity (network byte order)
    };
    
    static constexpr size_t HEADER_SIZE = sizeof(MessageHeader);
    static constexpr uint32_t MAX_MESSAGE_SIZE = 100 * 1024 * 1024; // 100MB limit
    
    // Network configuration
    bool is_server_;
    std::string host_;
    uint16_t port_;
    
    // Boost.Asio networking
    std::unique_ptr<boost::asio::io_context> io_context_;
    std::unique_ptr<boost::asio::ip::tcp::acceptor> acceptor_; // Server mode
    std::unique_ptr<boost::asio::ip::tcp::socket> socket_;     // Client mode or accepted connection
    std::thread network_thread_;
    
    // Ring buffer (template specialization based on mode)
    std::unique_ptr<RingBuffer> ring_buffer_;
    
    // Connection state
    std::atomic<bool> connected_;
    std::atomic<bool> network_running_;
    mutable std::mutex connection_mutex_;
    std::condition_variable connection_cv_;
    
    // Message handling
    std::vector<uint8_t> receive_buffer_;
    size_t bytes_needed_;
    bool header_received_;
    
    /**
     * @brief Initialize the ring buffer based on channel mode
     */
    void initialize_ring_buffer(size_t buffer_size, ChannelMode mode);
    
    /**
     * @brief Start server mode - listen for connections
     */
    void start_server();
    
    /**
     * @brief Start client mode - connect to server
     */
    void start_client();
    
    /**
     * @brief Network event loop
     */
    void network_loop();
    
    /**
     * @brief Handle new connection (server mode)
     */
    void handle_accept(const boost::system::error_code& error);
    
    /**
     * @brief Handle connection established (client mode)
     */
    void handle_connect(const boost::system::error_code& error);
    
    /**
     * @brief Start receiving data
     */
    void start_receive();
    
    /**
     * @brief Handle received data
     */
    void handle_receive(const boost::system::error_code& error, size_t bytes_transferred);
    
    /**
     * @brief Send data over TCP (zero-copy from ring buffer)
     * @param data Pointer to data in ring buffer
     * @param size Size of data
     */
    void send_data(const void* data, size_t size);
    
    /**
     * @brief Calculate simple checksum for message integrity
     * @param data Pointer to data
     * @param size Size of data
     * @return Checksum value
     */
    uint32_t calculate_checksum(const void* data, size_t size) const;
    
    /**
     * @brief Process complete message received from network
     * @param data Message data
     * @param size Message size
     */
    void process_received_message(const void* data, size_t size);
};

/**
 * @brief Factory function to create TCP channel
 * @param uri Channel URI
 * @param buffer_size Ring buffer size
 * @param mode Channel mode
 * @param type Channel type
 * @return Unique pointer to TCP channel implementation
 */
std::unique_ptr<ChannelImpl> create_tcp_channel(
    const std::string& uri,
    size_t buffer_size,
    ChannelMode mode,
    ChannelType type
);

} // namespace detail
} // namespace psyne