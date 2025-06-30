#pragma once

/**
 * @file quic_channel.hpp
 * @brief QUIC channel implementation for high-performance network transport
 *
 * Provides modern QUIC protocol support with:
 * - Built-in TLS 1.3 encryption
 * - Stream multiplexing
 * - 0-RTT connection resumption
 * - Connection migration
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include "channel_impl.hpp"
#include "../transport/quic.hpp"
#include <boost/asio.hpp>
#include <memory>
#include <thread>

namespace psyne {

// Forward declarations
class RingBuffer;

namespace detail {

namespace asio = boost::asio;

/**
 * @brief QUIC channel implementation
 * 
 * Implements zero-copy messaging over QUIC protocol.
 * Uses ring buffers for efficient data transfer.
 */
class QUICChannel : public ChannelImpl {
public:
    QUICChannel(const std::string& uri, size_t buffer_size,
                ChannelMode mode = ChannelMode::SPSC,
                ChannelType type = ChannelType::MultiType,
                bool is_server = false);
    
    ~QUICChannel() override;
    
    // Zero-copy interface (simplified for now)
    uint32_t reserve_write_slot(size_t size) noexcept override;
    void notify_message_ready(uint32_t offset, size_t size) noexcept override;
    RingBuffer& get_ring_buffer() noexcept override;
    const RingBuffer& get_ring_buffer() const noexcept override;
    void advance_read_pointer(size_t size) noexcept override;
    std::span<uint8_t> get_write_span(size_t size) noexcept;
    std::span<const uint8_t> buffer_span() const noexcept;
    
    // Legacy interface
    void* reserve_space(size_t size) override;
    void commit_message(void* handle) override;
    void* receive_message(size_t& size, uint32_t& type) override;
    void release_message(void* handle) override;
    
    // Connection management
    bool is_connected() const { return connected_.load(); }
    
private:
    void run_client(const std::string& host, uint16_t port);
    void run_server(uint16_t port);
    void handle_connection();
    void send_loop();
    void receive_loop();
    
    // Message queues (similar to TCP channel)
    struct PendingMessage {
        std::vector<uint8_t> data;
        uint32_t type;
    };
    
    std::queue<PendingMessage> send_queue_;
    std::queue<PendingMessage> recv_queue_;
    mutable std::mutex recv_mutex_;
    std::condition_variable recv_cv_;
    
    // QUIC connection state
    std::shared_ptr<transport::QUICConnection> connection_;
    std::unique_ptr<transport::QUICServer> server_;
    
    // Network components
    asio::io_context io_context_;
    std::thread io_thread_;
    
    // Connection state
    std::atomic<bool> connected_{false};
    std::atomic<bool> stopping_{false};
    bool is_server_;
    std::string host_;
    uint16_t port_;
    
    // Synchronization
    std::mutex send_mutex_;
    std::condition_variable send_cv_;
    std::queue<std::pair<uint32_t, size_t>> pending_sends_;
    
    // Metrics
    std::atomic<uint64_t> messages_sent_{0};
    std::atomic<uint64_t> messages_received_{0};
    std::atomic<uint64_t> bytes_sent_{0};
    std::atomic<uint64_t> bytes_received_{0};
    uint32_t type_ = 0; // Default message type
};

/**
 * @brief Create a QUIC channel
 */
std::unique_ptr<ChannelImpl> 
create_quic_channel(const std::string& uri, size_t buffer_size,
                    ChannelMode mode, ChannelType type,
                    const compression::CompressionConfig& compression_config);

} // namespace detail
} // namespace psyne