#pragma once

#include "channel_impl.hpp"
#include <atomic>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/websocket.hpp>
#include <condition_variable>
#include <memory>
#include <queue>
#include <thread>
#include <span>

namespace psyne {
namespace detail {

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;

/**
 * @brief WebSocket channel implementation for browser/web application
 * integration
 *
 * Provides zero-copy messaging over WebSocket protocol, enabling web clients
 * to communicate with Psyne-based services.
 */
class WebSocketChannel : public ChannelImpl {
public:
    /**
     * @brief Construct WebSocket channel
     * @param uri WebSocket URI (ws://host:port or wss://host:port)
     * @param buffer_size Size of internal buffers
     * @param is_server Whether this is server (true) or client (false)
     */
    WebSocketChannel(const std::string &uri, size_t buffer_size,
                     bool is_server);
    ~WebSocketChannel() override;

    // Zero-copy interface
    uint32_t reserve_write_slot(size_t size) noexcept override;
    void notify_message_ready(uint32_t offset, size_t size) noexcept override;
    std::span<uint8_t> get_write_span(size_t size) noexcept override;
    std::span<const uint8_t> buffer_span() const noexcept override;
    void advance_read_pointer(size_t size) noexcept override;
    
    // Legacy interface (deprecated)
    [[deprecated("Use reserve_write_slot() instead")]]
    void *reserve_space(size_t size) override;
    [[deprecated("Data is committed when written")]]
    void commit_message(void *handle) override;
    void *receive_message(size_t &size, uint32_t &type) override;
    void release_message(void *message) override;

private:
    /**
     * @brief Zero-copy WebSocket frame sending
     */
    void send_websocket_frame(std::span<const uint8_t> data);

public:
    debug::ChannelMetrics get_metrics() const override {
        debug::ChannelMetrics metrics;
        metrics.messages_sent = messages_sent_.load();
        metrics.bytes_sent = bytes_sent_.load();
        metrics.messages_received = messages_received_.load();
        metrics.bytes_received = bytes_received_.load();
        metrics.send_blocks = send_blocks_.load();
        metrics.receive_blocks = receive_blocks_.load();
        return metrics;
    }

private:
    void run_client(const std::string &host, uint16_t port);
    void run_server(uint16_t port);
    void handle_connection(tcp::socket socket);
    void process_messages();
    void send_loop();
    void receive_loop();

    struct PendingMessage {
        std::vector<uint8_t> data;
        uint32_t type;
    };

    // Network components
    net::io_context io_context_;
    std::unique_ptr<websocket::stream<tcp::socket>> ws_stream_;
    std::unique_ptr<tcp::acceptor> acceptor_;
    std::thread io_thread_;

    // Message queues
    std::queue<PendingMessage> send_queue_;
    std::queue<std::vector<uint8_t>> receive_queue_;
    std::mutex send_mutex_;
    std::mutex receive_mutex_;
    std::condition_variable send_cv_;
    std::condition_variable receive_cv_;

    // Buffers
    std::vector<uint8_t> send_buffer_;
    std::vector<uint8_t> receive_buffer_;
    size_t buffer_size_;

    // State
    std::atomic<bool> stopped_{false};
    std::atomic<bool> connected_{false};
    bool is_server_;

    // Metrics
    std::atomic<uint64_t> messages_sent_{0};
    std::atomic<uint64_t> bytes_sent_{0};
    std::atomic<uint64_t> messages_received_{0};
    std::atomic<uint64_t> bytes_received_{0};
    std::atomic<uint64_t> send_blocks_{0};
    std::atomic<uint64_t> receive_blocks_{0};

    // Protocol constants
    static constexpr size_t HEADER_SIZE =
        sizeof(uint32_t) + sizeof(uint32_t); // size + type
};

} // namespace detail
} // namespace psyne