#pragma once

#include "channel_impl.hpp"
#include <boost/asio.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace psyne {
namespace detail {

namespace asio = boost::asio;
using tcp = asio::ip::tcp;

// TCP message frame header
struct TCPFrameHeader {
    uint32_t length;      // Total message length (including this header)
    uint32_t checksum;    // xxHash32 checksum of the payload
    uint32_t type;        // Message type
    uint32_t reserved;    // Reserved for future use
};

// TCP Channel implementation
class TCPChannel : public ChannelImpl {
public:
    // Constructor for server (listening) mode
    TCPChannel(const std::string& uri, size_t buffer_size,
               ChannelMode mode, ChannelType type, uint16_t port);
    
    // Constructor for client (connecting) mode  
    TCPChannel(const std::string& uri, size_t buffer_size,
               ChannelMode mode, ChannelType type,
               const std::string& host, uint16_t port);
    
    ~TCPChannel();
    
    void* reserve_space(size_t size) override;
    void commit_message(void* handle) override;
    void* receive_message(size_t& size, uint32_t& type) override;
    void release_message(void* handle) override;
    
private:
    // Network components
    asio::io_context io_context_;
    std::unique_ptr<tcp::acceptor> acceptor_;  // For server mode
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
    
    // Async operations
    void start_accept();
    void handle_accept(const boost::system::error_code& error);
    void start_connect(const std::string& host, uint16_t port);
    void handle_connect(const boost::system::error_code& error);
    
    void start_read();
    void handle_read_header(const boost::system::error_code& error, size_t bytes_transferred);
    void handle_read_body(const boost::system::error_code& error, size_t bytes_transferred);
    
    void start_write();
    void handle_write(const boost::system::error_code& error, size_t bytes_transferred);
    
    void run_io_service();
    
    // Helper to calculate checksum
    uint32_t calculate_checksum(const uint8_t* data, size_t size);
};

// Factory function to create TCP channels from URI
std::unique_ptr<ChannelImpl> create_tcp_channel(
    const std::string& uri, size_t buffer_size,
    ChannelMode mode, ChannelType type);

} // namespace detail
} // namespace psyne