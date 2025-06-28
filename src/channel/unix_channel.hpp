#pragma once

#include "channel_impl.hpp"
#include <boost/asio.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <filesystem>

namespace psyne {
namespace detail {

namespace asio = boost::asio;
using unix_socket = asio::local::stream_protocol;

// Unix domain socket message frame header (same as TCP for consistency)
struct UnixFrameHeader {
    uint32_t length;      // Total message length (including this header)
    uint32_t checksum;    // xxHash32 checksum of the payload
    uint32_t type;        // Message type
    uint32_t reserved;    // Reserved for future use
};

// Unix Domain Socket Channel implementation
class UnixChannel : public ChannelImpl {
public:
    // Constructor for server (listening) mode
    UnixChannel(const std::string& uri, size_t buffer_size,
                ChannelMode mode, ChannelType type, const std::string& socket_path);
    
    // Constructor for client (connecting) mode  
    UnixChannel(const std::string& uri, size_t buffer_size,
                ChannelMode mode, ChannelType type,
                const std::string& socket_path, bool is_client);
    
    ~UnixChannel();
    
    void* reserve_space(size_t size) override;
    void commit_message(void* handle) override;
    void* receive_message(size_t& size, uint32_t& type) override;
    void release_message(void* handle) override;
    
private:
    // Network components
    asio::io_context io_context_;
    std::unique_ptr<unix_socket::acceptor> acceptor_;  // For server mode
    std::unique_ptr<unix_socket::socket> socket_;
    std::thread io_thread_;
    
    // Message queues
    std::queue<std::vector<uint8_t>> outgoing_messages_;
    std::queue<std::vector<uint8_t>> incoming_messages_;
    
    // Synchronization
    std::mutex outgoing_mutex_;
    std::mutex incoming_mutex_;
    std::condition_variable outgoing_cv_;
    std::condition_variable incoming_cv_;
    
    // Connection state
    bool is_server_;
    std::atomic<bool> connected_;
    std::atomic<bool> stopping_;
    std::string socket_path_;
    
    // Async operations
    void run_io_service();
    void start_accept();
    void start_connect();
    void handle_accept(std::shared_ptr<unix_socket::socket> client,
                      const boost::system::error_code& error);
    void handle_connect(const boost::system::error_code& error);
    
    // Message processing
    void start_receive(std::shared_ptr<unix_socket::socket> sock);
    void handle_receive_header(std::shared_ptr<unix_socket::socket> sock,
                              std::shared_ptr<UnixFrameHeader> header,
                              const boost::system::error_code& error,
                              size_t bytes_transferred);
    void handle_receive_payload(std::shared_ptr<unix_socket::socket> sock,
                               std::shared_ptr<std::vector<uint8_t>> buffer,
                               const UnixFrameHeader& header,
                               const boost::system::error_code& error,
                               size_t bytes_transferred);
    
    void start_send(std::shared_ptr<unix_socket::socket> sock);
    void handle_send(std::shared_ptr<unix_socket::socket> sock,
                    std::shared_ptr<std::vector<uint8_t>> buffer,
                    const boost::system::error_code& error,
                    size_t bytes_transferred);
    
    // Utility
    uint32_t calculate_checksum(const void* data, size_t size);
    void cleanup_socket_file();
    
    // Current active socket for communication
    std::shared_ptr<unix_socket::socket> active_socket_;
    std::mutex socket_mutex_;
};

} // namespace detail
} // namespace psyne