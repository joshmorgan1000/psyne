#include "tcp_channel.hpp"
#include "../utils/checksum.hpp"
#include <cstring>
#include <iostream>
#include <regex>

namespace psyne {
namespace detail {

// Server constructor
TCPChannel::TCPChannel(const std::string &uri, size_t buffer_size,
                       ChannelMode mode, ChannelType type, uint16_t port,
                       const compression::CompressionConfig &compression_config)
    : ChannelImpl(uri, buffer_size, mode, type), is_server_(true),
      connected_(false), stopping_(false),
      compression_manager_(compression_config) {
    // Create acceptor for listening
    acceptor_ = std::make_unique<tcp::acceptor>(io_context_,
                                                tcp::endpoint(tcp::v4(), port));

    // Start IO thread
    io_thread_ = std::thread([this]() { run_io_service(); });

    // Start accepting connections
    start_accept();
}

// Client constructor
TCPChannel::TCPChannel(const std::string &uri, size_t buffer_size,
                       ChannelMode mode, ChannelType type,
                       const std::string &host, uint16_t port,
                       const compression::CompressionConfig &compression_config)
    : ChannelImpl(uri, buffer_size, mode, type), is_server_(false),
      connected_(false), stopping_(false),
      compression_manager_(compression_config) {
    // Create socket
    socket_ = std::make_unique<tcp::socket>(io_context_);

    // Start IO thread
    io_thread_ = std::thread([this]() { run_io_service(); });

    // Start connecting
    start_connect(host, port);

    // Wait for connection (with timeout)
    auto start = std::chrono::steady_clock::now();
    while (!connected_ && !stopping_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (std::chrono::steady_clock::now() - start >
            std::chrono::seconds(5)) {
            throw std::runtime_error("TCP connection timeout");
        }
    }
}

TCPChannel::~TCPChannel() {
    stopping_ = true;
    io_context_.stop();
    if (io_thread_.joinable()) {
        io_thread_.join();
    }
}

uint32_t TCPChannel::reserve_write_slot(size_t size) noexcept {
    if (!connected_) {
        return BUFFER_FULL;
    }
    
    // For TCP, reserve space in ring buffer and return offset
    auto& ring_buffer = get_ring_buffer();
    uint32_t offset = ring_buffer.reserve_write_space(size);
    return offset;
}

void TCPChannel::notify_message_ready(uint32_t offset, size_t size) noexcept {
    if (!connected_) {
        return;
    }
    
    // Message is already in ring buffer at offset
    // Stream directly from ring buffer to TCP socket
    auto& ring_buffer = get_ring_buffer();
    auto data_span = ring_buffer.get_read_span(offset, size);
    
    // Prepare TCP frame header
    TCPFrameHeader header{};
    header.length = sizeof(TCPFrameHeader) + size;
    header.type = 1;
    header.checksum = calculate_checksum(data_span.data(), data_span.size());
    header.original_size = 0; // No compression in zero-copy path
    header.compression_type = 0;
    header.compression_level = 0;
    header.flags = 0;
    
    // Queue zero-copy send operation
    asio::post(io_context_, [this, header, offset, size]() {
        start_zero_copy_write(header, offset, size);
    });
}

std::span<uint8_t> TCPChannel::get_write_span(size_t size) noexcept {
    uint32_t offset = reserve_write_slot(size);
    if (offset == BUFFER_FULL) {
        return {};
    }
    
    auto& ring_buffer = get_ring_buffer();
    return ring_buffer.get_write_span(offset, size);
}

std::span<const uint8_t> TCPChannel::buffer_span() const noexcept {
    auto& ring_buffer = get_ring_buffer();
    return ring_buffer.available_read_span();
}

void TCPChannel::advance_read_pointer(size_t size) noexcept {
    auto& ring_buffer = get_ring_buffer();
    ring_buffer.advance_read_pointer(size);
}

void TCPChannel::start_zero_copy_write(const TCPFrameHeader& header, uint32_t offset, size_t size) {
    if (!connected_ || stopping_) {
        return;
    }
    
    // Get data directly from ring buffer
    auto& ring_buffer = get_ring_buffer();
    auto data_span = ring_buffer.get_read_span(offset, size);
    
    // Create gather buffer for efficient sending (header + data)
    std::array<asio::const_buffer, 2> buffers = {
        asio::buffer(&header, sizeof(header)),
        asio::buffer(data_span.data(), data_span.size())
    };
    
    asio::async_write(*socket_, buffers,
        [this, offset, size](const boost::system::error_code& error, size_t bytes_transferred) {
            if (!error) {
                // Successfully sent, advance ring buffer read position
                auto& ring_buffer = get_ring_buffer();
                ring_buffer.advance_read_pointer(size);
            } else if (!stopping_) {
                connected_ = false;
            }
        });
}

// Legacy deprecated methods (kept for compatibility)
void* TCPChannel::reserve_space(size_t size) {
    uint32_t offset = reserve_write_slot(size);
    if (offset == BUFFER_FULL) {
        return nullptr;
    }
    auto& ring_buffer = get_ring_buffer();
    auto span = ring_buffer.get_write_span(offset, size);
    return span.data();
}

void TCPChannel::commit_message(void* handle) {
    // This is a no-op in zero-copy design
    // Message was already written directly to ring buffer
}

void* TCPChannel::receive_message(size_t& size, uint32_t& type) {
    auto span = buffer_span();
    if (span.empty()) {
        return nullptr;
    }
    size = span.size();
    type = 1; // Default type
    return const_cast<uint8_t*>(span.data());
}

void TCPChannel::release_message(void* handle) {
    // In zero-copy design, caller must call advance_read_pointer()
}

void TCPChannel::start_accept() {
    if (!acceptor_ || stopping_)
        return;

    socket_ = std::make_unique<tcp::socket>(io_context_);
    acceptor_->async_accept(*socket_,
                            [this](const boost::system::error_code &error) {
                                if (!error) {
                                    connected_ = true;
                                    start_read();
                                } else if (!stopping_) {
                                    // Retry accept
                                    start_accept();
                                }
                            });
}

void TCPChannel::start_connect(const std::string &host, uint16_t port) {
    tcp::resolver resolver(io_context_);
    auto endpoints = resolver.resolve(host, std::to_string(port));

    asio::async_connect(
        *socket_, endpoints,
        [this](const boost::system::error_code &error, const tcp::endpoint &) {
            handle_connect(error);
        });
}

void TCPChannel::handle_connect(const boost::system::error_code &error) {
    if (!error) {
        connected_ = true;
        start_read();
    } else {
        std::cerr << "TCP connect failed: " << error.message() << std::endl;
    }
}

void TCPChannel::start_read() {
    if (!socket_ || !connected_ || stopping_)
        return;

    // Read header first
    recv_buffer_.resize(sizeof(TCPFrameHeader));
    asio::async_read(*socket_, asio::buffer(recv_buffer_),
                     [this](const boost::system::error_code &error,
                            size_t bytes_transferred) {
                         handle_read_header(error, bytes_transferred);
                     });
}

void TCPChannel::handle_read_header(const boost::system::error_code &error,
                                    size_t bytes_transferred) {
    if (error || stopping_) {
        if (!stopping_) {
            connected_ = false;
        }
        return;
    }

    // Parse header
    auto *header = reinterpret_cast<TCPFrameHeader *>(recv_buffer_.data());
    size_t payload_size = header->length - sizeof(TCPFrameHeader);

    // Resize buffer to include payload
    recv_buffer_.resize(header->length);

    // Read payload
    asio::async_read(*socket_,
                     asio::buffer(recv_buffer_.data() + sizeof(TCPFrameHeader),
                                  payload_size),
                     [this](const boost::system::error_code &error,
                            size_t bytes_transferred) {
                         handle_read_body(error, bytes_transferred);
                     });
}

void TCPChannel::handle_read_body(const boost::system::error_code &error,
                                  size_t bytes_transferred) {
    if (error || stopping_) {
        if (!stopping_) {
            connected_ = false;
        }
        return;
    }

    // Verify checksum
    auto *header = reinterpret_cast<TCPFrameHeader *>(recv_buffer_.data());
    auto *payload = recv_buffer_.data() + sizeof(TCPFrameHeader);
    size_t payload_size = header->length - sizeof(TCPFrameHeader);

    uint64_t checksum = calculate_checksum(payload, payload_size);
    if (checksum != header->checksum) {
        std::cerr << "TCP checksum mismatch!" << std::endl;
        start_read(); // Continue reading
        return;
    }

    // Add to receive queue
    {
        std::lock_guard<std::mutex> lock(recv_mutex_);
        PendingMessage msg;
        msg.data = std::move(recv_buffer_);
        msg.type = header->type;
        recv_queue_.push(std::move(msg));
    }
    recv_cv_.notify_one();

    // Continue reading
    start_read();
}

void TCPChannel::start_write() {
    std::lock_guard<std::mutex> lock(send_mutex_);

    if (send_queue_.empty() || !connected_ || stopping_) {
        return;
    }

    // Get next message
    auto &msg = send_queue_.front();

    asio::async_write(*socket_, asio::buffer(msg.data),
                      [this](const boost::system::error_code &error,
                             size_t bytes_transferred) {
                          handle_write(error, bytes_transferred);
                      });
}

void TCPChannel::handle_write(const boost::system::error_code &error,
                              size_t bytes_transferred) {
    if (error || stopping_) {
        if (!stopping_) {
            connected_ = false;
        }
        return;
    }

    // Remove sent message
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        send_queue_.pop();
    }

    // Send next message if any
    start_write();
}

void TCPChannel::run_io_service() {
    while (!stopping_) {
        try {
            io_context_.run();
            break; // Normal exit
        } catch (const std::exception &e) {
            std::cerr << "TCP IO error: " << e.what() << std::endl;
            if (!stopping_) {
                io_context_.restart();
            }
        }
    }
}

uint64_t TCPChannel::calculate_checksum(const uint8_t *data, size_t size) {
    return utils::tcp::calculate_checksum(data, size);
}

// Factory function to create TCP channels from URI
std::unique_ptr<ChannelImpl>
create_tcp_channel(const std::string &uri, size_t buffer_size, ChannelMode mode,
                   ChannelType type,
                   const compression::CompressionConfig &compression_config) {
    // Parse URI: tcp://host:port or tcp://:port (for server)
    std::regex uri_regex("^tcp://([^:]*):([0-9]+)$");
    std::smatch match;

    if (!std::regex_match(uri, match, uri_regex)) {
        throw std::invalid_argument(
            "Invalid TCP URI format. Use tcp://host:port or tcp://:port");
    }

    std::string host = match[1];

    // Validate and parse port number
    std::string port_str = match[2];
    int port_int;
    try {
        port_int = std::stoi(port_str);
    } catch (const std::exception &e) {
        throw std::invalid_argument("Invalid port number: " + port_str);
    }

    if (port_int < 0 || port_int > 65535) {
        throw std::invalid_argument(
            "Port number must be between 0 and 65535, got: " +
            std::to_string(port_int));
    }

    uint16_t port = static_cast<uint16_t>(port_int);

    if (host.empty()) {
        // Server mode - listen on all interfaces
        return std::make_unique<TCPChannel>(uri, buffer_size, mode, type, port,
                                            compression_config);
    } else {
        // Client mode - connect to host:port
        return std::make_unique<TCPChannel>(uri, buffer_size, mode, type, host,
                                            port, compression_config);
    }
}

} // namespace detail
} // namespace psyne