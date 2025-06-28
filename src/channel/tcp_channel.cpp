#include "tcp_channel.hpp"
#include "../utils/xxhash64.h"
#include <regex>
#include <iostream>
#include <cstring>

namespace psyne {
namespace detail {

// Server constructor
TCPChannel::TCPChannel(const std::string& uri, size_t buffer_size,
                       ChannelMode mode, ChannelType type, uint16_t port,
                       const compression::CompressionConfig& compression_config)
    : ChannelImpl(uri, buffer_size, mode, type)
    , is_server_(true)
    , connected_(false)
    , stopping_(false)
    , compression_manager_(compression_config) {
    
    // Create acceptor for listening
    acceptor_ = std::make_unique<tcp::acceptor>(io_context_, tcp::endpoint(tcp::v4(), port));
    
    // Start IO thread
    io_thread_ = std::thread([this]() { run_io_service(); });
    
    // Start accepting connections
    start_accept();
}

// Client constructor
TCPChannel::TCPChannel(const std::string& uri, size_t buffer_size,
                       ChannelMode mode, ChannelType type,
                       const std::string& host, uint16_t port,
                       const compression::CompressionConfig& compression_config)
    : ChannelImpl(uri, buffer_size, mode, type)
    , is_server_(false)
    , connected_(false)
    , stopping_(false)
    , compression_manager_(compression_config) {
    
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
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5)) {
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

void* TCPChannel::reserve_space(size_t size) {
    // Allocate space in send buffer
    send_buffer_.resize(sizeof(TCPFrameHeader) + size);
    return send_buffer_.data() + sizeof(TCPFrameHeader);
}

void TCPChannel::commit_message(void* handle) {
    if (!connected_) {
        return;  // Silently drop if not connected
    }
    
    // Calculate payload size
    auto* payload_start = static_cast<uint8_t*>(handle);
    auto* buffer_start = send_buffer_.data();
    size_t payload_size = send_buffer_.size() - sizeof(TCPFrameHeader);
    
    // Try compression if enabled
    size_t final_payload_size = payload_size;
    uint8_t* final_payload = payload_start;
    bool compressed = false;
    
    if (compression_manager_.should_compress(payload_size)) {
        size_t compressed_size = compression_manager_.compress_message(
            payload_start, payload_size, compression_buffer_);
        
        if (compressed_size > 0) {
            final_payload_size = compressed_size;
            final_payload = compression_buffer_.data();
            compressed = true;
        }
    }
    
    // Prepare final message buffer
    std::vector<uint8_t> final_buffer(sizeof(TCPFrameHeader) + final_payload_size);
    
    // Fill in header
    auto* header = reinterpret_cast<TCPFrameHeader*>(final_buffer.data());
    header->length = sizeof(TCPFrameHeader) + final_payload_size;
    header->type = 1;  // TODO: Get from message
    header->checksum = calculate_checksum(final_payload, final_payload_size);
    header->original_size = compressed ? static_cast<uint32_t>(payload_size) : 0;
    header->compression_type = static_cast<uint8_t>(compression_manager_.config().type);
    header->compression_level = static_cast<uint8_t>(compression_manager_.config().level);
    header->flags = 0;
    
    // Copy payload
    std::memcpy(final_buffer.data() + sizeof(TCPFrameHeader), 
                final_payload, final_payload_size);
    
    // Add to send queue
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        PendingMessage msg;
        msg.data = std::move(final_buffer);
        msg.type = header->type;
        send_queue_.push(std::move(msg));
        send_buffer_.clear();
    }
    
    // Trigger async write if not already writing
    asio::post(io_context_, [this]() { start_write(); });
}

void* TCPChannel::receive_message(size_t& size, uint32_t& type) {
    std::unique_lock<std::mutex> lock(recv_mutex_);
    
    if (recv_queue_.empty()) {
        // Wait for messages with timeout
        recv_cv_.wait_for(lock, std::chrono::milliseconds(100),
                         [this] { return !recv_queue_.empty() || stopping_; });
    }
    
    if (recv_queue_.empty()) {
        return nullptr;
    }
    
    // Get message from queue
    temp_recv_buffer_ = std::move(recv_queue_.front().data);
    type = recv_queue_.front().type;
    recv_queue_.pop();
    
    // Extract header
    auto* header = reinterpret_cast<const TCPFrameHeader*>(temp_recv_buffer_.data());
    uint8_t* payload = temp_recv_buffer_.data() + sizeof(TCPFrameHeader);
    size_t payload_size = temp_recv_buffer_.size() - sizeof(TCPFrameHeader);
    
    // Check if decompression is needed
    if (header->original_size > 0) {
        // Message is compressed, decompress it
        compression_buffer_.resize(header->original_size);
        
        size_t decompressed_size = compression_manager_.decompress_message(
            payload, payload_size, compression_buffer_.data(), compression_buffer_.size());
        
        if (decompressed_size == 0 || decompressed_size != header->original_size) {
            // Decompression failed
            return nullptr;
        }
        
        // Replace temp buffer with decompressed data
        temp_recv_buffer_.resize(sizeof(TCPFrameHeader) + decompressed_size);
        std::memcpy(temp_recv_buffer_.data() + sizeof(TCPFrameHeader),
                    compression_buffer_.data(), decompressed_size);
        
        size = decompressed_size;
    } else {
        // Message is not compressed
        size = payload_size;
    }
    
    return temp_recv_buffer_.data() + sizeof(TCPFrameHeader);
}

void TCPChannel::release_message(void* handle) {
    // Nothing to do - message is in temp_recv_buffer_
}

void TCPChannel::start_accept() {
    if (!acceptor_ || stopping_) return;
    
    socket_ = std::make_unique<tcp::socket>(io_context_);
    acceptor_->async_accept(*socket_,
        [this](const boost::system::error_code& error) {
            if (!error) {
                connected_ = true;
                start_read();
            } else if (!stopping_) {
                // Retry accept
                start_accept();
            }
        });
}

void TCPChannel::start_connect(const std::string& host, uint16_t port) {
    tcp::resolver resolver(io_context_);
    auto endpoints = resolver.resolve(host, std::to_string(port));
    
    asio::async_connect(*socket_, endpoints,
        [this](const boost::system::error_code& error, const tcp::endpoint&) {
            handle_connect(error);
        });
}

void TCPChannel::handle_connect(const boost::system::error_code& error) {
    if (!error) {
        connected_ = true;
        start_read();
    } else {
        std::cerr << "TCP connect failed: " << error.message() << std::endl;
    }
}

void TCPChannel::start_read() {
    if (!socket_ || !connected_ || stopping_) return;
    
    // Read header first
    recv_buffer_.resize(sizeof(TCPFrameHeader));
    asio::async_read(*socket_,
        asio::buffer(recv_buffer_),
        [this](const boost::system::error_code& error, size_t bytes_transferred) {
            handle_read_header(error, bytes_transferred);
        });
}

void TCPChannel::handle_read_header(const boost::system::error_code& error, 
                                   size_t bytes_transferred) {
    if (error || stopping_) {
        if (!stopping_) {
            connected_ = false;
        }
        return;
    }
    
    // Parse header
    auto* header = reinterpret_cast<TCPFrameHeader*>(recv_buffer_.data());
    size_t payload_size = header->length - sizeof(TCPFrameHeader);
    
    // Resize buffer to include payload
    recv_buffer_.resize(header->length);
    
    // Read payload
    asio::async_read(*socket_,
        asio::buffer(recv_buffer_.data() + sizeof(TCPFrameHeader), payload_size),
        [this](const boost::system::error_code& error, size_t bytes_transferred) {
            handle_read_body(error, bytes_transferred);
        });
}

void TCPChannel::handle_read_body(const boost::system::error_code& error,
                                 size_t bytes_transferred) {
    if (error || stopping_) {
        if (!stopping_) {
            connected_ = false;
        }
        return;
    }
    
    // Verify checksum
    auto* header = reinterpret_cast<TCPFrameHeader*>(recv_buffer_.data());
    auto* payload = recv_buffer_.data() + sizeof(TCPFrameHeader);
    size_t payload_size = header->length - sizeof(TCPFrameHeader);
    
    uint64_t checksum = calculate_checksum(payload, payload_size);
    if (checksum != header->checksum) {
        std::cerr << "TCP checksum mismatch!" << std::endl;
        start_read();  // Continue reading
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
    auto& msg = send_queue_.front();
    
    asio::async_write(*socket_,
        asio::buffer(msg.data),
        [this](const boost::system::error_code& error, size_t bytes_transferred) {
            handle_write(error, bytes_transferred);
        });
}

void TCPChannel::handle_write(const boost::system::error_code& error,
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
            break;  // Normal exit
        } catch (const std::exception& e) {
            std::cerr << "TCP IO error: " << e.what() << std::endl;
            if (!stopping_) {
                io_context_.restart();
            }
        }
    }
}

uint64_t TCPChannel::calculate_checksum(const uint8_t* data, size_t size) {
    return XXHash64::hash(data, size, 0);
}

// Factory function to create TCP channels from URI
std::unique_ptr<ChannelImpl> create_tcp_channel(
    const std::string& uri, size_t buffer_size,
    ChannelMode mode, ChannelType type,
    const compression::CompressionConfig& compression_config) {
    
    // Parse URI: tcp://host:port or tcp://:port (for server)
    std::regex uri_regex("^tcp://([^:]*):([0-9]+)$");
    std::smatch match;
    
    if (!std::regex_match(uri, match, uri_regex)) {
        throw std::invalid_argument("Invalid TCP URI format. Use tcp://host:port or tcp://:port");
    }
    
    std::string host = match[1];
    uint16_t port = std::stoi(match[2]);
    
    if (host.empty()) {
        // Server mode - listen on all interfaces
        return std::make_unique<TCPChannel>(uri, buffer_size, mode, type, port, compression_config);
    } else {
        // Client mode - connect to host:port
        return std::make_unique<TCPChannel>(uri, buffer_size, mode, type, host, port, compression_config);
    }
}

} // namespace detail
} // namespace psyne