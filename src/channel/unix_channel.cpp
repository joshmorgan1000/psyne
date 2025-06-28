#include "unix_channel.hpp"
#include "../utils/xxhash64.h"
#include <iostream>
#include <cstring>

namespace psyne {
namespace detail {

// Server constructor
UnixChannel::UnixChannel(const std::string& uri, size_t buffer_size,
                         ChannelMode mode, ChannelType type, const std::string& socket_path)
    : ChannelImpl(uri, buffer_size, mode, type)
    , is_server_(true)
    , connected_(false)
    , stopping_(false)
    , socket_path_(socket_path) {
    
    // Remove existing socket file if it exists
    cleanup_socket_file();
    
    // Create acceptor for listening
    acceptor_ = std::make_unique<unix_socket::acceptor>(io_context_, 
                                                       unix_socket::endpoint(socket_path_));
    
    // Start IO thread
    io_thread_ = std::thread([this]() { run_io_service(); });
    
    // Start accepting connections
    start_accept();
}

// Client constructor
UnixChannel::UnixChannel(const std::string& uri, size_t buffer_size,
                         ChannelMode mode, ChannelType type,
                         const std::string& socket_path, bool is_client)
    : ChannelImpl(uri, buffer_size, mode, type)
    , is_server_(false)
    , connected_(false)
    , stopping_(false)
    , socket_path_(socket_path) {
    
    // Create socket
    socket_ = std::make_unique<unix_socket::socket>(io_context_);
    
    // Start IO thread
    io_thread_ = std::thread([this]() { run_io_service(); });
    
    // Start connecting
    start_connect();
    
    // Wait for connection (with timeout)
    auto start = std::chrono::steady_clock::now();
    while (!connected_ && !stopping_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5)) {
            throw std::runtime_error("Failed to connect to Unix socket: timeout");
        }
    }
    
    if (stopping_) {
        throw std::runtime_error("Failed to connect to Unix socket: operation cancelled");
    }
}

UnixChannel::~UnixChannel() {
    stopping_ = true;
    
    // Stop IO operations
    io_context_.stop();
    
    // Close sockets
    if (socket_) {
        boost::system::error_code ec;
        socket_->close(ec);
    }
    
    if (acceptor_) {
        boost::system::error_code ec;
        acceptor_->close(ec);
    }
    
    // Wait for IO thread
    if (io_thread_.joinable()) {
        io_thread_.join();
    }
    
    // Cleanup socket file if we're the server
    if (is_server_) {
        cleanup_socket_file();
    }
}

void* UnixChannel::reserve_space(size_t size) {
    if (size == 0) return nullptr;
    
    // Add space for frame header
    size_t total_size = sizeof(UnixFrameHeader) + size;
    
    // Allocate buffer
    auto buffer = std::make_unique<std::vector<uint8_t>>(total_size);
    void* user_data = buffer->data() + sizeof(UnixFrameHeader);
    std::vector<uint8_t>* buffer_ptr = buffer.release();
    
    // Map user data pointer to buffer pointer
    {
        std::lock_guard<std::mutex> lock(handle_mutex_);
        user_data_to_buffer_[user_data] = buffer_ptr;
    }
    
    return user_data;
}

void UnixChannel::commit_message(void* handle) {
    if (!handle) return;
    
    // Find buffer from user data pointer
    std::vector<uint8_t>* buffer = nullptr;
    {
        std::lock_guard<std::mutex> lock(handle_mutex_);
        auto it = user_data_to_buffer_.find(handle);
        if (it == user_data_to_buffer_.end()) {
            return; // Invalid handle
        }
        buffer = it->second;
        user_data_to_buffer_.erase(it); // Remove mapping
    }
    
    // Get user data info
    uint8_t* user_data = buffer->data() + sizeof(UnixFrameHeader);
    size_t user_size = buffer->size() - sizeof(UnixFrameHeader);
    
    // Fill in frame header
    UnixFrameHeader* header = reinterpret_cast<UnixFrameHeader*>(buffer->data());
    header->length = static_cast<uint32_t>(buffer->size());
    header->checksum = calculate_checksum(user_data, user_size);
    header->type = 0; // TODO: Get from message type system
    
    // Add to outgoing queue
    {
        std::lock_guard<std::mutex> lock(outgoing_mutex_);
        outgoing_messages_.emplace(std::move(*buffer));
    }
    outgoing_cv_.notify_one();
    
    delete buffer;
}

void* UnixChannel::receive_message(size_t& size, uint32_t& type) {
    std::unique_lock<std::mutex> lock(incoming_mutex_);
    
    // Wait for message
    incoming_cv_.wait(lock, [this] { 
        return !incoming_messages_.empty() || stopping_; 
    });
    
    if (stopping_ || incoming_messages_.empty()) {
        size = 0;
        type = 0;
        return nullptr;
    }
    
    // Get message
    auto message = std::move(incoming_messages_.front());
    incoming_messages_.pop();
    
    // Extract header info
    const UnixFrameHeader* header = reinterpret_cast<const UnixFrameHeader*>(message.data());
    size = header->length - sizeof(UnixFrameHeader);
    type = header->type;
    
    // Return user data (skip header)
    auto* buffer = new std::vector<uint8_t>(std::move(message));
    return buffer->data() + sizeof(UnixFrameHeader);
}

void UnixChannel::release_message(void* handle) {
    if (!handle) return;
    
    // Calculate buffer start from user data pointer
    uint8_t* user_data = static_cast<uint8_t*>(handle);
    uint8_t* buffer_start = user_data - sizeof(UnixFrameHeader);
    
    auto* buffer = reinterpret_cast<std::vector<uint8_t>*>(buffer_start);
    delete buffer;
}

void UnixChannel::run_io_service() {
    while (!stopping_) {
        try {
            io_context_.run();
            break;
        } catch (const std::exception& e) {
            std::cerr << "Unix channel IO error: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void UnixChannel::start_accept() {
    if (!acceptor_ || stopping_) return;
    
    auto client = std::make_shared<unix_socket::socket>(io_context_);
    
    acceptor_->async_accept(*client,
        [this, client](const boost::system::error_code& error) {
            handle_accept(client, error);
        });
}

void UnixChannel::start_connect() {
    if (!socket_ || stopping_) return;
    
    socket_->async_connect(unix_socket::endpoint(socket_path_),
        [this](const boost::system::error_code& error) {
            handle_connect(error);
        });
}

void UnixChannel::handle_accept(std::shared_ptr<unix_socket::socket> client,
                               const boost::system::error_code& error) {
    if (error || stopping_) {
        if (!stopping_) {
            std::cerr << "Unix socket accept error: " << error.message() << std::endl;
        }
        return;
    }
    
    std::cout << "Unix socket client connected" << std::endl;
    
    // Store active socket
    {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        active_socket_ = client;
    }
    
    connected_ = true;
    
    // Start message operations
    start_receive(client);
    start_send(client);
    
    // Continue accepting new connections
    start_accept();
}

void UnixChannel::handle_connect(const boost::system::error_code& error) {
    if (error || stopping_) {
        if (!stopping_) {
            std::cerr << "Unix socket connect error: " << error.message() << std::endl;
        }
        return;
    }
    
    std::cout << "Connected to Unix socket server" << std::endl;
    
    // Store active socket
    {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        active_socket_ = std::shared_ptr<unix_socket::socket>(socket_.release());
    }
    
    connected_ = true;
    
    // Start message operations
    start_receive(active_socket_);
    start_send(active_socket_);
}

void UnixChannel::start_receive(std::shared_ptr<unix_socket::socket> sock) {
    if (!sock || stopping_) return;
    
    auto header = std::make_shared<UnixFrameHeader>();
    
    asio::async_read(*sock,
        asio::buffer(header.get(), sizeof(UnixFrameHeader)),
        [this, sock, header](const boost::system::error_code& error, size_t bytes_transferred) {
            handle_receive_header(sock, header, error, bytes_transferred);
        });
}

void UnixChannel::handle_receive_header(std::shared_ptr<unix_socket::socket> sock,
                                       std::shared_ptr<UnixFrameHeader> header,
                                       const boost::system::error_code& error,
                                       size_t bytes_transferred) {
    if (error || stopping_) {
        if (!stopping_) {
            std::cerr << "Unix socket receive header error: " << error.message() << std::endl;
            connected_ = false;
        }
        return;
    }
    
    if (bytes_transferred != sizeof(UnixFrameHeader)) {
        std::cerr << "Unix socket incomplete header received" << std::endl;
        start_receive(sock); // Try again
        return;
    }
    
    // Validate header
    if (header->length < sizeof(UnixFrameHeader) || header->length > 64 * 1024 * 1024) {
        std::cerr << "Unix socket invalid message length: " << header->length << std::endl;
        start_receive(sock); // Try again
        return;
    }
    
    // Allocate buffer for complete message (including header)
    auto buffer = std::make_shared<std::vector<uint8_t>>(header->length);
    
    // Copy header to buffer
    std::memcpy(buffer->data(), header.get(), sizeof(UnixFrameHeader));
    
    // Read payload
    size_t payload_size = header->length - sizeof(UnixFrameHeader);
    if (payload_size > 0) {
        asio::async_read(*sock,
            asio::buffer(buffer->data() + sizeof(UnixFrameHeader), payload_size),
            [this, sock, buffer, header_copy = *header](const boost::system::error_code& error, size_t bytes_transferred) {
                handle_receive_payload(sock, buffer, header_copy, error, bytes_transferred);
            });
    } else {
        // Header-only message
        handle_receive_payload(sock, buffer, *header, boost::system::error_code{}, 0);
    }
}

void UnixChannel::handle_receive_payload(std::shared_ptr<unix_socket::socket> sock,
                                        std::shared_ptr<std::vector<uint8_t>> buffer,
                                        const UnixFrameHeader& header,
                                        const boost::system::error_code& error,
                                        size_t bytes_transferred) {
    if (error || stopping_) {
        if (!stopping_) {
            std::cerr << "Unix socket receive payload error: " << error.message() << std::endl;
            connected_ = false;
        }
        return;
    }
    
    size_t expected_payload = header.length - sizeof(UnixFrameHeader);
    if (bytes_transferred != expected_payload) {
        std::cerr << "Unix socket incomplete payload received" << std::endl;
        start_receive(sock); // Try again
        return;
    }
    
    // Verify checksum
    const uint8_t* payload = buffer->data() + sizeof(UnixFrameHeader);
    uint64_t calculated_checksum = calculate_checksum(payload, expected_payload);
    
    if (calculated_checksum != header.checksum) {
        std::cerr << "Unix socket checksum mismatch" << std::endl;
        start_receive(sock); // Try again
        return;
    }
    
    // Add to incoming queue
    {
        std::lock_guard<std::mutex> lock(incoming_mutex_);
        incoming_messages_.emplace(std::move(*buffer));
    }
    incoming_cv_.notify_one();
    
    // Continue receiving
    start_receive(sock);
}

void UnixChannel::start_send(std::shared_ptr<unix_socket::socket> sock) {
    if (!sock || stopping_) return;
    
    std::unique_lock<std::mutex> lock(outgoing_mutex_);
    
    // Wait for messages to send
    outgoing_cv_.wait(lock, [this] { 
        return !outgoing_messages_.empty() || stopping_; 
    });
    
    if (stopping_ || outgoing_messages_.empty()) {
        return;
    }
    
    // Get message to send
    auto buffer = std::make_shared<std::vector<uint8_t>>(std::move(outgoing_messages_.front()));
    outgoing_messages_.pop();
    lock.unlock();
    
    // Send asynchronously
    asio::async_write(*sock,
        asio::buffer(*buffer),
        [this, sock, buffer](const boost::system::error_code& error, size_t bytes_transferred) {
            handle_send(sock, buffer, error, bytes_transferred);
        });
}

void UnixChannel::handle_send(std::shared_ptr<unix_socket::socket> sock,
                             std::shared_ptr<std::vector<uint8_t>> buffer,
                             const boost::system::error_code& error,
                             size_t bytes_transferred) {
    if (error || stopping_) {
        if (!stopping_) {
            std::cerr << "Unix socket send error: " << error.message() << std::endl;
            connected_ = false;
        }
        return;
    }
    
    if (bytes_transferred != buffer->size()) {
        std::cerr << "Unix socket incomplete send" << std::endl;
        return;
    }
    
    // Continue sending if there are more messages
    start_send(sock);
}

uint64_t UnixChannel::calculate_checksum(const void* data, size_t size) {
    return XXHash64::hash(data, size, 0x12345678);
}

void UnixChannel::cleanup_socket_file() {
    if (!socket_path_.empty()) {
        std::error_code ec;
        std::filesystem::remove(socket_path_, ec);
        // Ignore errors - file might not exist
    }
}

} // namespace detail
} // namespace psyne