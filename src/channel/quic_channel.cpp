#include "quic_channel.hpp"
#include <iostream>
#include <regex>
#include <stdexcept>

namespace psyne {
namespace detail {

QUICChannel::QUICChannel(const std::string& uri, size_t buffer_size,
                         ChannelMode mode, ChannelType type, bool is_server)
    : ChannelImpl(uri, buffer_size, mode, type, true),
      is_server_(is_server) {
    
    // Initialize message queues
    
    // Parse URI: quic://host:port or quic://:port
    std::regex uri_regex(R"(quic://([^:]*):(\d+))");
    std::smatch match;
    
    if (!std::regex_match(uri, match, uri_regex)) {
        throw std::runtime_error("Invalid QUIC URI: " + uri);
    }
    
    host_ = match[1].str();
    port_ = static_cast<uint16_t>(std::stoi(match[2].str()));
    
    // Start network thread
    io_thread_ = std::thread([this]() {
        if (is_server_) {
            run_server(port_);
        } else {
            run_client(host_, port_);
        }
    });
}

QUICChannel::~QUICChannel() {
    stop();
    
    stopping_ = true;
    send_cv_.notify_all();
    
    if (connection_) {
        connection_->close();
    }
    
    io_context_.stop();
    
    if (io_thread_.joinable()) {
        io_thread_.join();
    }
}

uint32_t QUICChannel::reserve_write_slot(size_t size) noexcept {
    if (!connected_.load()) {
        return 0xFFFFFFFF; // BUFFER_FULL
    }
    
    // For simplified implementation, just return 0 as offset
    return 0;
}

void QUICChannel::notify_message_ready(uint32_t offset, size_t size) noexcept {
    if (!connected_.load()) {
        return;
    }
    
    // For simplified implementation, queue the message size
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        pending_sends_.push({offset, size});
    }
    send_cv_.notify_one();
}

RingBuffer& QUICChannel::get_ring_buffer() noexcept {
    // For now, throw - this method should not be called in the simplified implementation
    throw std::runtime_error("get_ring_buffer not implemented for QUIC channel");
}

const RingBuffer& QUICChannel::get_ring_buffer() const noexcept {
    // For now, throw - this method should not be called in the simplified implementation
    throw std::runtime_error("get_ring_buffer not implemented for QUIC channel");
}

void QUICChannel::advance_read_pointer(size_t size) noexcept {
    // For simplified implementation, just remove from recv queue
    std::lock_guard<std::mutex> lock(recv_mutex_);
    if (!recv_queue_.empty()) {
        recv_queue_.pop();
    }
}

std::span<uint8_t> QUICChannel::get_write_span(size_t size) noexcept {
    // Simplified implementation - return a temporary buffer
    static thread_local std::vector<uint8_t> temp_buffer;
    temp_buffer.resize(size);
    return std::span<uint8_t>{temp_buffer.data(), size};
}

std::span<const uint8_t> QUICChannel::buffer_span() const noexcept {
    std::lock_guard<std::mutex> lock(recv_mutex_);
    if (recv_queue_.empty()) {
        return {};
    }
    
    const auto& msg = recv_queue_.front();
    return std::span<const uint8_t>{msg.data.data(), msg.data.size()};
}

void QUICChannel::run_client(const std::string& host, uint16_t port) {
    try {
        transport::QUICConfig config;
        config.max_idle_timeout_ms = 30000;
        config.enable_0rtt = true;
        
        // Create QUIC client connection
        connection_ = transport::create_quic_client(host, port, config);
        
        if (!connection_) {
            std::cerr << "Failed to connect to QUIC server at " << host << ":" << port << std::endl;
            return;
        }
        
        connected_ = true;
        std::cout << "QUIC client connected to " << host << ":" << port << std::endl;
        
        // Start send/receive loops
        std::thread send_thread([this]() { send_loop(); });
        std::thread receive_thread([this]() { receive_loop(); });
        
        // Run IO context
        io_context_.run();
        
        send_thread.join();
        receive_thread.join();
        
    } catch (const std::exception& e) {
        std::cerr << "QUIC client error: " << e.what() << std::endl;
        connected_ = false;
    }
}

void QUICChannel::run_server(uint16_t port) {
    try {
        transport::QUICConfig config;
        config.max_idle_timeout_ms = 30000;
        config.enable_0rtt = true;
        
        // Create QUIC server
        server_ = transport::create_quic_server(port, config);
        
        if (!server_->start()) {
            std::cerr << "Failed to start QUIC server on port " << port << std::endl;
            return;
        }
        
        std::cout << "QUIC server listening on port " << port << std::endl;
        
        // Accept one connection for channel
        connection_ = server_->accept();
        
        if (!connection_) {
            std::cerr << "Failed to accept QUIC connection" << std::endl;
            return;
        }
        
        connected_ = true;
        std::cout << "QUIC server accepted connection" << std::endl;
        
        // Start send/receive loops
        std::thread send_thread([this]() { send_loop(); });
        std::thread receive_thread([this]() { receive_loop(); });
        
        // Run IO context
        io_context_.run();
        
        send_thread.join();
        receive_thread.join();
        
    } catch (const std::exception& e) {
        std::cerr << "QUIC server error: " << e.what() << std::endl;
        connected_ = false;
    }
}

void QUICChannel::send_loop() {
    // Create a QUIC stream for sending
    auto stream = connection_->create_stream(transport::QUICStreamDirection::UNIDIRECTIONAL);
    if (!stream) {
        std::cerr << "Failed to create QUIC stream for sending" << std::endl;
        return;
    }
    
    while (!stopping_ && connected_) {
        std::unique_lock<std::mutex> lock(send_mutex_);
        
        send_cv_.wait(lock, [this]() { 
            return !pending_sends_.empty() || stopping_; 
        });
        
        if (stopping_) break;
        
        while (!pending_sends_.empty()) {
            auto [offset, size] = pending_sends_.front();
            pending_sends_.pop();
            lock.unlock();
            
            // For simplified implementation, create dummy data
            std::vector<uint8_t> data(size, 0);
            std::span<uint8_t> data_span{data.data(), size};
            
            if (!data_span.empty()) {
                // Send header with size and type info
                struct MessageHeader {
                    uint32_t size;
                    uint32_t type;
                } header = {
                    static_cast<uint32_t>(size),
                    static_cast<uint32_t>(type_)
                };
                
                // Send header then data
                stream->send(&header, sizeof(header));
                stream->send(data_span.data(), data_span.size());
                
                // Update metrics
                messages_sent_++;
                bytes_sent_ += size;
            }
            
            lock.lock();
        }
    }
}

void QUICChannel::receive_loop() {
    // Create a QUIC stream for receiving
    auto stream = connection_->accept_stream();
    if (!stream) {
        std::cerr << "Failed to accept QUIC stream for receiving" << std::endl;
        return;
    }
    
    while (!stopping_ && connected_) {
        // Read message header
        struct MessageHeader {
            uint32_t size;
            uint32_t type;
        } header;
        
        ssize_t n = stream->receive(&header, sizeof(header));
        if (n != sizeof(header)) {
            if (n == 0) break; // Stream closed
            continue;
        }
        
        // Create message buffer
        std::vector<uint8_t> message_data(header.size);
        ssize_t received = stream->receive(message_data.data(), message_data.size());
        
        if (received == static_cast<ssize_t>(header.size)) {
            // Add to receive queue
            {
                std::lock_guard<std::mutex> lock(recv_mutex_);
                recv_queue_.push({std::move(message_data), header.type});
            }
            recv_cv_.notify_one();
            
            // Update metrics
            messages_received_++;
            bytes_received_ += header.size;
        }
    }
}

// Legacy interface implementations
void* QUICChannel::reserve_space(size_t size) {
    // Simplified implementation - return temporary space
    static thread_local std::vector<uint8_t> temp_buffer;
    temp_buffer.resize(size);
    return temp_buffer.data();
}

void QUICChannel::commit_message(void* handle) {
    // No-op in zero-copy design
}

void* QUICChannel::receive_message(size_t& size, uint32_t& type) {
    std::lock_guard<std::mutex> lock(recv_mutex_);
    
    if (recv_queue_.empty()) {
        return nullptr;
    }
    
    const auto& msg = recv_queue_.front();
    size = msg.data.size();
    type = msg.type;
    
    return const_cast<uint8_t*>(msg.data.data());
}

void QUICChannel::release_message(void* handle) {
    if (!handle) return;
    
    auto* slab = static_cast<SlabHeader*>(handle);
    size_t total_size = sizeof(SlabHeader) + slab->len;
    
    advance_read_pointer(total_size);
}

// Factory function
std::unique_ptr<ChannelImpl> 
create_quic_channel(const std::string& uri, size_t buffer_size,
                    ChannelMode mode, ChannelType type,
                    const compression::CompressionConfig& compression_config) {
    // Parse URI to determine if server or client
    std::regex uri_regex(R"(quic://([^:]*):(\d+))");
    std::smatch match;
    
    if (!std::regex_match(uri, match, uri_regex)) {
        throw std::runtime_error("Invalid QUIC URI: " + uri);
    }
    
    std::string host = match[1].str();
    bool is_server = host.empty() || host == "*";
    
    auto channel = std::make_unique<QUICChannel>(uri, buffer_size, mode, type, is_server);
    
    // Note: Compression could be applied at the QUIC stream level
    // but is not implemented in this version
    
    return channel;
}

} // namespace detail
} // namespace psyne