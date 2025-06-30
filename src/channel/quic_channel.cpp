#include "quic_channel.hpp"
#include "../utils/logger.hpp"
#include <regex>

namespace psyne {
namespace detail {

QUICChannel::QUICChannel(const std::string& uri, size_t buffer_size,
                         ChannelMode mode, ChannelType type, bool is_server)
    : ChannelImpl(uri, buffer_size, mode, type, true),
      is_server_(is_server) {
    
    // Create ring buffer based on mode
    switch (mode) {
    case ChannelMode::SPSC:
        ring_buffer_ = std::make_unique<SPSCRingBuffer>(buffer_size);
        break;
    case ChannelMode::MPSC:
        ring_buffer_ = std::make_unique<MPSCRingBuffer>(buffer_size);
        break;
    case ChannelMode::SPMC:
        ring_buffer_ = std::make_unique<SPMCRingBuffer>(buffer_size);
        break;
    case ChannelMode::MPMC:
        ring_buffer_ = std::make_unique<MPMCRingBuffer>(buffer_size);
        break;
    }
    
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
        return Channel::BUFFER_FULL;
    }
    
    return ring_buffer_->reserve_write_space(size);
}

void QUICChannel::notify_message_ready(uint32_t offset, size_t size) noexcept {
    if (!connected_.load()) {
        return;
    }
    
    // Queue for sending
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        pending_sends_.push({offset, size});
    }
    send_cv_.notify_one();
}

RingBuffer& QUICChannel::get_ring_buffer() noexcept {
    return *ring_buffer_;
}

const RingBuffer& QUICChannel::get_ring_buffer() const noexcept {
    return *ring_buffer_;
}

void QUICChannel::advance_read_pointer(size_t size) noexcept {
    ring_buffer_->advance_read_pointer(size);
}

std::span<uint8_t> QUICChannel::get_write_span(size_t size) noexcept {
    uint32_t offset = reserve_write_slot(size);
    if (offset == Channel::BUFFER_FULL) {
        return {};
    }
    
    return ring_buffer_->get_write_span(offset, size);
}

std::span<const uint8_t> QUICChannel::buffer_span() const noexcept {
    return ring_buffer_->available_read_span();
}

void QUICChannel::run_client(const std::string& host, uint16_t port) {
    try {
        transport::QUICConfig config;
        config.max_idle_timeout_ms = 30000;
        config.enable_0rtt = true;
        
        // Create QUIC client connection
        auto client = transport::create_quic_client(config);
        connection_ = client->connect(host, port);
        
        if (!connection_) {
            log_error("Failed to connect to QUIC server at {}:{}", host, port);
            return;
        }
        
        connected_ = true;
        log_info("QUIC client connected to {}:{}", host, port);
        
        // Start send/receive loops
        std::thread send_thread([this]() { send_loop(); });
        std::thread receive_thread([this]() { receive_loop(); });
        
        // Run IO context
        io_context_.run();
        
        send_thread.join();
        receive_thread.join();
        
    } catch (const std::exception& e) {
        log_error("QUIC client error: {}", e.what());
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
            log_error("Failed to start QUIC server on port {}", port);
            return;
        }
        
        log_info("QUIC server listening on port {}", port);
        
        // Accept one connection for channel
        connection_ = server_->accept();
        
        if (!connection_) {
            log_error("Failed to accept QUIC connection");
            return;
        }
        
        connected_ = true;
        log_info("QUIC server accepted connection");
        
        // Start send/receive loops
        std::thread send_thread([this]() { send_loop(); });
        std::thread receive_thread([this]() { receive_loop(); });
        
        // Run IO context
        io_context_.run();
        
        send_thread.join();
        receive_thread.join();
        
    } catch (const std::exception& e) {
        log_error("QUIC server error: {}", e.what());
        connected_ = false;
    }
}

void QUICChannel::send_loop() {
    // Create a QUIC stream for sending
    auto stream = connection_->create_stream(transport::QUICStreamDirection::UNIDIRECTIONAL);
    if (!stream) {
        log_error("Failed to create QUIC stream for sending");
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
            
            // Get data from ring buffer
            auto data_span = ring_buffer_->get_read_span(offset, size);
            
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
        log_error("Failed to accept QUIC stream for receiving");
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
        
        // Reserve space in ring buffer
        uint32_t offset = ring_buffer_->reserve_write_space(header.size);
        if (offset == RingBuffer::BUFFER_FULL) {
            // Buffer full, skip message
            std::vector<uint8_t> discard(header.size);
            stream->receive(discard.data(), discard.size());
            continue;
        }
        
        // Read directly into ring buffer
        auto write_span = ring_buffer_->get_write_span(offset, header.size);
        ssize_t received = stream->receive(write_span.data(), write_span.size());
        
        if (received == static_cast<ssize_t>(header.size)) {
            // Notify that message is ready
            ring_buffer_->commit_write(offset, header.size);
            
            // Update metrics
            messages_received_++;
            bytes_received_ += header.size;
        }
    }
}

// Legacy interface implementations
void* QUICChannel::reserve_space(size_t size) {
    uint32_t offset = reserve_write_slot(size);
    if (offset == Channel::BUFFER_FULL) {
        return nullptr;
    }
    
    auto span = ring_buffer_->get_write_span(offset, size);
    return span.data();
}

void QUICChannel::commit_message(void* handle) {
    // No-op in zero-copy design
}

void* QUICChannel::receive_message(size_t& size, uint32_t& type) {
    auto read_span = ring_buffer_->available_read_span();
    
    if (read_span.size() < sizeof(SlabHeader)) {
        return nullptr;
    }
    
    auto* slab = reinterpret_cast<SlabHeader*>(read_span.data());
    size = slab->len;
    type = slab->reserved;
    
    if (read_span.size() < sizeof(SlabHeader) + size) {
        return nullptr;
    }
    
    return slab;
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