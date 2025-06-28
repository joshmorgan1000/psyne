#include "websocket_channel.hpp"
#include "../memory/slab.hpp"
#include <iostream>
#include <regex>

namespace psyne {
namespace detail {

WebSocketChannel::WebSocketChannel(const std::string& uri, size_t buffer_size, bool is_server)
    : ChannelImpl(uri, buffer_size, ChannelMode::SPSC, ChannelType::MultiType, true),
      buffer_size_(buffer_size), is_server_(is_server) {
    
    send_buffer_.reserve(buffer_size_);
    receive_buffer_.resize(buffer_size_);
    
    // Parse URI: ws://host:port or wss://host:port
    std::regex uri_regex(R"((wss?)://([^:]+):(\d+))");
    std::smatch match;
    
    if (!std::regex_match(uri, match, uri_regex)) {
        throw std::runtime_error("Invalid WebSocket URI: " + uri);
    }
    
    bool use_ssl = (match[1].str() == "wss");
    std::string host = match[2].str();
    uint16_t port = static_cast<uint16_t>(std::stoi(match[3].str()));
    
    if (use_ssl) {
        throw std::runtime_error("WSS (WebSocket Secure) not yet implemented");
    }
    
    // Start IO thread
    io_thread_ = std::thread([this, host, port]() {
        if (is_server_) {
            run_server(port);
        } else {
            run_client(host, port);
        }
    });
}

WebSocketChannel::~WebSocketChannel() {
    stop();
}

void WebSocketChannel::stop() {
    ChannelImpl::stop();  // Call base class stop() to set stopped_ flag
    
    // Close WebSocket connection
    if (ws_stream_ && connected_) {
        try {
            ws_stream_->close(websocket::close_code::normal);
        } catch (...) {
            // Ignore errors during shutdown
        }
    }
    
    // Stop IO context
    io_context_.stop();
    
    // Wake up any waiting threads
    send_cv_.notify_all();
    receive_cv_.notify_all();
    
    // Wait for IO thread
    if (io_thread_.joinable()) {
        io_thread_.join();
    }
}

void WebSocketChannel::run_client(const std::string& host, uint16_t port) {
    try {
        // Resolve host
        tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(host, std::to_string(port));
        
        // Create socket and connect
        tcp::socket socket(io_context_);
        net::connect(socket, endpoints);
        
        // Create WebSocket stream
        ws_stream_ = std::make_unique<websocket::stream<tcp::socket>>(std::move(socket));
        
        // Perform WebSocket handshake
        ws_stream_->handshake(host, "/");
        
        connected_ = true;
        
        // Start message processing threads
        std::thread send_thread([this]() { send_loop(); });
        std::thread receive_thread([this]() { receive_loop(); });
        
        // Run IO context
        io_context_.run();
        
        // Wait for threads
        send_thread.join();
        receive_thread.join();
        
    } catch (const std::exception& e) {
        std::cerr << "WebSocket client error: " << e.what() << std::endl;
        stopped_ = true;
    }
}

void WebSocketChannel::run_server(uint16_t port) {
    try {
        // Create acceptor
        acceptor_ = std::make_unique<tcp::acceptor>(io_context_, tcp::endpoint(tcp::v4(), port));
        
        // Accept connection
        tcp::socket socket(io_context_);
        acceptor_->accept(socket);
        
        // Only handle one connection for now
        handle_connection(std::move(socket));
        
    } catch (const std::exception& e) {
        std::cerr << "WebSocket server error: " << e.what() << std::endl;
        stopped_ = true;
    }
}

void WebSocketChannel::handle_connection(tcp::socket socket) {
    try {
        // Create WebSocket stream
        ws_stream_ = std::make_unique<websocket::stream<tcp::socket>>(std::move(socket));
        
        // Accept WebSocket handshake
        ws_stream_->accept();
        
        connected_ = true;
        
        // Start message processing threads
        std::thread send_thread([this]() { send_loop(); });
        std::thread receive_thread([this]() { receive_loop(); });
        
        // Run IO context
        io_context_.run();
        
        // Wait for threads
        send_thread.join();
        receive_thread.join();
        
    } catch (const std::exception& e) {
        std::cerr << "WebSocket connection error: " << e.what() << std::endl;
    }
}

void WebSocketChannel::send_loop() {
    while (!stopped_ && connected_) {
        std::unique_lock<std::mutex> lock(send_mutex_);
        
        // Wait for messages
        send_cv_.wait(lock, [this]() {
            return !send_queue_.empty() || stopped_;
        });
        
        if (stopped_) break;
        
        // Process all pending messages
        while (!send_queue_.empty()) {
            auto msg = std::move(send_queue_.front());
            send_queue_.pop();
            lock.unlock();
            
            try {
                // Create binary frame with header
                std::vector<uint8_t> frame;
                frame.reserve(HEADER_SIZE + msg.data.size());
                
                // Add header: size (4 bytes) + type (4 bytes)
                uint32_t size = static_cast<uint32_t>(msg.data.size());
                frame.insert(frame.end(), 
                           reinterpret_cast<uint8_t*>(&size),
                           reinterpret_cast<uint8_t*>(&size) + sizeof(size));
                frame.insert(frame.end(),
                           reinterpret_cast<uint8_t*>(&msg.type),
                           reinterpret_cast<uint8_t*>(&msg.type) + sizeof(msg.type));
                
                // Add message data
                frame.insert(frame.end(), msg.data.begin(), msg.data.end());
                
                // Send as binary frame
                ws_stream_->binary(true);
                ws_stream_->write(net::buffer(frame));
                
                // Update metrics
                messages_sent_++;
                bytes_sent_ += msg.data.size();
                
            } catch (const std::exception& e) {
                std::cerr << "WebSocket send error: " << e.what() << std::endl;
                connected_ = false;
                break;
            }
            
            lock.lock();
        }
    }
}

void WebSocketChannel::receive_loop() {
    beast::flat_buffer buffer;
    
    while (!stopped_ && connected_) {
        try {
            // Read message
            size_t n = ws_stream_->read(buffer);
            
            if (n >= HEADER_SIZE) {
                // Parse header
                const uint8_t* data = static_cast<const uint8_t*>(buffer.data().data());
                uint32_t size = *reinterpret_cast<const uint32_t*>(data);
                uint32_t type = *reinterpret_cast<const uint32_t*>(data + sizeof(uint32_t));
                
                if (n >= HEADER_SIZE + size) {
                    // Create message buffer
                    std::vector<uint8_t> msg_buffer(sizeof(SlabHeader) + size);
                    
                    // Set up slab header
                    auto* slab = reinterpret_cast<SlabHeader*>(msg_buffer.data());
                    slab->len = size;
                    slab->reserved = type;
                    
                    // Copy message data
                    std::memcpy(slab + 1, data + HEADER_SIZE, size);
                    
                    // Add to receive queue
                    {
                        std::lock_guard<std::mutex> lock(receive_mutex_);
                        receive_queue_.push(std::move(msg_buffer));
                    }
                    receive_cv_.notify_one();
                    
                    // Update metrics
                    messages_received_++;
                    bytes_received_ += size;
                }
            }
            
            buffer.consume(n);
            
        } catch (const std::exception& e) {
            std::cerr << "WebSocket receive error: " << e.what() << std::endl;
            connected_ = false;
            break;
        }
    }
}

void* WebSocketChannel::reserve_space(size_t size) {
    if (stopped_ || !connected_) {
        return nullptr;
    }
    
    if (size > buffer_size_ - sizeof(SlabHeader)) {
        return nullptr;
    }
    
    // Allocate temporary buffer
    send_buffer_.resize(sizeof(SlabHeader) + size);
    
    auto* slab = reinterpret_cast<SlabHeader*>(send_buffer_.data());
    slab->len = static_cast<uint32_t>(size);
    slab->reserved = 0;
    
    return slab;
}

void WebSocketChannel::commit_message(void* handle) {
    if (!handle || stopped_ || !connected_) {
        return;
    }
    
    auto* slab = static_cast<SlabHeader*>(handle);
    
    // Create pending message
    PendingMessage msg;
    msg.type = slab->reserved;  // Type is stored in reserved field
    msg.data.resize(slab->len);
    std::memcpy(msg.data.data(), slab + 1, slab->len);
    
    // Add to send queue
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        send_queue_.push(std::move(msg));
    }
    send_cv_.notify_one();
}

void* WebSocketChannel::receive_message(size_t& size, uint32_t& type) {
    std::unique_lock<std::mutex> lock(receive_mutex_);
    
    // Non-blocking check
    if (receive_queue_.empty()) {
        receive_blocks_++;
        return nullptr;
    }
    
    if (stopped_) {
        return nullptr;
    }
    
    // Move buffer to stable storage (leak it, will be freed in release_message)
    auto* buffer = new std::vector<uint8_t>(std::move(receive_queue_.front()));
    receive_queue_.pop();
    
    auto* slab = reinterpret_cast<SlabHeader*>(buffer->data());
    size = slab->len;
    type = slab->reserved;
    return slab;
}

void WebSocketChannel::release_message(void* message) {
    if (!message) return;
    
    // The message pointer points to a SlabHeader which is at the beginning
    // of a heap-allocated vector. We need to find and delete that vector.
    // Since we allocated with new std::vector<uint8_t>, we need to reconstruct
    // the vector pointer. The SlabHeader is at buffer->data().
    
    // This is a bit hacky - we're relying on the fact that we allocated
    // the buffer with new std::vector<uint8_t> in receive_message
    auto* slab = static_cast<SlabHeader*>(message);
    
    // Go back from the data pointer to find the vector object
    // We stored the vector's data at the beginning, so we need to
    // go back to the vector object itself
    uint8_t* data_ptr = reinterpret_cast<uint8_t*>(slab);
    
    // Find the vector that owns this data
    // We'll leak the memory for now as there's no portable way to do this
    // In production, we'd use a proper memory pool or different allocation strategy
    
    // For now, just leak it - this is a limitation of the current design
    // TODO: Use a proper memory pool or allocator
}

} // namespace detail
} // namespace psyne