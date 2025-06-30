#include "websocket_channel.hpp"
#include "../memory/ring_buffer_impl.hpp"
#include <iostream>
#include <regex>
#include <unordered_map>

namespace psyne {
namespace detail {

WebSocketChannel::WebSocketChannel(const std::string &uri, size_t buffer_size,
                                   bool is_server)
    : ChannelImpl(uri, buffer_size, ChannelMode::SPSC, ChannelType::MultiType,
                  true),
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
    ChannelImpl::stop(); // Call base class stop() to set stopped_ flag

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

void WebSocketChannel::run_client(const std::string &host, uint16_t port) {
    try {
        // Resolve host
        tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(host, std::to_string(port));

        // Create socket and connect
        tcp::socket socket(io_context_);
        net::connect(socket, endpoints);

        // Create WebSocket stream
        ws_stream_ =
            std::make_unique<websocket::stream<tcp::socket>>(std::move(socket));

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

    } catch (const std::exception &e) {
        std::cerr << "WebSocket client error: " << e.what() << std::endl;
        stopped_ = true;
    }
}

void WebSocketChannel::run_server(uint16_t port) {
    try {
        // Create acceptor
        acceptor_ = std::make_unique<tcp::acceptor>(
            io_context_, tcp::endpoint(tcp::v4(), port));

        // Accept connection
        tcp::socket socket(io_context_);
        acceptor_->accept(socket);

        // Only handle one connection for now
        handle_connection(std::move(socket));

    } catch (const std::exception &e) {
        std::cerr << "WebSocket server error: " << e.what() << std::endl;
        stopped_ = true;
    }
}

void WebSocketChannel::handle_connection(tcp::socket socket) {
    try {
        // Create WebSocket stream
        ws_stream_ =
            std::make_unique<websocket::stream<tcp::socket>>(std::move(socket));

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

    } catch (const std::exception &e) {
        std::cerr << "WebSocket connection error: " << e.what() << std::endl;
    }
}

void WebSocketChannel::send_loop() {
    while (!stopped_ && connected_) {
        std::unique_lock<std::mutex> lock(send_mutex_);

        // Wait for messages
        send_cv_.wait(lock,
                      [this]() { return !send_queue_.empty() || stopped_; });

        if (stopped_)
            break;

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
                frame.insert(frame.end(), reinterpret_cast<uint8_t *>(&size),
                             reinterpret_cast<uint8_t *>(&size) + sizeof(size));
                frame.insert(
                    frame.end(), reinterpret_cast<uint8_t *>(&msg.type),
                    reinterpret_cast<uint8_t *>(&msg.type) + sizeof(msg.type));

                // Add message data
                frame.insert(frame.end(), msg.data.begin(), msg.data.end());

                // Send as binary frame
                ws_stream_->binary(true);
                ws_stream_->write(net::buffer(frame));

                // Update metrics
                messages_sent_++;
                bytes_sent_ += msg.data.size();

            } catch (const std::exception &e) {
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
                const uint8_t *data =
                    static_cast<const uint8_t *>(buffer.data().data());
                uint32_t size = *reinterpret_cast<const uint32_t *>(data);
                uint32_t type = *reinterpret_cast<const uint32_t *>(
                    data + sizeof(uint32_t));

                if (n >= HEADER_SIZE + size) {
                    // Create message buffer
                    std::vector<uint8_t> msg_buffer(sizeof(SlabHeader) + size);

                    // Set up slab header
                    auto *slab =
                        reinterpret_cast<SlabHeader *>(msg_buffer.data());
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

        } catch (const std::exception &e) {
            std::cerr << "WebSocket receive error: " << e.what() << std::endl;
            connected_ = false;
            break;
        }
    }
}

uint32_t WebSocketChannel::reserve_write_slot(size_t size) noexcept {
    if (stopped_ || !connected_) {
        return BUFFER_FULL;
    }

    if (size > buffer_size_) {
        return BUFFER_FULL;
    }

    // Reserve space directly in ring buffer for zero-copy WebSocket
    auto& ring_buffer = get_ring_buffer();
    return ring_buffer.reserve_write_space(size);
}

void WebSocketChannel::notify_message_ready(uint32_t offset, size_t size) noexcept {
    if (stopped_ || !connected_) {
        return;
    }

    // Message is already in ring buffer at offset
    // Send directly from ring buffer for zero-copy WebSocket
    auto& ring_buffer = get_ring_buffer();
    auto data_span = ring_buffer.get_read_span(offset, size);
    
    // Send WebSocket frame with zero-copy data
    send_websocket_frame(data_span);
    
    // Advance read pointer after successful send
    ring_buffer.advance_read_pointer(size);
    }
    send_cv_.notify_one();
}

std::span<uint8_t> WebSocketChannel::get_write_span(size_t size) noexcept {
    uint32_t offset = reserve_write_slot(size);
    if (offset == BUFFER_FULL) {
        return {};
    }
    
    auto& ring_buffer = get_ring_buffer();
    return ring_buffer.get_write_span(offset, size);
}

std::span<const uint8_t> WebSocketChannel::buffer_span() const noexcept {
    auto& ring_buffer = get_ring_buffer();
    return ring_buffer.available_read_span();
}

void WebSocketChannel::advance_read_pointer(size_t size) noexcept {
    auto& ring_buffer = get_ring_buffer();
    ring_buffer.advance_read_pointer(size);
}

void WebSocketChannel::send_websocket_frame(std::span<const uint8_t> data) {
    if (!connected_ || stopped_) {
        return;
    }
    
    // Create WebSocket frame header for zero-copy sending
    std::vector<uint8_t> frame;
    frame.reserve(14); // Max header size
    
    // WebSocket frame format: FIN + opcode + payload length
    frame.push_back(0x82); // FIN=1, opcode=binary
    
    if (data.size() < 126) {
        frame.push_back(static_cast<uint8_t>(data.size()));
    } else if (data.size() < 65536) {
        frame.push_back(126);
        frame.push_back(static_cast<uint8_t>(data.size() >> 8));
        frame.push_back(static_cast<uint8_t>(data.size() & 0xFF));
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; --i) {
            frame.push_back(static_cast<uint8_t>((data.size() >> (i * 8)) & 0xFF));
        }
    }
    
    // Zero-copy WebSocket send using scatter-gather I/O
    std::array<asio::const_buffer, 2> buffers = {
        asio::buffer(frame),
        asio::buffer(data.data(), data.size())
    };
    
    asio::async_write(socket_, buffers,
        [this](const boost::system::error_code& error, size_t bytes_transferred) {
            if (error && !stopped_) {
                connected_ = false;
            }
        });
}

// Legacy deprecated methods (kept for compatibility)
void* WebSocketChannel::reserve_space(size_t size) {
    uint32_t offset = reserve_write_slot(size);
    if (offset == BUFFER_FULL) {
        return nullptr;
    }
    auto& ring_buffer = get_ring_buffer();
    auto span = ring_buffer.get_write_span(offset, size);
    return span.data();
}

void WebSocketChannel::commit_message(void* handle) {
    // This is a no-op in zero-copy design
    // Message was already written directly to ring buffer
}

void *WebSocketChannel::receive_message(size_t &size, uint32_t &type) {
    std::unique_lock<std::mutex> lock(receive_mutex_);

    // Non-blocking check
    if (receive_queue_.empty()) {
        receive_blocks_++;
        return nullptr;
    }

    if (stopped_) {
        return nullptr;
    }

    // Move buffer to stable storage with tracking
    auto buffer = std::make_unique<std::vector<uint8_t>>(std::move(receive_queue_.front()));
    receive_queue_.pop();

    auto *slab = reinterpret_cast<SlabHeader *>(buffer->data());
    size = slab->len;
    type = slab->reserved;
    
    // Track the buffer for later cleanup
    {
        static std::unordered_map<void*, std::unique_ptr<std::vector<uint8_t>>> allocated_buffers;
        static std::mutex buffer_mutex;
        
        std::lock_guard<std::mutex> lock(buffer_mutex);
        allocated_buffers[buffer->data()] = std::move(buffer);
    }
    
    return slab;
}

void WebSocketChannel::release_message(void *message) {
    if (!message)
        return;

    // The message pointer points to a SlabHeader which is at the beginning
    // of a heap-allocated vector. We need to find and delete that vector.
    // Since we allocated with new std::vector<uint8_t>, we need to reconstruct
    // the vector pointer. The SlabHeader is at buffer->data().

    // This is a bit hacky - we're relying on the fact that we allocated
    // the buffer with new std::vector<uint8_t> in receive_message
    auto *slab = static_cast<SlabHeader *>(message);

    // Go back from the data pointer to find the vector object
    // We stored the vector's data at the beginning, so we need to
    // go back to the vector object itself
    uint8_t *data_ptr = reinterpret_cast<uint8_t *>(slab);

    // Find the vector that owns this data
    // We'll leak the memory for now as there's no portable way to do this
    // In production, we'd use a proper memory pool or different allocation
    // strategy

    // Use a simple tracking mechanism for allocated buffers
    // In production, a proper memory pool would be more efficient
    static std::unordered_map<void*, std::unique_ptr<std::vector<uint8_t>>> allocated_buffers;
    static std::mutex buffer_mutex;
    
    std::lock_guard<std::mutex> lock(buffer_mutex);
    allocated_buffers.erase(data_ptr);
}

} // namespace detail
} // namespace psyne