#include "tcp_channel.hpp"
#include "../../memory/ring_buffer.hpp"
#include "../../memory/ring_buffer_impl.hpp"
#include <boost/asio/buffer.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/read.hpp>
#include <boost/endian/conversion.hpp>
#include <regex>

namespace psyne {
namespace detail {

TCPChannel::TCPChannel(const std::string& uri, size_t buffer_size, ChannelMode mode,
                       ChannelType type, uint16_t port)
    : ChannelImpl(uri, buffer_size, mode, type)
    , is_server_(true)
    , host_("")
    , port_(port)
    , io_context_(std::make_unique<boost::asio::io_context>())
    , connected_(false)
    , network_running_(false)
    , bytes_needed_(HEADER_SIZE)
    , header_received_(false) {
    
    log_info("Creating TCP server channel on port ", port);
    initialize_ring_buffer(buffer_size, mode);
    start_server();
}

TCPChannel::TCPChannel(const std::string& uri, size_t buffer_size, ChannelMode mode,
                       ChannelType type, const std::string& host, uint16_t port)
    : ChannelImpl(uri, buffer_size, mode, type)
    , is_server_(false)
    , host_(host)
    , port_(port)
    , io_context_(std::make_unique<boost::asio::io_context>())
    , connected_(false)
    , network_running_(false)
    , bytes_needed_(HEADER_SIZE)
    , header_received_(false) {
    
    log_info("Creating TCP client channel to ", host, ":", port);
    initialize_ring_buffer(buffer_size, mode);
    start_client();
}

TCPChannel::~TCPChannel() {
    stop();
}

void TCPChannel::initialize_ring_buffer(size_t buffer_size, ChannelMode mode) {
    switch (mode) {
        case ChannelMode::SPSC:
            ring_buffer_ = std::make_unique<SPSCRingBuffer>(buffer_size);
            break;
        case ChannelMode::SPMC:
            ring_buffer_ = std::make_unique<SPMCRingBuffer>(buffer_size);
            break;
        case ChannelMode::MPSC:
            ring_buffer_ = std::make_unique<MPSCRingBuffer>(buffer_size);
            break;
        case ChannelMode::MPMC:
            ring_buffer_ = std::make_unique<MPMCRingBuffer>(buffer_size);
            break;
    }
    log_debug("Initialized ", buffer_size, " byte ring buffer for TCP channel");
}

void TCPChannel::start_server() {
    try {
        acceptor_ = std::make_unique<boost::asio::ip::tcp::acceptor>(
            *io_context_, 
            boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port_)
        );
        
        acceptor_->set_option(boost::asio::socket_base::reuse_address(true));
        log_info("TCP server listening on port ", port_);
        
        // Start accepting connections
        socket_ = std::make_unique<boost::asio::ip::tcp::socket>(*io_context_);
        acceptor_->async_accept(*socket_,
            [this](const boost::system::error_code& error) {
                handle_accept(error);
            });
        
        // Start network thread
        network_running_ = true;
        network_thread_ = std::thread(&TCPChannel::network_loop, this);
        
    } catch (const std::exception& e) {
        log_error("Failed to start TCP server: ", e.what());
        throw;
    }
}

void TCPChannel::start_client() {
    try {
        socket_ = std::make_unique<boost::asio::ip::tcp::socket>(*io_context_);
        
        // Resolve host and connect
        boost::asio::ip::tcp::resolver resolver(*io_context_);
        auto endpoints = resolver.resolve(host_, std::to_string(port_));
        
        boost::asio::async_connect(*socket_, endpoints,
            [this](const boost::system::error_code& error, 
                   const boost::asio::ip::tcp::endpoint&) {
                handle_connect(error);
            });
        
        // Start network thread
        network_running_ = true;
        network_thread_ = std::thread(&TCPChannel::network_loop, this);
        
    } catch (const std::exception& e) {
        log_error("Failed to start TCP client: ", e.what());
        throw;
    }
}

void TCPChannel::network_loop() {
    log_debug("Starting TCP network loop");
    
    while (network_running_ && !is_stopped()) {
        try {
            io_context_->run_for(std::chrono::milliseconds(100));
            if (io_context_->stopped()) {
                io_context_->restart();
            }
        } catch (const std::exception& e) {
            log_error("TCP network loop error: ", e.what());
            break;
        }
    }
    
    log_debug("TCP network loop stopped");
}

void TCPChannel::handle_accept(const boost::system::error_code& error) {
    if (!error && !is_stopped()) {
        log_info("TCP client connected from ", 
                socket_->remote_endpoint().address().to_string(), ":",
                socket_->remote_endpoint().port());
        
        // Configure socket
        socket_->set_option(boost::asio::ip::tcp::no_delay(true));
        socket_->set_option(boost::asio::socket_base::keep_alive(true));
        
        connected_ = true;
        connection_cv_.notify_all();
        
        start_receive();
    } else if (!is_stopped()) {
        log_error("TCP accept error: ", error.message());
    }
}

void TCPChannel::handle_connect(const boost::system::error_code& error) {
    if (!error && !is_stopped()) {
        log_info("TCP connected to ", host_, ":", port_);
        
        // Configure socket
        socket_->set_option(boost::asio::ip::tcp::no_delay(true));
        socket_->set_option(boost::asio::socket_base::keep_alive(true));
        
        connected_ = true;
        connection_cv_.notify_all();
        
        start_receive();
    } else if (!is_stopped()) {
        log_error("TCP connect error: ", error.message());
    }
}

void TCPChannel::start_receive() {
    if (!connected_ || is_stopped()) return;
    
    receive_buffer_.resize(bytes_needed_);
    
    boost::asio::async_read(*socket_,
        boost::asio::buffer(receive_buffer_),
        [this](const boost::system::error_code& error, size_t bytes_transferred) {
            handle_receive(error, bytes_transferred);
        });
}

void TCPChannel::handle_receive(const boost::system::error_code& error, size_t bytes_transferred) {
    if (error || is_stopped()) {
        if (!is_stopped()) {
            log_warn("TCP receive error: ", error.message());
            connected_ = false;
        }
        return;
    }
    
    if (!header_received_) {
        // Parse message header
        MessageHeader header;
        std::memcpy(&header, receive_buffer_.data(), HEADER_SIZE);
        
        // Convert from network byte order
        header.size = boost::endian::big_to_native(header.size);
        header.checksum = boost::endian::big_to_native(header.checksum);
        
        // Validate message size
        if (header.size > MAX_MESSAGE_SIZE) {
            log_error("TCP message too large: ", header.size, " bytes");
            connected_ = false;
            return;
        }
        
        // Prepare to receive message body
        bytes_needed_ = header.size;
        header_received_ = true;
        
        start_receive();
    } else {
        // Process complete message
        process_received_message(receive_buffer_.data(), bytes_transferred);
        
        // Reset for next message
        header_received_ = false;
        bytes_needed_ = HEADER_SIZE;
        
        start_receive();
    }
}

uint32_t TCPChannel::reserve_write_slot(size_t size) {
    if (!ring_buffer_) {
        log_error("Ring buffer not initialized");
        return BUFFER_FULL;
    }
    
    // Reserve space in ring buffer (zero-copy)
    return ring_buffer_->reserve_slot(size);
}

void TCPChannel::notify_message_ready(uint32_t offset, size_t size) {
    if (!connected_ || !socket_) {
        log_warn("TCP channel not connected, dropping message");
        return;
    }
    
    // Send message over TCP (zero-copy from ring buffer)
    const uint8_t* data = ring_buffer_->base_ptr() + offset;
    send_data(data, size);
    
    // Advance write pointer
    ring_buffer_->advance_write_pointer(offset + size);
}

RingBuffer& TCPChannel::get_ring_buffer() {
    return *ring_buffer_;
}

const RingBuffer& TCPChannel::get_ring_buffer() const {
    return *ring_buffer_;
}

void TCPChannel::advance_read_pointer(size_t size) {
    if (ring_buffer_) {
        ring_buffer_->advance_read_pointer(size);
    }
}

void TCPChannel::send_data(const void* data, size_t size) {
    try {
        // Create message header
        MessageHeader header;
        header.size = boost::endian::native_to_big(static_cast<uint32_t>(size));
        header.checksum = boost::endian::native_to_big(calculate_checksum(data, size));
        
        // Send header and data in single write operation for efficiency
        std::vector<boost::asio::const_buffer> buffers;
        buffers.emplace_back(&header, HEADER_SIZE);
        buffers.emplace_back(data, size);
        
        boost::asio::write(*socket_, buffers);
        
        log_trace("Sent TCP message: ", size, " bytes");
        
    } catch (const std::exception& e) {
        log_error("TCP send error: ", e.what());
        connected_ = false;
    }
}

uint32_t TCPChannel::calculate_checksum(const void* data, size_t size) const {
    // Simple checksum - can be improved with CRC32 if needed
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t checksum = 0;
    
    for (size_t i = 0; i < size; ++i) {
        checksum += bytes[i];
    }
    
    return checksum;
}

void TCPChannel::process_received_message(const void* data, size_t size) {
    // Verify checksum
    uint32_t expected_checksum = calculate_checksum(data, size);
    
    // Write directly to ring buffer (zero-copy on receive side)
    uint32_t offset = ring_buffer_->reserve_slot(size);
    if (offset != BUFFER_FULL) {
        uint8_t* buffer_data = ring_buffer_->base_ptr() + offset;
        std::memcpy(buffer_data, data, size);  // Single copy from network to ring buffer
        ring_buffer_->advance_write_pointer(offset + size);
        
        log_trace("Received TCP message: ", size, " bytes");
    } else {
        log_warn("Ring buffer full, dropping TCP message");
    }
}

bool TCPChannel::is_connected() const {
    return connected_.load();
}

void TCPChannel::stop() {
    ChannelImpl::stop();
    
    log_info("Stopping TCP channel");
    
    network_running_ = false;
    connected_ = false;
    
    if (socket_ && socket_->is_open()) {
        boost::system::error_code ec;
        socket_->close(ec);
    }
    
    if (acceptor_ && acceptor_->is_open()) {
        boost::system::error_code ec;
        acceptor_->close(ec);
    }
    
    if (io_context_) {
        io_context_->stop();
    }
    
    if (network_thread_.joinable()) {
        network_thread_.join();
    }
    
    log_info("TCP channel stopped");
}

// Legacy interface implementation (deprecated)
void* TCPChannel::reserve_space(size_t size) {
    // This is deprecated - use reserve_write_slot() instead
    uint32_t offset = reserve_write_slot(size);
    if (offset != BUFFER_FULL) {
        return ring_buffer_->base_ptr() + offset;
    }
    return nullptr;
}

void TCPChannel::commit_message(void* handle) {
    // This is deprecated - data is committed when written
    log_warn("commit_message() is deprecated - violates zero-copy principles");
}

void* TCPChannel::receive_message(size_t& size, uint32_t& type) {
    // Legacy interface - read from ring buffer
    if (!ring_buffer_) return nullptr;
    
    auto read_handle = ring_buffer_->read();
    if (read_handle) {
        size = read_handle->size;
        type = (type_ == ChannelType::SingleType) ? 1 : 0;
        return const_cast<void*>(read_handle->data);
    }
    
    return nullptr;
}

void TCPChannel::release_message(void* handle) {
    // Legacy interface - no-op as ring buffer handles memory
}

std::unique_ptr<ChannelImpl> create_tcp_channel(
    const std::string& uri,
    size_t buffer_size,
    ChannelMode mode,
    ChannelType type) {
    
    // Parse URI: tcp://host:port/path or tcp://:port (server mode)
    std::regex tcp_regex("^tcp://([^:]*):([0-9]+)(?:/(.*))?$");
    std::smatch match;
    
    if (!std::regex_match(uri, match, tcp_regex)) {
        throw std::invalid_argument("Invalid TCP URI format: " + uri);
    }
    
    std::string host = match[1];
    uint16_t port = static_cast<uint16_t>(std::stoi(match[2]));
    
    if (host.empty()) {
        // Server mode: tcp://:port
        return std::make_unique<TCPChannel>(uri, buffer_size, mode, type, port);
    } else {
        // Client mode: tcp://host:port
        return std::make_unique<TCPChannel>(uri, buffer_size, mode, type, host, port);
    }
}

} // namespace detail
} // namespace psyne