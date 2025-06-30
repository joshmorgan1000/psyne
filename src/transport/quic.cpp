#include "quic.hpp"
#include <boost/asio.hpp>
#include <iostream>
#include <random>

namespace psyne {
namespace transport {

namespace asio = boost::asio;
using udp = asio::ip::udp;

// QUICStream implementation
QUICStream::QUICStream(uint64_t stream_id, QUICStreamDirection direction)
    : stream_id_(stream_id), direction_(direction) {}

QUICStream::~QUICStream() {
    close();
}

ssize_t QUICStream::send(const void* data, size_t size) {
    if (!is_readable()) return -1;
    
    std::lock_guard<std::mutex> lock(send_mutex_);
    
    // Check flow control
    uint64_t available = get_available_send_window();
    if (available == 0) return -1;
    
    size_t to_send = std::min(size, static_cast<size_t>(available));
    
    // Queue data for sending
    std::vector<uint8_t> buffer(to_send);
    std::memcpy(buffer.data(), data, to_send);
    send_queue_.push(std::move(buffer));
    
    stream_data_sent_ += to_send;
    
    return static_cast<ssize_t>(to_send);
}

ssize_t QUICStream::receive(void* buffer, size_t buffer_size) {
    if (!is_readable()) return -1;
    
    std::lock_guard<std::mutex> lock(recv_mutex_);
    
    if (recv_queue_.empty()) {
        return recv_finished_ ? 0 : -1; // EOF or would block
    }
    
    auto& front = recv_queue_.front();
    size_t to_copy = std::min(buffer_size, front.size());
    std::memcpy(buffer, front.data(), to_copy);
    
    if (to_copy < front.size()) {
        // Partial read, update front buffer
        front.erase(front.begin(), front.begin() + to_copy);
    } else {
        // Full read, remove buffer
        recv_queue_.pop();
    }
    
    stream_data_received_ += to_copy;
    
    return static_cast<ssize_t>(to_copy);
}

bool QUICStream::try_send(const void* data, size_t size) {
    return send(data, size) == static_cast<ssize_t>(size);
}

bool QUICStream::try_receive(void* buffer, size_t buffer_size, size_t* received_size) {
    ssize_t result = receive(buffer, buffer_size);
    if (result >= 0) {
        if (received_size) *received_size = result;
        return true;
    }
    return false;
}

void QUICStream::close() {
    if (state_ != QUICStreamState::CLOSED) {
        state_ = QUICStreamState::CLOSED;
        send_finished_ = true;
        recv_finished_ = true;
    }
}

void QUICStream::reset(uint64_t error_code) {
    state_ = QUICStreamState::RESET;
    close();
}

bool QUICStream::is_readable() const {
    auto s = state_.load();
    return s == QUICStreamState::OPEN || s == QUICStreamState::HALF_CLOSED_LOCAL;
}

bool QUICStream::is_writable() const {
    auto s = state_.load();
    return s == QUICStreamState::OPEN || s == QUICStreamState::HALF_CLOSED_REMOTE;
}

void QUICStream::set_max_stream_data(uint64_t max_data) {
    max_stream_data_recv_ = max_data;
}

uint64_t QUICStream::get_available_send_window() const {
    uint64_t max_data = max_stream_data_send_.load();
    uint64_t sent = stream_data_sent_.load();
    return (max_data > sent) ? (max_data - sent) : 0;
}

uint64_t QUICStream::get_available_receive_window() const {
    uint64_t max_data = max_stream_data_recv_.load();
    uint64_t received = stream_data_received_.load();
    return (max_data > received) ? (max_data - received) : 0;
}

// QUICConnection implementation
QUICConnection::QUICConnection(bool is_server, const QUICConfig& config)
    : config_(config), is_server_(is_server), 
      congestion_window_(config.initial_congestion_window * config.max_udp_payload_size) {
    
    // Initialize connection IDs
    local_connection_id_ = generate_connection_id();
    
    // Set initial flow control limits
    max_data_send_ = config.initial_max_data;
    max_data_recv_ = config.initial_max_data;
}

QUICConnection::~QUICConnection() {
    close();
    running_ = false;
    if (connection_thread_.joinable()) {
        connection_thread_.join();
    }
}

bool QUICConnection::connect(const std::string& remote_address, uint16_t remote_port) {
    if (is_server_) return false;
    
    remote_address_ = remote_address;
    remote_port_ = remote_port;
    
    // Start connection thread
    running_ = true;
    connection_thread_ = std::thread(&QUICConnection::connection_loop, this);
    
    // Initiate handshake
    state_ = QUICConnectionState::HANDSHAKE;
    
    // Wait for connection to be established
    std::unique_lock<std::mutex> lock(streams_mutex_);
    connection_cv_.wait_for(lock, std::chrono::seconds(5), [this]() {
        return state_ == QUICConnectionState::ESTABLISHED;
    });
    
    return state_ == QUICConnectionState::ESTABLISHED;
}

bool QUICConnection::listen(uint16_t local_port) {
    if (!is_server_) return false;
    
    local_port_ = local_port;
    
    // Start connection thread
    running_ = true;
    connection_thread_ = std::thread(&QUICConnection::connection_loop, this);
    
    state_ = QUICConnectionState::INITIAL;
    
    return true;
}

void QUICConnection::close(uint64_t error_code, const std::string& reason) {
    if (state_ == QUICConnectionState::CLOSED) return;
    
    state_ = QUICConnectionState::CLOSING;
    
    // Notify close callback
    if (close_callback_) {
        close_callback_(error_code, reason);
    }
    
    // Close all streams
    std::lock_guard<std::mutex> lock(streams_mutex_);
    for (auto& [id, stream] : streams_) {
        stream->close();
    }
    streams_.clear();
    
    state_ = QUICConnectionState::CLOSED;
}

std::shared_ptr<QUICStream> QUICConnection::create_stream(QUICStreamDirection direction) {
    if (!is_connected()) return nullptr;
    
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    // Calculate next stream ID based on direction and role
    uint64_t stream_id = next_stream_id_;
    if (is_server_) {
        // Server initiated streams: 1, 5, 9, ... (bidi) or 3, 7, 11, ... (uni)
        stream_id = (direction == QUICStreamDirection::BIDIRECTIONAL) ? 1 : 3;
        stream_id += (next_stream_id_ * 4);
    } else {
        // Client initiated streams: 0, 4, 8, ... (bidi) or 2, 6, 10, ... (uni)
        stream_id = (direction == QUICStreamDirection::BIDIRECTIONAL) ? 0 : 2;
        stream_id += (next_stream_id_ * 4);
    }
    next_stream_id_++;
    
    auto stream = std::make_shared<QUICStream>(stream_id, direction);
    streams_[stream_id] = stream;
    
    stats_.streams_created++;
    
    return stream;
}

std::shared_ptr<QUICStream> QUICConnection::accept_stream() {
    std::unique_lock<std::mutex> lock(streams_mutex_);
    
    stream_cv_.wait(lock, [this]() {
        return !incoming_streams_.empty() || state_ != QUICConnectionState::ESTABLISHED;
    });
    
    if (incoming_streams_.empty()) {
        return nullptr;
    }
    
    auto stream = incoming_streams_.front();
    incoming_streams_.pop();
    
    return stream;
}

std::shared_ptr<QUICStream> QUICConnection::get_stream(uint64_t stream_id) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    auto it = streams_.find(stream_id);
    return (it != streams_.end()) ? it->second : nullptr;
}

bool QUICConnection::supports_0rtt() const {
    return config_.enable_0rtt && stats_.using_0rtt;
}

bool QUICConnection::supports_migration() const {
    return config_.enable_migration;
}

void QUICConnection::migrate_to_path(const std::string& new_local_address,
                                    const std::string& new_remote_address) {
    if (!supports_migration()) return;
    
    // Path migration logic would go here
    local_address_ = new_local_address;
    remote_address_ = new_remote_address;
    
    stats_.connection_migrations++;
}

QUICStats QUICConnection::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    auto stats = stats_;
    stats.rtt_ms = smoothed_rtt_ms_.load();
    stats.rtt_variance_ms = rtt_variance_ms_.load();
    stats.congestion_window = congestion_window_.load();
    return stats;
}

void QUICConnection::set_stream_callback(std::function<void(std::shared_ptr<QUICStream>)> callback) {
    stream_callback_ = callback;
}

void QUICConnection::set_connection_close_callback(
    std::function<void(uint64_t error_code, const std::string& reason)> callback) {
    close_callback_ = callback;
}

void QUICConnection::connection_loop() {
    // Simplified connection loop
    if (!is_server_) {
        // Client: simulate handshake completion
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        state_ = QUICConnectionState::ESTABLISHED;
        connection_cv_.notify_all();
    }
    
    while (running_ && state_ != QUICConnectionState::CLOSED) {
        // Process packets, handle retransmissions, etc.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// QUICServer implementation
QUICServer::QUICServer(uint16_t port, const QUICConfig& config)
    : port_(port), config_(config) {}

QUICServer::~QUICServer() {
    stop();
}

bool QUICServer::start() {
    if (running_.exchange(true)) {
        return false; // Already running
    }
    
    acceptor_thread_ = std::thread(&QUICServer::acceptor_loop, this);
    
    std::cout << "QUIC server started on port " << port_ << std::endl;
    return true;
}

void QUICServer::stop() {
    if (!running_.exchange(false)) {
        return; // Already stopped
    }
    
    if (acceptor_thread_.joinable()) {
        acceptor_thread_.join();
    }
    
    // Close all connections
    std::lock_guard<std::mutex> lock(connections_mutex_);
    for (auto& conn : connections_) {
        conn->close();
    }
    connections_.clear();
}

std::shared_ptr<QUICConnection> QUICServer::accept() {
    // Simulate accepting a connection
    auto connection = std::make_shared<QUICConnection>(true, config_);
    
    if (connection->listen(port_)) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        connections_.push_back(connection);
        
        // Simulate connection establishment
        connection->set_state(QUICConnectionState::ESTABLISHED);
        
        if (connection_handler_) {
            connection_handler_(connection);
        }
        
        return connection;
    }
    
    return nullptr;
}

size_t QUICServer::get_connection_count() const {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    return connections_.size();
}

void QUICServer::set_connection_handler(
    std::function<void(std::shared_ptr<QUICConnection>)> handler) {
    connection_handler_ = handler;
}

void QUICServer::acceptor_loop() {
    while (running_) {
        // In a real implementation, this would listen for UDP packets
        // and create new connections as needed
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Factory functions
std::shared_ptr<QUICConnection> create_quic_client(const std::string& remote_address,
                                                   uint16_t remote_port,
                                                   const QUICConfig& config) {
    auto connection = std::make_shared<QUICConnection>(false, config);
    if (connection->connect(remote_address, remote_port)) {
        return connection;
    }
    return nullptr;
}

std::unique_ptr<QUICServer> create_quic_server(uint16_t port,
                                               const QUICConfig& config) {
    return std::make_unique<QUICServer>(port, config);
}

std::unique_ptr<QUICChannel> create_quic_channel(const std::string& remote_address,
                                                 uint16_t remote_port,
                                                 const QUICConfig& config) {
    auto connection = create_quic_client(remote_address, remote_port, config);
    if (connection) {
        return std::make_unique<QUICChannel>(connection);
    }
    return nullptr;
}

// Utility functions
std::vector<uint8_t> generate_connection_id(size_t length) {
    std::vector<uint8_t> id(length);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    for (auto& byte : id) {
        byte = static_cast<uint8_t>(dis(gen));
    }
    
    return id;
}

std::vector<uint8_t> encode_varint(uint64_t value) {
    std::vector<uint8_t> result;
    
    if (value < 64) {
        result.push_back(static_cast<uint8_t>(value));
    } else if (value < 16384) {
        result.push_back(static_cast<uint8_t>(0x40 | (value >> 8)));
        result.push_back(static_cast<uint8_t>(value & 0xFF));
    } else if (value < 1073741824) {
        result.push_back(static_cast<uint8_t>(0x80 | (value >> 24)));
        result.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
        result.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
        result.push_back(static_cast<uint8_t>(value & 0xFF));
    } else {
        result.push_back(static_cast<uint8_t>(0xC0 | (value >> 56)));
        result.push_back(static_cast<uint8_t>((value >> 48) & 0xFF));
        result.push_back(static_cast<uint8_t>((value >> 40) & 0xFF));
        result.push_back(static_cast<uint8_t>((value >> 32) & 0xFF));
        result.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
        result.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
        result.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
        result.push_back(static_cast<uint8_t>(value & 0xFF));
    }
    
    return result;
}

uint64_t decode_varint(const uint8_t* data, size_t* bytes_consumed) {
    uint8_t first_byte = data[0];
    uint8_t length = 1 << ((first_byte & 0xC0) >> 6);
    
    uint64_t value = first_byte & (0xFF >> (2 * (length / 2)));
    
    for (size_t i = 1; i < length; ++i) {
        value = (value << 8) | data[i];
    }
    
    if (bytes_consumed) {
        *bytes_consumed = length;
    }
    
    return value;
}

const char* get_quic_version() {
    return "QUIC/1.0 (Psyne)";
}

bool is_quic_supported() {
    return true;
}
// QUICChannel implementation
QUICChannel::QUICChannel(std::shared_ptr<QUICConnection> connection)
    : connection_(connection) {
    if (!connection || !connection->is_connected()) {
        throw std::runtime_error("Invalid or disconnected QUIC connection");
    }
    
    // Create a bidirectional stream for the channel
    stream_ = connection->create_stream(QUICStreamDirection::BIDIRECTIONAL);
    if (!stream_) {
        throw std::runtime_error("Failed to create QUIC stream");
    }
}

QUICChannel::~QUICChannel() {
    if (stream_) {
        stream_->close();
    }
}

size_t QUICChannel::send(const void* data, size_t size, uint32_t type_id) {
    if (!stream_ || !stream_->is_writable()) {
        return 0;
    }
    
    // Send type_id as header
    if (stream_->send(&type_id, sizeof(type_id)) != sizeof(type_id)) {
        return 0;
    }
    
    // Send data
    ssize_t sent = stream_->send(data, size);
    return (sent > 0) ? static_cast<size_t>(sent) : 0;
}

size_t QUICChannel::receive(void* buffer, size_t buffer_size, uint32_t* type_id) {
    if (!stream_ || !stream_->is_readable()) {
        return 0;
    }
    
    // Receive type_id header
    uint32_t msg_type = 0;
    if (stream_->receive(&msg_type, sizeof(msg_type)) != sizeof(msg_type)) {
        return 0;
    }
    
    if (type_id) {
        *type_id = msg_type;
    }
    
    // Receive data
    ssize_t received = stream_->receive(buffer, buffer_size);
    return (received > 0) ? static_cast<size_t>(received) : 0;
}

bool QUICChannel::try_send(const void* data, size_t size, uint32_t type_id) {
    return send(data, size, type_id) == size;
}

bool QUICChannel::try_receive(void* buffer, size_t buffer_size,
                             size_t* received_size, uint32_t* type_id) {
    size_t result = receive(buffer, buffer_size, type_id);
    if (result > 0) {
        if (received_size) *received_size = result;
        return true;
    }
    return false;
}

} // namespace transport
} // namespace psyne