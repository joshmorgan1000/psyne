#include "udp_multicast_channel.hpp"
#include "../memory/ring_buffer_impl.hpp"
#include "../utils/checksum.hpp"
#include <regex>
#include <iostream>
#include <chrono>
#include <cstring>

namespace psyne {
namespace detail {

// Magic number for UDP multicast messages
static constexpr uint32_t MULTICAST_MAGIC = 0x50594E45; // "PYNE"

UDPMulticastChannel::UDPMulticastChannel(const std::string& uri, size_t buffer_size,
                                        ChannelMode mode, ChannelType type,
                                        MulticastRole role,
                                        const compression::CompressionConfig& compression_config,
                                        const std::string& interface_address)
    : ChannelImpl(uri, buffer_size, mode, type)
    , role_(role)
    , joined_group_(false)
    , stopping_(false)
    , sequence_number_(0)
    , interface_address_(interface_address)
    , ttl_(1)
    , loopback_enabled_(false)
    , compression_manager_(compression_config) {
    
    parse_uri(uri);
    setup_socket();
    
    // Start IO thread
    io_thread_ = std::thread([this]() { run_io_service(); });
    
    // For subscribers, automatically join the multicast group
    if (role_ == MulticastRole::Subscriber) {
        join_group();
        start_receive();
    } else {
        // For publishers, enable loopback for local testing
        set_loopback(true);
    }
}

UDPMulticastChannel::~UDPMulticastChannel() {
    stopping_.store(true);
    
    if (joined_group_) {
        leave_group();
    }
    
    if (socket_) {
        socket_->close();
    }
    
    io_context_.stop();
    
    if (io_thread_.joinable()) {
        io_thread_.join();
    }
}

void UDPMulticastChannel::parse_uri(const std::string& uri) {
    // Parse URI: udp://multicast_address:port
    std::regex uri_regex("^udp://([^:]+):([0-9]+)$");
    std::smatch match;
    
    if (!std::regex_match(uri, match, uri_regex)) {
        throw std::invalid_argument("Invalid UDP multicast URI format. Use udp://multicast_address:port");
    }
    
    std::string address = match[1];
    uint16_t port = static_cast<uint16_t>(std::stoi(match[2]));
    
    multicast_endpoint_ = udp::endpoint(asio::ip::make_address(address), port);
    
    // Validate that it's a multicast address
    auto addr = multicast_endpoint_.address();
    if (!addr.is_multicast()) {
        throw std::invalid_argument("Address must be a valid multicast address (224.0.0.0-239.255.255.255)");
    }
}

void UDPMulticastChannel::setup_socket() {
    socket_ = std::make_unique<udp::socket>(io_context_);
    
    boost::system::error_code ec;
    socket_->open(udp::v4(), ec);
    if (ec) {
        throw std::runtime_error("Failed to open UDP socket: " + ec.message());
    }
    
    // Set socket options
    socket_->set_option(asio::socket_base::reuse_address(true), ec);
    if (ec) {
        std::cerr << "Warning: Failed to set reuse_address: " << ec.message() << std::endl;
    }
    
    // For subscribers, bind to the multicast port
    if (role_ == MulticastRole::Subscriber) {
        udp::endpoint bind_endpoint(udp::v4(), multicast_endpoint_.port());
        socket_->bind(bind_endpoint, ec);
        if (ec) {
            throw std::runtime_error("Failed to bind to port " + 
                                   std::to_string(multicast_endpoint_.port()) + ": " + ec.message());
        }
    } else {
        // For publishers, bind to any available port
        socket_->bind(udp::endpoint(udp::v4(), 0), ec);
        if (ec) {
            throw std::runtime_error("Failed to bind publisher socket: " + ec.message());
        }
    }
    
    local_endpoint_ = socket_->local_endpoint();
}

void UDPMulticastChannel::join_group() {
    if (role_ != MulticastRole::Subscriber) {
        throw std::logic_error("Only subscribers can join multicast groups");
    }
    
    boost::system::error_code ec;
    
    if (!interface_address_.empty()) {
        auto interface_addr = asio::ip::make_address_v4(interface_address_);
        socket_->set_option(asio::ip::multicast::join_group(
            multicast_endpoint_.address().to_v4(), interface_addr), ec);
    } else {
        socket_->set_option(asio::ip::multicast::join_group(
            multicast_endpoint_.address().to_v4()), ec);
    }
    
    if (ec) {
        throw std::runtime_error("Failed to join multicast group: " + ec.message());
    }
    
    joined_group_ = true;
    std::cout << "Joined multicast group " << multicast_endpoint_.address().to_string() 
              << " on port " << multicast_endpoint_.port() << std::endl;
    std::cout << "Local socket bound to: " << socket_->local_endpoint().address().to_string() 
              << ":" << socket_->local_endpoint().port() << std::endl;
}

void UDPMulticastChannel::leave_group() {
    if (!joined_group_ || role_ != MulticastRole::Subscriber) {
        return;
    }
    
    boost::system::error_code ec;
    socket_->set_option(asio::ip::multicast::leave_group(
        multicast_endpoint_.address().to_v4()), ec);
    
    if (ec) {
        std::cerr << "Warning: Failed to leave multicast group: " << ec.message() << std::endl;
    }
    
    joined_group_ = false;
}

void UDPMulticastChannel::set_ttl(int ttl) {
    if (role_ != MulticastRole::Publisher) {
        throw std::logic_error("TTL can only be set for publishers");
    }
    
    ttl_ = ttl;
    boost::system::error_code ec;
    socket_->set_option(asio::ip::multicast::hops(ttl), ec);
    if (ec) {
        throw std::runtime_error("Failed to set TTL: " + ec.message());
    }
}

void UDPMulticastChannel::set_loopback(bool enable) {
    loopback_enabled_ = enable;
    boost::system::error_code ec;
    socket_->set_option(asio::ip::multicast::enable_loopback(enable), ec);
    if (ec) {
        throw std::runtime_error("Failed to set loopback: " + ec.message());
    }
}

void* UDPMulticastChannel::reserve_space(size_t size) {
    if (role_ != MulticastRole::Publisher) {
        return nullptr; // Only publishers can send
    }
    
    // Allocate space for: UDPMulticastHeader + SlabHeader + user data
    size_t total_size = sizeof(UDPMulticastHeader) + sizeof(SlabHeader) + size;
    send_buffer_.resize(total_size);
    
    // Create a SlabHeader in the buffer after the UDPMulticastHeader
    auto* slab_header = reinterpret_cast<SlabHeader*>(send_buffer_.data() + sizeof(UDPMulticastHeader));
    slab_header->len = 0; // Will be set when message is committed
    slab_header->reserved = 0;
    
    return slab_header; // Return SlabHeader pointer as expected by Message template
}

void UDPMulticastChannel::commit_message(void* handle) {
    std::cout << "DEBUG: commit_message called, role=" << (int)role_ << ", handle=" << handle << std::endl;
    
    if (role_ != MulticastRole::Publisher || !handle) {
        std::cout << "DEBUG: commit_message early return" << std::endl;
        return;
    }
    
    // Handle is a SlabHeader*
    auto* slab_header = static_cast<SlabHeader*>(handle);
    uint8_t* payload_start = static_cast<uint8_t*>(slab_header->data());
    size_t payload_size = slab_header->len;
    
    std::cout << "DEBUG: payload_size=" << payload_size << std::endl;
    
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
    std::vector<uint8_t> final_buffer(sizeof(UDPMulticastHeader) + final_payload_size);
    
    // Fill in header
    auto* header = reinterpret_cast<UDPMulticastHeader*>(final_buffer.data());
    header->magic = MULTICAST_MAGIC;
    header->sequence_number = sequence_number_.fetch_add(1);
    header->message_length = sizeof(UDPMulticastHeader) + final_payload_size;
    header->message_type = 1; // TODO: Get from message
    header->timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    header->checksum = calculate_checksum(final_payload, final_payload_size);
    header->original_size = compressed ? static_cast<uint32_t>(payload_size) : 0;
    header->compression_type = static_cast<uint8_t>(compression_manager_.config().type);
    header->flags = 0;
    header->reserved = 0;
    
    // Copy payload
    std::memcpy(final_buffer.data() + sizeof(UDPMulticastHeader), 
                final_payload, final_payload_size);
    
    // Add to send queue
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        PendingMessage msg;
        msg.data = std::move(final_buffer);
        msg.type = header->message_type;
        msg.sequence_number = header->sequence_number;
        msg.timestamp = header->timestamp;
        send_queue_.push(std::move(msg));
        send_buffer_.clear();
    }
    
    // Trigger async send
    asio::post(io_context_, [this]() { start_send(); });
}

void* UDPMulticastChannel::receive_message(size_t& size, uint32_t& type) {
    if (role_ != MulticastRole::Subscriber) {
        return nullptr; // Only subscribers can receive
    }
    
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
    
    // Extract header and payload
    auto* header = reinterpret_cast<const UDPMulticastHeader*>(temp_recv_buffer_.data());
    uint8_t* payload = temp_recv_buffer_.data() + sizeof(UDPMulticastHeader);
    size_t payload_size = temp_recv_buffer_.size() - sizeof(UDPMulticastHeader);
    
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
        temp_recv_buffer_.resize(sizeof(UDPMulticastHeader) + decompressed_size);
        std::memcpy(temp_recv_buffer_.data() + sizeof(UDPMulticastHeader),
                    compression_buffer_.data(), decompressed_size);
        
        size = decompressed_size;
    } else {
        // Message is not compressed
        size = payload_size;
    }
    
    return temp_recv_buffer_.data() + sizeof(UDPMulticastHeader);
}

void UDPMulticastChannel::release_message(void* handle) {
    // Nothing to do - message is in temp_recv_buffer_
}

void UDPMulticastChannel::start_receive() {
    if (role_ != MulticastRole::Subscriber || stopping_) {
        return;
    }
    
    recv_buffer_.resize(65536); // Maximum UDP packet size
    
    socket_->async_receive_from(
        asio::buffer(recv_buffer_),
        sender_endpoint_tmp_,
        [this](const boost::system::error_code& error, size_t bytes_transferred) {
            handle_receive(error, bytes_transferred, sender_endpoint_tmp_);
        });
}

void UDPMulticastChannel::handle_receive(const boost::system::error_code& error, 
                                        size_t bytes_transferred, 
                                        udp::endpoint sender_endpoint) {
    std::cout << "DEBUG: handle_receive called, bytes: " << bytes_transferred << std::endl;
    
    if (error) {
        if (!stopping_) {
            std::cerr << "UDP receive error: " << error.message() << std::endl;
        }
        return;
    }
    
    if (bytes_transferred < sizeof(UDPMulticastHeader)) {
        update_stats_dropped();
        start_receive(); // Continue receiving
        return;
    }
    
    // Validate message
    auto* header = reinterpret_cast<const UDPMulticastHeader*>(recv_buffer_.data());
    if (!validate_message(*header, recv_buffer_.data() + sizeof(UDPMulticastHeader), 
                         bytes_transferred - sizeof(UDPMulticastHeader))) {
        update_stats_dropped();
        start_receive(); // Continue receiving
        return;
    }
    
    // Check sequence number for ordering
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        if (stats_.last_sequence_number != 0 && 
            header->sequence_number != stats_.last_sequence_number + 1) {
            update_stats_sequence_error();
        }
        stats_.last_sequence_number = header->sequence_number;
    }
    
    // Add to receive queue
    {
        std::lock_guard<std::mutex> lock(recv_mutex_);
        PendingMessage msg;
        msg.data.resize(bytes_transferred);
        std::memcpy(msg.data.data(), recv_buffer_.data(), bytes_transferred);
        msg.type = header->message_type;
        msg.sequence_number = header->sequence_number;
        msg.timestamp = header->timestamp;
        recv_queue_.push(std::move(msg));
        
        // Limit queue size
        if (recv_queue_.size() > 1000) {
            recv_queue_.pop();
            update_stats_dropped();
        }
    }
    
    update_stats_received(bytes_transferred);
    recv_cv_.notify_one();
    
    // Continue receiving
    start_receive();
}

void UDPMulticastChannel::start_send() {
    if (role_ != MulticastRole::Publisher || stopping_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(send_mutex_);
    if (send_queue_.empty()) {
        return;
    }
    
    auto& msg = send_queue_.front();
    
    std::cout << "DEBUG: Sending " << msg.data.size() << " bytes to " 
              << multicast_endpoint_.address().to_string() << ":" << multicast_endpoint_.port() << std::endl;
    
    socket_->async_send_to(
        asio::buffer(msg.data),
        multicast_endpoint_,
        [this](const boost::system::error_code& error, size_t bytes_transferred) {
            handle_send(error, bytes_transferred);
        });
}

void UDPMulticastChannel::handle_send(const boost::system::error_code& error, 
                                     size_t bytes_transferred) {
    std::cout << "DEBUG: handle_send called, bytes: " << bytes_transferred;
    if (error) {
        std::cout << ", error: " << error.message() << std::endl;
        std::cerr << "UDP send error: " << error.message() << std::endl;
        return;
    } else {
        std::cout << " (success)" << std::endl;
    }
    
    update_stats_sent(bytes_transferred);
    
    // Remove sent message and continue sending
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        if (!send_queue_.empty()) {
            send_queue_.pop();
        }
        
        // Send next message if available
        if (!send_queue_.empty()) {
            asio::post(io_context_, [this]() { start_send(); });
        }
    }
}

void UDPMulticastChannel::run_io_service() {
    try {
        io_context_.run();
    } catch (const std::exception& e) {
        if (!stopping_) {
            std::cerr << "IO service error: " << e.what() << std::endl;
        }
    }
}

uint64_t UDPMulticastChannel::calculate_checksum(const uint8_t* data, size_t size) {
    return utils::udp::calculate_checksum(data, size);
}

bool UDPMulticastChannel::validate_message(const UDPMulticastHeader& header, 
                                          const uint8_t* payload, 
                                          size_t payload_size) {
    // Check magic number
    if (header.magic != MULTICAST_MAGIC) {
        return false;
    }
    
    // Check message length
    if (header.message_length != sizeof(UDPMulticastHeader) + payload_size) {
        return false;
    }
    
    // Verify checksum
    uint64_t computed_checksum = calculate_checksum(payload, payload_size);
    if (header.checksum != computed_checksum) {
        return false;
    }
    
    return true;
}

void UDPMulticastChannel::update_stats_sent(size_t bytes) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.packets_sent++;
    stats_.bytes_sent += bytes;
}

void UDPMulticastChannel::update_stats_received(size_t bytes) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.packets_received++;
    stats_.bytes_received += bytes;
}

void UDPMulticastChannel::update_stats_dropped() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.packets_dropped++;
}

void UDPMulticastChannel::update_stats_sequence_error() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.sequence_errors++;
}

// Factory function
std::unique_ptr<ChannelImpl> create_udp_multicast_channel(
    const std::string& uri, size_t buffer_size,
    ChannelMode mode, ChannelType type,
    MulticastRole role,
    const compression::CompressionConfig& compression_config,
    const std::string& interface_address) {
    
    return std::make_unique<UDPMulticastChannel>(uri, buffer_size, mode, type, 
                                                role, compression_config, interface_address);
}

} // namespace detail
} // namespace psyne