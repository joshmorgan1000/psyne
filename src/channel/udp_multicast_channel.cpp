#include "udp_multicast_channel.hpp"
#include "../memory/ring_buffer_impl.hpp"
#include "../utils/checksum.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <regex>

namespace psyne {
namespace detail {

// Magic number for UDP multicast messages
static constexpr uint32_t MULTICAST_MAGIC = 0x50594E45; // "PYNE"

UDPMulticastChannel::UDPMulticastChannel(
    const std::string &uri, size_t buffer_size, ChannelMode mode,
    ChannelType type, MulticastRole role,
    const compression::CompressionConfig &compression_config,
    const std::string &interface_address)
    : ChannelImpl(uri, buffer_size, mode, type), role_(role),
      joined_group_(false), stopping_(false), sequence_number_(0),
      interface_address_(interface_address), ttl_(1), loopback_enabled_(false),
      compression_manager_(compression_config) {
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

void UDPMulticastChannel::parse_uri(const std::string &uri) {
    // Parse URI: udp://multicast_address:port
    std::regex uri_regex("^udp://([^:]+):([0-9]+)$");
    std::smatch match;

    if (!std::regex_match(uri, match, uri_regex)) {
        throw std::invalid_argument("Invalid UDP multicast URI format. Use "
                                    "udp://multicast_address:port");
    }

    std::string address = match[1];
    uint16_t port = static_cast<uint16_t>(std::stoi(match[2]));

    multicast_endpoint_ = udp::endpoint(asio::ip::make_address(address), port);

    // Validate that it's a multicast address
    auto addr = multicast_endpoint_.address();
    if (!addr.is_multicast()) {
        throw std::invalid_argument("Address must be a valid multicast address "
                                    "(224.0.0.0-239.255.255.255)");
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
        std::cerr << "Warning: Failed to set reuse_address: " << ec.message()
                  << std::endl;
    }

    // For subscribers, bind to the multicast port
    if (role_ == MulticastRole::Subscriber) {
        udp::endpoint bind_endpoint(udp::v4(), multicast_endpoint_.port());
        socket_->bind(bind_endpoint, ec);
        if (ec) {
            throw std::runtime_error(
                "Failed to bind to port " +
                std::to_string(multicast_endpoint_.port()) + ": " +
                ec.message());
        }
    } else {
        // For publishers, bind to any available port
        socket_->bind(udp::endpoint(udp::v4(), 0), ec);
        if (ec) {
            throw std::runtime_error("Failed to bind publisher socket: " +
                                     ec.message());
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
        socket_->set_option(
            asio::ip::multicast::join_group(
                multicast_endpoint_.address().to_v4(), interface_addr),
            ec);
    } else {
        socket_->set_option(asio::ip::multicast::join_group(
                                multicast_endpoint_.address().to_v4()),
                            ec);
    }

    if (ec) {
        throw std::runtime_error("Failed to join multicast group: " +
                                 ec.message());
    }

    joined_group_ = true;
    std::cout << "Joined multicast group "
              << multicast_endpoint_.address().to_string() << " on port "
              << multicast_endpoint_.port() << std::endl;
    std::cout << "Local socket bound to: "
              << socket_->local_endpoint().address().to_string() << ":"
              << socket_->local_endpoint().port() << std::endl;
}

void UDPMulticastChannel::leave_group() {
    if (!joined_group_ || role_ != MulticastRole::Subscriber) {
        return;
    }

    boost::system::error_code ec;
    socket_->set_option(
        asio::ip::multicast::leave_group(multicast_endpoint_.address().to_v4()),
        ec);

    if (ec) {
        std::cerr << "Warning: Failed to leave multicast group: "
                  << ec.message() << std::endl;
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

uint32_t UDPMulticastChannel::reserve_write_slot(size_t size) noexcept {
    if (role_ != MulticastRole::Publisher) {
        return BUFFER_FULL;
    }
    
    // Reserve space directly in ring buffer for zero-copy
    auto& ring_buffer = get_ring_buffer();
    return ring_buffer.reserve_write_space(size);
}

void UDPMulticastChannel::notify_message_ready(uint32_t offset, size_t size) noexcept {
    if (role_ != MulticastRole::Publisher) {
        return;
    }
    
    // Message is already in ring buffer at offset
    // Stream directly from ring buffer for zero-copy UDP multicast
    auto& ring_buffer = get_ring_buffer();
    auto data_span = ring_buffer.get_read_span(offset, size);
    
    // Prepare UDP multicast header
    UDPMulticastHeader header{};
    header.magic = MULTICAST_MAGIC;
    header.sequence_number = sequence_number_.fetch_add(1);
    header.message_length = sizeof(UDPMulticastHeader) + size;
    header.message_type = 1;
    header.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    header.checksum = calculate_checksum(data_span.data(), data_span.size());
    header.original_size = 0; // No compression in zero-copy path
    header.compression_type = 0;
    header.flags = 0;
    header.reserved = 0;
    
    // Queue zero-copy multicast send
    asio::post(io_context_, [this, header, offset, size]() {
        start_zero_copy_multicast(header, offset, size);
    });
}

std::span<uint8_t> UDPMulticastChannel::get_write_span(size_t size) noexcept {
    uint32_t offset = reserve_write_slot(size);
    if (offset == BUFFER_FULL) {
        return {};
    }
    
    auto& ring_buffer = get_ring_buffer();
    return ring_buffer.get_write_span(offset, size);
}

std::span<const uint8_t> UDPMulticastChannel::buffer_span() const noexcept {
    auto& ring_buffer = get_ring_buffer();
    return ring_buffer.available_read_span();
}

void UDPMulticastChannel::advance_read_pointer(size_t size) noexcept {
    auto& ring_buffer = get_ring_buffer();
    ring_buffer.advance_read_pointer(size);
}

void UDPMulticastChannel::start_zero_copy_multicast(const UDPMulticastHeader& header, uint32_t offset, size_t size) {
    if (role_ != MulticastRole::Publisher || stopping_) {
        return;
    }
    
    // Get data directly from ring buffer
    auto& ring_buffer = get_ring_buffer();
    auto data_span = ring_buffer.get_read_span(offset, size);
    
    // Create scatter-gather buffer for efficient UDP multicast (header + data)
    std::array<asio::const_buffer, 2> buffers = {
        asio::buffer(&header, sizeof(header)),
        asio::buffer(data_span.data(), data_span.size())
    };
    
    socket_->async_send_to(buffers, multicast_endpoint_,
        [this, offset, size](const boost::system::error_code& error, size_t bytes_transferred) {
            if (!error) {
                // Successfully sent, advance ring buffer read position
                auto& ring_buffer = get_ring_buffer();
                ring_buffer.advance_read_pointer(size);
                update_stats_sent(bytes_transferred);
            } else {
                update_stats_dropped();
            }
        });
}

void *UDPMulticastChannel::receive_message(size_t &size, uint32_t &type) {
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
    auto *header =
        reinterpret_cast<const UDPMulticastHeader *>(temp_recv_buffer_.data());
    uint8_t *payload = temp_recv_buffer_.data() + sizeof(UDPMulticastHeader);
    size_t payload_size = temp_recv_buffer_.size() - sizeof(UDPMulticastHeader);

    // Check if decompression is needed
    if (header->original_size > 0) {
        // Message is compressed, decompress it
        compression_buffer_.resize(header->original_size);

        size_t decompressed_size = compression_manager_.decompress_message(
            payload, payload_size, compression_buffer_.data(),
            compression_buffer_.size());

        if (decompressed_size == 0 ||
            decompressed_size != header->original_size) {
            // Decompression failed
            return nullptr;
        }

        // Replace temp buffer with decompressed data
        temp_recv_buffer_.resize(sizeof(UDPMulticastHeader) +
                                 decompressed_size);
        std::memcpy(temp_recv_buffer_.data() + sizeof(UDPMulticastHeader),
                    compression_buffer_.data(), decompressed_size);

        size = decompressed_size;
    } else {
        // Message is not compressed
        size = payload_size;
    }

    return temp_recv_buffer_.data() + sizeof(UDPMulticastHeader);
}

// Legacy deprecated methods (kept for compatibility)  
void* UDPMulticastChannel::reserve_space(size_t size) {
    uint32_t offset = reserve_write_slot(size);
    if (offset == BUFFER_FULL) {
        return nullptr;
    }
    auto& ring_buffer = get_ring_buffer();
    auto span = ring_buffer.get_write_span(offset, size);
    return span.data();
}

void UDPMulticastChannel::commit_message(void* handle) {
    // This is a no-op in zero-copy design
    // Message was already written directly to ring buffer
}

void UDPMulticastChannel::release_message(void *handle) {
    // In zero-copy design, caller must call advance_read_pointer()
}

void UDPMulticastChannel::start_receive() {
    if (role_ != MulticastRole::Subscriber || stopping_) {
        return;
    }

    recv_buffer_.resize(65536); // Maximum UDP packet size

    socket_->async_receive_from(
        asio::buffer(recv_buffer_), sender_endpoint_tmp_,
        [this](const boost::system::error_code &error,
               size_t bytes_transferred) {
            handle_receive(error, bytes_transferred, sender_endpoint_tmp_);
        });
}

void UDPMulticastChannel::handle_receive(const boost::system::error_code &error,
                                         size_t bytes_transferred,
                                         udp::endpoint sender_endpoint) {
    std::cout << "DEBUG: handle_receive called, bytes: " << bytes_transferred
              << std::endl;

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
    auto *header =
        reinterpret_cast<const UDPMulticastHeader *>(recv_buffer_.data());
    if (!validate_message(*header,
                          recv_buffer_.data() + sizeof(UDPMulticastHeader),
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

    auto &msg = send_queue_.front();

    std::cout << "DEBUG: Sending " << msg.data.size() << " bytes to "
              << multicast_endpoint_.address().to_string() << ":"
              << multicast_endpoint_.port() << std::endl;

    socket_->async_send_to(asio::buffer(msg.data), multicast_endpoint_,
                           [this](const boost::system::error_code &error,
                                  size_t bytes_transferred) {
                               handle_send(error, bytes_transferred);
                           });
}

void UDPMulticastChannel::handle_send(const boost::system::error_code &error,
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
    } catch (const std::exception &e) {
        if (!stopping_) {
            std::cerr << "IO service error: " << e.what() << std::endl;
        }
    }
}

uint64_t UDPMulticastChannel::calculate_checksum(const uint8_t *data,
                                                 size_t size) {
    return utils::udp::calculate_checksum(data, size);
}

bool UDPMulticastChannel::validate_message(const UDPMulticastHeader &header,
                                           const uint8_t *payload,
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
    const std::string &uri, size_t buffer_size, ChannelMode mode,
    ChannelType type, MulticastRole role,
    const compression::CompressionConfig &compression_config,
    const std::string &interface_address) {
    return std::make_unique<UDPMulticastChannel>(uri, buffer_size, mode, type,
                                                 role, compression_config,
                                                 interface_address);
}

} // namespace detail
} // namespace psyne