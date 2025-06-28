#include "rdma_channel.hpp"
#include "../utils/checksum.hpp"
#include <regex>
#include <iostream>
#include <chrono>
#include <cstring>
#include <algorithm>

namespace psyne {
namespace detail {

RDMAChannel::RDMAChannel(const std::string& uri, size_t buffer_size,
                         ChannelMode mode, ChannelType type,
                         RDMARole role, const RDMAConfig& config)
    : ChannelImpl(uri, buffer_size, mode, type)
    , role_(role)
    , config_(config)
    , connected_(false)
    , stopping_(false)
    , sequence_number_(0) {
    
    parse_uri(uri);
    
    // Allocate RDMA buffers
    send_buffer_.resize(buffer_size / 2);
    recv_buffer_.resize(buffer_size / 2);
    
    // Setup mock RDMA device
    if (!setup_rdma_device()) {
        throw std::runtime_error("Failed to initialize RDMA device");
    }
    
    // Create queue pair
    if (!create_queue_pair()) {
        throw std::runtime_error("Failed to create RDMA queue pair");
    }
    
    // Start completion and receive handlers
    completion_thread_ = std::thread([this]() { run_completion_handler(); });
    recv_thread_ = std::thread([this]() { run_receive_handler(); });
    
    // Establish connection for client
    if (role_ == RDMARole::Client) {
        if (!establish_connection()) {
            throw std::runtime_error("Failed to establish RDMA connection");
        }
    }
    
    std::cout << "RDMA Channel initialized: " << uri 
              << " (device: " << device_.name 
              << ", transport: " << (config_.transport_mode == RDMATransportMode::RC ? "RC" : 
                                   config_.transport_mode == RDMATransportMode::UC ? "UC" : "UD")
              << ")" << std::endl;
}

RDMAChannel::~RDMAChannel() {
    stopping_.store(true);
    connected_.store(false);
    
    if (completion_thread_.joinable()) {
        completion_thread_.join();
    }
    
    if (recv_thread_.joinable()) {
        recv_thread_.join();
    }
    
    // Cleanup memory registrations
    memory_regions_.clear();
    
    std::cout << "RDMA Channel destroyed" << std::endl;
}

void RDMAChannel::parse_uri(const std::string& uri) {
    // Parse RDMA URI: rdma://host:port
    std::regex uri_regex(R"(rdma://([^:]+):(\d+))");
    std::smatch match;
    
    if (std::regex_match(uri, match, uri_regex)) {
        remote_address_ = match[1].str();
        remote_port_ = static_cast<uint16_t>(std::stoi(match[2].str()));
    } else {
        throw std::invalid_argument("Invalid RDMA URI format: " + uri);
    }
}

bool RDMAChannel::setup_rdma_device() {
    // In a real implementation, this would:
    // 1. Query available RDMA devices using ibv_get_device_list()
    // 2. Open the device with ibv_open_device()
    // 3. Allocate protection domain with ibv_alloc_pd()
    // 4. Create completion queue with ibv_create_cq()
    
    // Mock implementation
    if (config_.device_name.empty()) {
        device_.name = "mlx5_0";  // Default to Mellanox ConnectX device
    } else {
        device_.name = config_.device_name;
    }
    
    device_.active = true;
    device_.port_count = 2;
    device_.max_mr_size = 1ULL << 63;
    
    std::cout << "RDMA device '" << device_.name << "' initialized" << std::endl;
    return true;
}

bool RDMAChannel::create_queue_pair() {
    // In a real implementation, this would:
    // 1. Create queue pair with ibv_create_qp()
    // 2. Modify QP state: RESET -> INIT -> RTR -> RTS
    // 3. Exchange connection information (QPN, GID, LID)
    
    // Mock implementation
    qp_.qp_num = 12345 + static_cast<uint32_t>(std::hash<std::string>{}(uri()) % 10000);
    qp_.transport = config_.transport_mode;
    qp_.sq_depth = config_.queue_depth;
    qp_.rq_depth = config_.queue_depth;
    qp_.connected = false;
    
    std::cout << "RDMA QP created: QPN=" << qp_.qp_num 
              << ", depth=" << qp_.sq_depth << std::endl;
    return true;
}

bool RDMAChannel::establish_connection() {
    // In a real implementation, this would:
    // 1. Resolve remote address using rdma_resolve_addr()
    // 2. Resolve route using rdma_resolve_route()
    // 3. Exchange QP information and modify QP states
    // 4. Post initial receive buffers
    
    // Mock implementation - simulate connection establishment
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    qp_.connected = true;
    connected_.store(true);
    
    std::cout << "RDMA connection established to " << remote_address_ 
              << ":" << remote_port_ << std::endl;
    return true;
}

void* RDMAChannel::reserve_space(size_t size) {
    if (!connected_.load()) {
        return nullptr;
    }
    
    if (size + sizeof(RDMAHeader) > send_buffer_.size()) {
        return nullptr;
    }
    
    // Return pointer after header space
    return send_buffer_.data() + sizeof(RDMAHeader);
}

void RDMAChannel::commit_message(void* handle) {
    if (!connected_.load() || !handle) {
        return;
    }
    
    // Calculate message size
    uint8_t* payload_start = static_cast<uint8_t*>(handle);
    uint8_t* buffer_start = send_buffer_.data();
    size_t payload_size = send_buffer_.size() - sizeof(RDMAHeader) - 
                         (payload_start - buffer_start - sizeof(RDMAHeader));
    
    // Fill RDMA header
    RDMAHeader* header = reinterpret_cast<RDMAHeader*>(buffer_start);
    header->length = static_cast<uint32_t>(sizeof(RDMAHeader) + payload_size);
    header->message_type = 1; // Default message type
    header->sequence = sequence_number_++;
    header->timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    header->checksum = calculate_checksum(payload_start, payload_size);
    header->flags = 0;
    
    // Post send operation
    if (mock_post_send(buffer_start, header->length)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.messages_sent++;
        stats_.bytes_sent += header->length;
    }
}

void* RDMAChannel::receive_message(size_t& size, uint32_t& type) {
    // In this mock implementation, we don't have a real receive queue
    // In a real implementation, this would check completion queue for received messages
    size = 0;
    type = 0;
    return nullptr;
}

void RDMAChannel::release_message(void* handle) {
    // In a real implementation, this would return the receive buffer to the pool
    // For mock implementation, this is a no-op
}

bool RDMAChannel::mock_post_send(const void* data, size_t length) {
    // In a real implementation, this would:
    // 1. Create scatter-gather element (SGE)
    // 2. Create work request (WR)
    // 3. Post to send queue with ibv_post_send()
    
    // Mock implementation - simulate successful send
    if (length <= config_.max_inline_data) {
        // Inline send - lowest latency
        std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    } else {
        // Regular send - slightly higher latency
        std::this_thread::sleep_for(std::chrono::nanoseconds(500));
    }
    
    return true;
}

bool RDMAChannel::mock_post_recv() {
    // In a real implementation, this would:
    // 1. Create SGE pointing to receive buffer
    // 2. Create receive work request
    // 3. Post to receive queue with ibv_post_recv()
    
    // Mock implementation
    return true;
}

void RDMAChannel::mock_poll_completions() {
    // In a real implementation, this would:
    // 1. Poll completion queue with ibv_poll_cq()
    // 2. Process work completions (WC)
    // 3. Handle send/receive completions and errors
    
    // Mock implementation - simulate some completions
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.send_completions++;
}

void RDMAChannel::run_completion_handler() {
    while (!stopping_.load()) {
        if (connected_.load()) {
            mock_poll_completions();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void RDMAChannel::run_receive_handler() {
    while (!stopping_.load()) {
        if (connected_.load()) {
            mock_post_recv();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

RDMAStats RDMAChannel::get_rdma_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void RDMAChannel::reset_rdma_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = RDMAStats{};
}

void RDMAChannel::set_qos(uint8_t traffic_class, uint8_t service_level) {
    // In a real implementation, this would modify QP attributes
    std::cout << "RDMA QoS set: TC=" << (int)traffic_class 
              << ", SL=" << (int)service_level << std::endl;
}

void RDMAChannel::set_adaptive_routing(bool enable) {
    // In a real implementation, this would configure adaptive routing
    std::cout << "RDMA adaptive routing: " << (enable ? "enabled" : "disabled") << std::endl;
}

void* RDMAChannel::register_memory(void* addr, size_t length) {
    // In a real implementation, this would:
    // 1. Register memory region with ibv_reg_mr()
    // 2. Get local and remote keys for RDMA operations
    
    MockMemoryRegion mr;
    mr.addr = addr;
    mr.length = length;
    mr.lkey = 0x10000 + memory_regions_.size();
    mr.rkey = 0x20000 + memory_regions_.size();
    
    memory_regions_.push_back(mr);
    
    std::cout << "RDMA memory registered: addr=" << addr 
              << ", length=" << length 
              << ", lkey=0x" << std::hex << mr.lkey << std::dec << std::endl;
    
    // Return mock handle (in real implementation, would return ibv_mr*)
    return reinterpret_cast<void*>(memory_regions_.size());
}

void RDMAChannel::unregister_memory(void* handle) {
    // In a real implementation, this would call ibv_dereg_mr()
    size_t index = reinterpret_cast<size_t>(handle) - 1;
    if (index < memory_regions_.size()) {
        std::cout << "RDMA memory unregistered: index=" << index << std::endl;
        // Mark as invalid (real implementation would free the MR)
        memory_regions_[index].addr = nullptr;
    }
}

void RDMAChannel::update_latency_stats(uint64_t latency_ns) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (stats_.messages_sent == 1) {
        stats_.min_latency_ns = latency_ns;
        stats_.max_latency_ns = latency_ns;
        stats_.avg_latency_ns = latency_ns;
    } else {
        stats_.min_latency_ns = std::min(stats_.min_latency_ns, static_cast<double>(latency_ns));
        stats_.max_latency_ns = std::max(stats_.max_latency_ns, static_cast<double>(latency_ns));
        
        // Update running average
        double alpha = 0.1; // Smoothing factor
        stats_.avg_latency_ns = alpha * latency_ns + (1.0 - alpha) * stats_.avg_latency_ns;
    }
}

uint64_t RDMAChannel::calculate_checksum(const uint8_t* data, size_t size) {
    // Use the same hash seed as other channels for consistency
    return utils::calculate_checksum(data, size, 0);
}

} // namespace detail

// Factory function implementations
namespace rdma {

std::unique_ptr<Channel> create_server(uint16_t port, size_t buffer_size) {
    std::string uri = "rdma://0.0.0.0:" + std::to_string(port);
    
    detail::RDMAConfig config{};  // Use default config
    auto impl = std::make_unique<psyne::detail::RDMAChannel>(
        uri, buffer_size, ChannelMode::SPMC, ChannelType::MultiType,
        psyne::detail::RDMARole::Server, config);
    
    // Create wrapper (similar to other channel types)
    class RDMAChannelWrapper : public Channel {
    public:
        explicit RDMAChannelWrapper(std::unique_ptr<psyne::detail::RDMAChannel> impl)
            : impl_(std::move(impl)) {}
        
        void stop() override { /* impl_->stop(); */ }
        bool is_stopped() const override { return false; }
        const std::string& uri() const override { return impl_->uri(); }
        ChannelType type() const override { return impl_->type(); }
        ChannelMode mode() const override { return impl_->mode(); }
        
        void* receive_raw_message(size_t& size, uint32_t& type) override {
            return impl_->receive_message(size, type);
        }
        
        void release_raw_message(void* handle) override {
            impl_->release_message(handle);
        }
        
        bool has_metrics() const override { return true; }
        
        debug::ChannelMetrics get_metrics() const override {
            auto rdma_stats = impl_->get_rdma_stats();
            debug::ChannelMetrics metrics;
            metrics.messages_sent = rdma_stats.messages_sent;
            metrics.messages_received = rdma_stats.messages_received;
            metrics.bytes_sent = rdma_stats.bytes_sent;
            metrics.bytes_received = rdma_stats.bytes_received;
            return metrics;
        }
        
        void reset_metrics() override {
            impl_->reset_rdma_stats();
        }
        
        // RDMA-specific methods
        psyne::detail::RDMAStats get_rdma_stats() const {
            return impl_->get_rdma_stats();
        }
        
        void set_qos(uint8_t traffic_class, uint8_t service_level) {
            impl_->set_qos(traffic_class, service_level);
        }
        
    private:
        psyne::detail::ChannelImpl* impl() override { return impl_.get(); }
        const psyne::detail::ChannelImpl* impl() const override { return impl_.get(); }
        
        std::unique_ptr<psyne::detail::RDMAChannel> impl_;
    };
    
    return std::make_unique<RDMAChannelWrapper>(std::move(impl));
}

std::unique_ptr<Channel> create_client(const std::string& host, uint16_t port,
                                      size_t buffer_size) {
    std::string uri = "rdma://" + host + ":" + std::to_string(port);
    
    detail::RDMAConfig config{};  // Use default config
    auto impl = std::make_unique<psyne::detail::RDMAChannel>(
        uri, buffer_size, ChannelMode::SPSC, ChannelType::MultiType,
        psyne::detail::RDMARole::Client, config);
    
    // Use same wrapper pattern
    class RDMAChannelWrapper : public Channel {
    public:
        explicit RDMAChannelWrapper(std::unique_ptr<psyne::detail::RDMAChannel> impl)
            : impl_(std::move(impl)) {}
        
        void stop() override { /* impl_->stop(); */ }
        bool is_stopped() const override { return false; }
        const std::string& uri() const override { return impl_->uri(); }
        ChannelType type() const override { return impl_->type(); }
        ChannelMode mode() const override { return impl_->mode(); }
        
        void* receive_raw_message(size_t& size, uint32_t& type) override {
            return impl_->receive_message(size, type);
        }
        
        void release_raw_message(void* handle) override {
            impl_->release_message(handle);
        }
        
        bool has_metrics() const override { return true; }
        
        debug::ChannelMetrics get_metrics() const override {
            auto rdma_stats = impl_->get_rdma_stats();
            debug::ChannelMetrics metrics;
            metrics.messages_sent = rdma_stats.messages_sent;
            metrics.messages_received = rdma_stats.messages_received;
            metrics.bytes_sent = rdma_stats.bytes_sent;
            metrics.bytes_received = rdma_stats.bytes_received;
            return metrics;
        }
        
        void reset_metrics() override {
            impl_->reset_rdma_stats();
        }
        
    private:
        psyne::detail::ChannelImpl* impl() override { return impl_.get(); }
        const psyne::detail::ChannelImpl* impl() const override { return impl_.get(); }
        
        std::unique_ptr<psyne::detail::RDMAChannel> impl_;
    };
    
    return std::make_unique<RDMAChannelWrapper>(std::move(impl));
}

std::unique_ptr<Channel> create_channel(const std::string& uri,
                                       psyne::detail::RDMARole role,
                                       size_t buffer_size) {
    if (role == psyne::detail::RDMARole::Server) {
        // Extract port from URI for server
        std::regex uri_regex(R"(rdma://[^:]*:(\d+))");
        std::smatch match;
        if (std::regex_match(uri, match, uri_regex)) {
            uint16_t port = static_cast<uint16_t>(std::stoi(match[1].str()));
            return create_server(port, buffer_size);
        } else {
            throw std::invalid_argument("Invalid RDMA URI for server: " + uri);
        }
    } else {
        // Extract host and port for client
        std::regex uri_regex(R"(rdma://([^:]+):(\d+))");
        std::smatch match;
        if (std::regex_match(uri, match, uri_regex)) {
            std::string host = match[1].str();
            uint16_t port = static_cast<uint16_t>(std::stoi(match[2].str()));
            return create_client(host, port, buffer_size);
        } else {
            throw std::invalid_argument("Invalid RDMA URI for client: " + uri);
        }
    }
}

} // namespace rdma

} // namespace psyne