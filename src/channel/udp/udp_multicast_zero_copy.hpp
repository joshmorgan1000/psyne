#pragma once

/**
 * @file udp_multicast_zero_copy.hpp
 * @brief Zero-copy UDP multicast channel for high-throughput broadcasting
 * 
 * Optimized for AI/ML scenarios where one producer broadcasts to many consumers:
 * - SIMD-optimized packet processing
 * - Zero-copy ring buffer streaming  
 * - GPU direct memory access
 * - Modern C++20 ranges and algorithms
 */

#include "../memory/ring_buffer.hpp"
#include "channel_impl.hpp"
#include <boost/asio.hpp>
#include <span>
#include <ranges>
#include <algorithm>

namespace psyne {
namespace detail {

namespace asio = boost::asio;
using udp = asio::ip::udp;

/**
 * @brief UDP multicast frame header optimized for high throughput
 */
struct alignas(16) UDPMulticastHeader {
    uint32_t sequence_number;   ///< For ordering and loss detection
    uint32_t payload_size;      ///< Size of actual data
    uint32_t fragment_id;       ///< Fragment index for large messages
    uint32_t total_fragments;   ///< Total fragments for message
    uint64_t channel_id;        ///< Channel identifier for multiplexing
    uint32_t checksum;          ///< Fast CRC32 checksum
    uint32_t compression_type;  ///< Compression algorithm used
    uint32_t original_size;     ///< Size before compression
    uint32_t flags;             ///< Control flags
} __attribute__((packed));

/**
 * @brief Configuration for zero-copy UDP multicast
 */
struct UDPMulticastConfig {
    // Network settings
    std::string multicast_address = "239.255.1.1";
    uint16_t port = 5010;
    std::string interface_address = "0.0.0.0";
    
    // Performance settings
    size_t mtu_size = 1500;                    // Network MTU
    size_t max_packet_size = 1472;             // MTU minus headers
    size_t send_buffer_size = 8 * 1024 * 1024; // 8MB OS send buffer
    size_t recv_buffer_size = 8 * 1024 * 1024; // 8MB OS receive buffer
    
    // Zero-copy optimizations
    bool use_kernel_bypass = false;    // Use DPDK/AF_XDP for kernel bypass
    bool enable_simd = true;           // SIMD packet processing
    bool enable_batching = true;       // Batch multiple packets
    size_t batch_size = 64;            // Packets per batch
    
    // Reliability settings
    bool enable_forward_error_correction = false; // FEC for packet loss
    uint32_t fec_redundancy_percent = 20;         // 20% redundancy
    
    // Compression
    bool enable_compression = true;     // LZ4 compression
    size_t compression_threshold = 256; // Compress messages > 256 bytes
    
    // GPU optimization
    bool gpu_direct = false;           // Direct GPU memory access
    size_t gpu_batch_size = 1024;     // GPU processing batch size
    
    // Timing
    std::chrono::microseconds batch_timeout{50}; // Max batching delay
    std::chrono::seconds heartbeat_interval{1};  // Keepalive interval
};

/**
 * @brief High-performance UDP multicast channel with zero-copy
 */
class ZeroCopyUDPMulticastChannel : public ChannelImpl {
public:
    // Constructor for sender
    ZeroCopyUDPMulticastChannel(const std::string& uri, size_t buffer_size,
                               ChannelMode mode, ChannelType type,
                               const UDPMulticastConfig& config = {});
    
    // Constructor for receiver
    ZeroCopyUDPMulticastChannel(const std::string& uri, size_t buffer_size,
                               ChannelMode mode, ChannelType type,
                               const std::string& multicast_address, uint16_t port,
                               const UDPMulticastConfig& config = {});
    
    ~ZeroCopyUDPMulticastChannel();

    // Zero-copy interface
    uint32_t reserve_write_slot(size_t size) noexcept override;
    void notify_message_ready(uint32_t offset, size_t size) noexcept override;
    RingBuffer& get_ring_buffer() noexcept override;
    const RingBuffer& get_ring_buffer() const noexcept override;
    void advance_read_pointer(size_t size) noexcept override;
    
    // Modern C++20 interface with ranges
    std::span<uint8_t> get_write_span(size_t size) noexcept override;
    std::span<const uint8_t> buffer_span() const noexcept override;
    
    /**
     * @brief Get range view over available messages
     */
    auto available_messages() const -> std::ranges::view auto;
    
    /**
     * @brief Process messages using ranges and algorithms
     */
    template<std::ranges::range Range>
    void process_message_range(Range&& messages);

    // Multicast-specific operations
    
    /**
     * @brief Join multicast group
     */
    void join_group(const std::string& multicast_address);
    
    /**
     * @brief Leave multicast group
     */
    void leave_group(const std::string& multicast_address);
    
    /**
     * @brief Get multicast statistics
     */
    struct MulticastStats {
        uint64_t packets_sent = 0;
        uint64_t packets_received = 0;
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        uint64_t packets_lost = 0;
        uint64_t packets_duplicated = 0;
        uint64_t packets_out_of_order = 0;
        double packet_loss_rate = 0.0;
        uint32_t max_sequence_gap = 0;
        std::chrono::milliseconds average_latency{0};
        
        // Zero-copy performance metrics
        uint64_t zero_copy_sends = 0;
        uint64_t zero_copy_receives = 0;
        uint64_t memory_copies_avoided = 0;
        size_t bytes_avoided_copying = 0;
    };
    
    MulticastStats get_stats() const noexcept;
    void reset_stats() noexcept;

    // Legacy interface (deprecated)
    [[deprecated("Use reserve_write_slot() instead")]]
    void* reserve_space(size_t size) override;
    [[deprecated("Data is committed when written")]]
    void commit_message(void* handle) override;
    void* receive_message(size_t& size, uint32_t& type) override;
    void release_message(void* handle) override;

private:
    /**
     * @brief SIMD-optimized packet fragmentation
     */
    std::vector<std::span<const uint8_t>> simd_fragment_message(std::span<const uint8_t> data);
    
    /**
     * @brief SIMD-optimized packet reassembly
     */
    std::optional<std::vector<uint8_t>> simd_reassemble_fragments(uint32_t sequence_number);
    
    /**
     * @brief Zero-copy packet sending with batching
     */
    void batch_send_packets();
    
    /**
     * @brief Zero-copy packet receiving with SIMD processing
     */
    void simd_receive_packets();
    
    /**
     * @brief Compress data using SIMD-accelerated algorithms
     */
    std::vector<uint8_t> simd_compress(std::span<const uint8_t> data);
    std::vector<uint8_t> simd_decompress(std::span<const uint8_t> compressed, size_t original_size);
    
    /**
     * @brief Calculate checksum using SIMD CRC32
     */
    uint32_t simd_checksum(std::span<const uint8_t> data) const noexcept;
    
    /**
     * @brief Forward Error Correction encoding/decoding
     */
    std::vector<uint8_t> fec_encode(std::span<const uint8_t> data);
    std::optional<std::vector<uint8_t>> fec_decode(const std::vector<std::span<const uint8_t>>& packets);

    // Core components
    std::unique_ptr<RingBuffer> ring_buffer_;
    UDPMulticastConfig config_;
    
    // Network components
    asio::io_context io_context_;
    udp::socket socket_;
    udp::endpoint multicast_endpoint_;
    std::vector<udp::endpoint> receiver_endpoints_; // For sender tracking
    
    // Zero-copy packet management
    struct PendingPacket {
        uint32_t sequence_number;
        std::span<const uint8_t> data;
        std::chrono::steady_clock::time_point timestamp;
        bool needs_fragmentation;
    };
    
    std::vector<PendingPacket> send_queue_;
    std::mutex send_queue_mutex_;
    
    // Message reassembly
    struct FragmentedMessage {
        std::vector<std::vector<uint8_t>> fragments;
        std::bitset<256> received_fragments; // Max 256 fragments per message
        uint32_t total_fragments;
        size_t total_size;
        std::chrono::steady_clock::time_point start_time;
    };
    
    std::unordered_map<uint32_t, FragmentedMessage> pending_messages_;
    std::mutex reassembly_mutex_;
    
    // Statistics and monitoring
    mutable std::mutex stats_mutex_;
    MulticastStats stats_;
    
    // Async I/O management
    std::jthread send_thread_;
    std::jthread receive_thread_;
    std::jthread stats_thread_;
    std::atomic<bool> should_stop_{false};
    
    // Sequence number management
    std::atomic<uint32_t> next_sequence_number_{1};
    std::atomic<uint32_t> last_received_sequence_{0};
    
    // Performance optimization state
    bool is_sender_;
    std::atomic<size_t> pending_send_bytes_{0};
    
    // SIMD optimization detection
    bool has_avx2_ = false;
    bool has_avx512_ = false;
    
    // Kernel bypass (DPDK/AF_XDP) interface
    struct KernelBypassInterface;
    std::unique_ptr<KernelBypassInterface> kernel_bypass_;
};

/**
 * @brief Concept for multicast-compatible message types
 */
template<typename T>
concept MulticastCompatible = MessageType<T> && requires {
    // Must be efficiently broadcastable
    requires std::is_trivially_copyable_v<T> || requires(T t) {
        { t.multicast_serialize() } -> std::convertible_to<std::span<const uint8_t>>;
        { t.multicast_deserialize(std::span<const uint8_t>{}) } -> std::same_as<bool>;
    };
    
    // Should fit within reasonable packet limits
    requires (T::calculate_size() <= 64 * 1024); // Max 64KB per message for efficiency
};

/**
 * @brief Factory for creating optimized UDP multicast channels
 */
class UDPMulticastChannelFactory {
public:
    /**
     * @brief Create high-throughput broadcasting channel
     */
    template<MulticastCompatible MessageType>
    static std::unique_ptr<ZeroCopyUDPMulticastChannel>
    create_broadcast_channel(const std::string& multicast_address = "239.255.1.1",
                           uint16_t port = 5010) {
        UDPMulticastConfig config;
        config.multicast_address = multicast_address;
        config.port = port;
        config.enable_batching = true;
        config.batch_size = 128;           // Large batches for throughput
        config.enable_compression = true;   // Bandwidth efficiency
        config.use_kernel_bypass = true;   // Maximum performance
        config.batch_timeout = std::chrono::microseconds{100}; // Batch for throughput
        
        std::string uri = "udp://" + multicast_address + ":" + std::to_string(port);
        return std::make_unique<ZeroCopyUDPMulticastChannel>(
            uri, calculate_broadcast_buffer_size<MessageType>(),
            ChannelMode::SPMC, ChannelType::SingleType, config
        );
    }
    
    /**
     * @brief Create low-latency streaming channel
     */
    template<MulticastCompatible MessageType>
    static std::unique_ptr<ZeroCopyUDPMulticastChannel>
    create_streaming_channel(const std::string& multicast_address = "239.255.1.2",
                           uint16_t port = 5011) {
        UDPMulticastConfig config;
        config.multicast_address = multicast_address;
        config.port = port;
        config.enable_batching = true;
        config.batch_size = 16;            // Smaller batches for latency
        config.enable_compression = false; // Skip compression for speed
        config.use_kernel_bypass = true;   // Kernel bypass for latency
        config.batch_timeout = std::chrono::microseconds{10}; // Low latency
        
        std::string uri = "udp://" + multicast_address + ":" + std::to_string(port);
        return std::make_unique<ZeroCopyUDPMulticastChannel>(
            uri, calculate_streaming_buffer_size<MessageType>(),
            ChannelMode::SPMC, ChannelType::SingleType, config
        );
    }

private:
    template<typename T>
    static size_t calculate_broadcast_buffer_size() {
        // Large buffer for high throughput broadcasting
        if constexpr (FixedSizeMessage<T>) {
            return T::static_size() * 10000; // Buffer 10K messages
        } else {
            return 128 * 1024 * 1024; // 128MB for dynamic messages
        }
    }
    
    template<typename T>
    static size_t calculate_streaming_buffer_size() {
        // Smaller buffer optimized for low latency
        if constexpr (FixedSizeMessage<T>) {
            return T::static_size() * 1000; // Buffer 1K messages
        } else {
            return 32 * 1024 * 1024; // 32MB for dynamic messages
        }
    }
};

} // namespace detail
} // namespace psyne