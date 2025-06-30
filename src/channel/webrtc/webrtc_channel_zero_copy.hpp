#pragma once

/**
 * @file webrtc_channel_zero_copy.hpp
 * @brief Zero-copy optimized WebRTC channel for P2P messaging
 * 
 * This implementation extends zero-copy principles to WebRTC data channels:
 * - Direct streaming from ring buffer to WebRTC data channel
 * - GPU buffer coordination for AI/ML workloads  
 * - SIMD-optimized encoding/decoding
 * - Modern C++20 coroutines for async operations
 */

#include "../../memory/ring_buffer.hpp"
#include "channel_impl.hpp"
#include <span>
#include <concepts>
#include <coroutine>

namespace psyne {
namespace detail {

/**
 * @brief WebRTC message frame optimized for zero-copy
 */
struct alignas(16) WebRTCFrameHeader {
    uint32_t payload_size;     ///< Size of actual message data
    uint32_t ring_buffer_id;   ///< Source ring buffer identifier
    uint64_t sequence_number;  ///< For ordering and duplicate detection
    uint32_t chunk_index;      ///< For large message fragmentation
    uint32_t total_chunks;     ///< Total chunks for this message
    uint32_t checksum;         ///< Fast integrity check
    uint32_t flags;            ///< Control flags (compressed, encrypted, etc)
};

/**
 * @brief Zero-copy WebRTC data channel configuration
 */
struct WebRTCChannelConfig {
    // WebRTC configuration
    std::vector<std::string> stun_servers = {
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302"
    };
    
    // Data channel settings optimized for zero-copy
    std::string label = "psyne-zero-copy";
    bool ordered = true;           // Ordered delivery for reliability
    uint16_t max_retransmits = 3;  // Limited retries for low latency
    
    // Zero-copy optimizations
    size_t max_message_size = 64 * 1024;  // 64KB chunks for efficiency
    bool enable_compression = true;        // LZ4 compression for network efficiency
    bool enable_batching = true;          // Batch small messages
    
    // GPU optimization
    bool gpu_direct = false;              // Direct GPU memory access
    size_t gpu_batch_size = 1024;        // GPU batch processing size
    
    // Performance tuning
    std::chrono::microseconds batch_timeout{100};  // Max batching delay
    size_t send_buffer_size = 1024 * 1024;         // 1MB send buffer
    size_t receive_buffer_size = 1024 * 1024;      // 1MB receive buffer
};

/**
 * @brief High-performance WebRTC channel with zero-copy semantics
 */
class ZeroCopyWebRTCChannel : public ChannelImpl {
public:
    ZeroCopyWebRTCChannel(const std::string& uri, size_t buffer_size,
                         ChannelMode mode, ChannelType type,
                         const WebRTCChannelConfig& config = {});
    
    ~ZeroCopyWebRTCChannel();

    // Zero-copy interface
    uint32_t reserve_write_slot(size_t size) noexcept override;
    void notify_message_ready(uint32_t offset, size_t size) noexcept override;
    RingBuffer& get_ring_buffer() noexcept override;
    const RingBuffer& get_ring_buffer() const noexcept override;
    void advance_read_pointer(size_t size) noexcept override;
    
    // Modern C++20 interface
    std::span<uint8_t> get_write_span(size_t size) noexcept override;
    std::span<const uint8_t> buffer_span() const noexcept override;

    // WebRTC-specific operations
    
    /**
     * @brief Connect to remote peer
     * @param peer_id Remote peer identifier
     * @param signaling_server WebSocket signaling server URL
     */
    [[nodiscard]] std::coroutine_handle<> connect(const std::string& peer_id,
                                                  const std::string& signaling_server);
    
    /**
     * @brief Accept incoming peer connection
     * @param signaling_server WebSocket signaling server URL for listening
     */
    [[nodiscard]] std::coroutine_handle<> listen(const std::string& signaling_server);
    
    /**
     * @brief Get connection statistics
     */
    struct ConnectionStats {
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        uint64_t packets_sent = 0;
        uint64_t packets_received = 0;
        uint32_t round_trip_time_ms = 0;
        uint32_t jitter_ms = 0;
        double packet_loss_rate = 0.0;
        bool is_connected = false;
        std::string connection_state;
    };
    
    ConnectionStats get_stats() const noexcept;

    // Legacy interface (deprecated)
    [[deprecated("Use reserve_write_slot() instead")]]
    void* reserve_space(size_t size) override;
    [[deprecated("Data is committed when written")]]
    void commit_message(void* handle) override;
    void* receive_message(size_t& size, uint32_t& type) override;
    void release_message(void* handle) override;

private:
    /**
     * @brief Async coroutine for zero-copy message sending
     */
    std::coroutine_handle<> async_send_zero_copy(uint32_t offset, size_t size);
    
    /**
     * @brief Async coroutine for zero-copy message receiving
     */
    std::coroutine_handle<> async_receive_zero_copy();
    
    /**
     * @brief Fragment large messages for WebRTC MTU limits
     */
    std::vector<std::span<const uint8_t>> fragment_message(std::span<const uint8_t> data);
    
    /**
     * @brief Reassemble fragmented messages
     */
    std::optional<std::span<uint8_t>> reassemble_fragments(uint64_t sequence_number);
    
    /**
     * @brief Compress data using LZ4 for network efficiency
     */
    std::vector<uint8_t> compress_data(std::span<const uint8_t> data);
    
    /**
     * @brief Decompress received data
     */
    std::vector<uint8_t> decompress_data(std::span<const uint8_t> compressed_data, size_t original_size);
    
    /**
     * @brief SIMD-optimized data encoding
     */
    void simd_encode_data(std::span<const uint8_t> src, std::span<uint8_t> dst);
    void simd_decode_data(std::span<const uint8_t> src, std::span<uint8_t> dst);

    // Core components
    std::unique_ptr<RingBuffer> ring_buffer_;
    WebRTCChannelConfig config_;
    
    // WebRTC components (simplified interface)
    struct WebRTCPeerConnection;
    struct WebRTCDataChannel;
    std::unique_ptr<WebRTCPeerConnection> peer_connection_;
    std::unique_ptr<WebRTCDataChannel> data_channel_;
    
    // Connection state
    std::atomic<bool> is_connected_{false};
    std::atomic<bool> should_stop_{false};
    std::string peer_id_;
    
    // Zero-copy optimizations
    struct PendingSend {
        uint32_t offset;
        size_t size;
        std::chrono::steady_clock::time_point timestamp;
        uint64_t sequence_number;
    };
    
    std::vector<PendingSend> pending_sends_;
    std::mutex send_mutex_;
    
    // Message fragmentation/reassembly
    struct FragmentedMessage {
        std::vector<std::vector<uint8_t>> chunks;
        size_t total_size;
        uint32_t received_chunks;
        uint32_t total_chunks;
        std::chrono::steady_clock::time_point start_time;
    };
    
    std::unordered_map<uint64_t, FragmentedMessage> pending_fragments_;
    std::mutex fragments_mutex_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    ConnectionStats stats_;
    
    // Async I/O
    std::jthread io_thread_;
    std::atomic<uint64_t> next_sequence_number_{1};
};

/**
 * @brief Concept for WebRTC-compatible message types
 */
template<typename T>
concept WebRTCCompatible = MessageType<T> && requires {
    // Must be efficiently serializable over network
    requires std::is_trivially_copyable_v<T> || requires(T t) {
        { t.webrtc_serialize() } -> std::convertible_to<std::span<const uint8_t>>;
        { t.webrtc_deserialize(std::span<const uint8_t>{}) } -> std::same_as<bool>;
    };
    
    // Should have reasonable size limits for WebRTC
    requires (T::calculate_size() <= 1024 * 1024); // Max 1MB per message
};

/**
 * @brief Factory for creating WebRTC channels optimized for different workloads
 */
class WebRTCChannelFactory {
public:
    /**
     * @brief Create WebRTC channel optimized for AI/ML tensor streaming
     */
    template<WebRTCCompatible MessageType>
    static std::unique_ptr<ZeroCopyWebRTCChannel>
    create_ai_streaming_channel(const std::string& peer_id, 
                               const std::string& signaling_server = "ws://localhost:8080") {
        WebRTCChannelConfig config;
        config.ordered = false;           // Allow out-of-order for lower latency
        config.max_retransmits = 1;       // Minimal retries for real-time
        config.enable_compression = true; // Compress tensors
        config.gpu_direct = true;         // GPU optimization
        config.batch_timeout = std::chrono::microseconds{50}; // Low latency batching
        
        std::string uri = "webrtc://" + peer_id;
        return std::make_unique<ZeroCopyWebRTCChannel>(
            uri, calculate_ai_buffer_size<MessageType>(), 
            ChannelMode::SPSC, ChannelType::SingleType, config
        );
    }
    
    /**
     * @brief Create WebRTC channel for reliable data transfer
     */
    template<WebRTCCompatible MessageType>
    static std::unique_ptr<ZeroCopyWebRTCChannel>
    create_reliable_channel(const std::string& peer_id,
                           const std::string& signaling_server = "ws://localhost:8080") {
        WebRTCChannelConfig config;
        config.ordered = true;            // Guaranteed ordering
        config.max_retransmits = 10;      // Higher reliability
        config.enable_compression = true; // Bandwidth efficiency
        config.gpu_direct = false;        // Standard memory
        config.batch_timeout = std::chrono::microseconds{500}; // Higher throughput batching
        
        std::string uri = "webrtc://" + peer_id;
        return std::make_unique<ZeroCopyWebRTCChannel>(
            uri, calculate_reliable_buffer_size<MessageType>(),
            ChannelMode::SPSC, ChannelType::SingleType, config
        );
    }

private:
    template<typename T>
    static size_t calculate_ai_buffer_size() {
        // Size for real-time AI streaming (lower latency, higher memory usage)
        if constexpr (FixedSizeMessage<T>) {
            return T::static_size() * 100; // Buffer 100 messages
        } else {
            return 16 * 1024 * 1024; // 16MB for dynamic messages
        }
    }
    
    template<typename T>
    static size_t calculate_reliable_buffer_size() {
        // Size for reliable transfer (higher latency tolerance, memory efficient)
        if constexpr (FixedSizeMessage<T>) {
            return T::static_size() * 1000; // Buffer 1000 messages
        } else {
            return 64 * 1024 * 1024; // 64MB for dynamic messages
        }
    }
};

} // namespace detail
} // namespace psyne