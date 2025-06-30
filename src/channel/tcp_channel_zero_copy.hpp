#pragma once

/**
 * @file tcp_channel_zero_copy.hpp
 * @brief Zero-copy optimized TCP channel implementation
 * 
 * This implementation follows CORE_DESIGN.md principles for network transport:
 * - Sender streams directly from ring buffer (zero-copy from user perspective)
 * - Receiver writes directly to matching buffer type (GPU buffers write to GPU-visible memory)
 * - Uses modern C++20 features: std::span, concepts, coroutines
 */

#include "../memory/ring_buffer.hpp"
#include "channel_impl.hpp"
#include <boost/asio.hpp>
#include <span>
#include <concepts>
#include <memory>

namespace psyne {
namespace detail {

namespace asio = boost::asio;
using tcp = asio::ip::tcp;

/**
 * @brief Lightweight TCP frame header for zero-copy messaging
 * 
 * Minimal overhead header for network transport while maintaining
 * zero-copy semantics on both ends.
 */
struct alignas(16) TCPFrameHeader {
    uint32_t message_size;    ///< Size of message payload
    uint32_t offset;          ///< Offset in destination ring buffer  
    uint64_t channel_id;      ///< Channel identifier for multiplexing
    uint32_t checksum;        ///< Fast checksum for integrity
    uint32_t flags;           ///< Future extensibility
};

/**
 * @brief Zero-copy TCP channel optimized for AI/ML workloads
 * 
 * Key optimizations:
 * - Direct streaming from/to ring buffers
 * - Batched sends for high throughput
 * - GPU buffer coordination
 * - Coroutine-based async I/O
 */
class ZeroCopyTCPChannel : public ChannelImpl {
public:
    ZeroCopyTCPChannel(const std::string& uri, size_t buffer_size, 
                       ChannelMode mode, ChannelType type,
                       const std::string& host, uint16_t port);
    
    ZeroCopyTCPChannel(const std::string& uri, size_t buffer_size,
                       ChannelMode mode, ChannelType type, uint16_t port);
    
    ~ZeroCopyTCPChannel();

    // Zero-copy interface implementation
    uint32_t reserve_write_slot(size_t size) noexcept override;
    void notify_message_ready(uint32_t offset, size_t size) noexcept override;
    RingBuffer& get_ring_buffer() noexcept override;
    const RingBuffer& get_ring_buffer() const noexcept override;
    void advance_read_pointer(size_t size) noexcept override;
    
    // Modern C++20 interface
    std::span<uint8_t> get_write_span(size_t size) noexcept override;
    std::span<const uint8_t> buffer_span() const noexcept override;

    // Legacy interface (deprecated)
    [[deprecated("Use reserve_write_slot() instead")]]
    void* reserve_space(size_t size) override;
    [[deprecated("Data is committed when written")]]
    void commit_message(void* handle) override;
    void* receive_message(size_t& size, uint32_t& type) override;
    void release_message(void* handle) override;

private:
    /**
     * @brief Async send coroutine for zero-copy network streaming
     * 
     * Streams data directly from ring buffer to network without copying.
     */
    asio::awaitable<void> async_send_from_ring_buffer(uint32_t offset, size_t size);
    
    /**
     * @brief Async receive coroutine for zero-copy network reading
     * 
     * Reads data directly into ring buffer from network.
     */
    asio::awaitable<bool> async_receive_to_ring_buffer();
    
    /**
     * @brief Batch multiple small messages for network efficiency
     */
    void try_batch_send() noexcept;
    
    /**
     * @brief Handle TCP connection management
     */
    void handle_connection();
    void handle_reconnection();

    // Core components
    std::unique_ptr<RingBuffer> ring_buffer_;
    asio::io_context io_context_;
    tcp::socket socket_;
    tcp::acceptor acceptor_;  // For server mode
    
    // Connection info
    std::string host_;
    uint16_t port_;
    bool is_server_;
    bool is_connected_;
    
    // Zero-copy optimizations
    struct PendingSend {
        uint32_t offset;
        size_t size;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::vector<PendingSend> pending_sends_;
    std::mutex send_mutex_;
    
    // Async I/O management
    std::thread io_thread_;
    std::atomic<bool> should_stop_;
    
    // Performance settings
    static constexpr size_t MAX_BATCH_SIZE = 64;
    static constexpr std::chrono::microseconds BATCH_TIMEOUT{100};
    static constexpr size_t TCP_BUFFER_SIZE = 64 * 1024;  // 64KB TCP buffer
};

/**
 * @brief Concept for network-serializable message types
 */
template<typename T>
concept NetworkSerializable = MessageType<T> && requires {
    // Must be POD or have custom serialization
    requires std::is_trivially_copyable_v<T> || requires(T t) {
        { t.serialize() } -> std::convertible_to<std::span<const uint8_t>>;
        { t.deserialize(std::span<const uint8_t>{}) } -> std::same_as<bool>;
    };
};

/**
 * @brief Factory for creating optimized TCP channels
 */
class TCPChannelFactory {
public:
    /**
     * @brief Create zero-copy TCP channel optimized for specific workload
     */
    template<NetworkSerializable MessageType>
    static std::unique_ptr<ZeroCopyTCPChannel> 
    create_for_workload(const std::string& host, uint16_t port, 
                       size_t buffer_size = 0) {
        // Auto-size buffer based on message type
        if (buffer_size == 0) {
            if constexpr (FixedSizeMessage<MessageType>) {
                // Size for ~1000 messages of this type
                buffer_size = MessageType::static_size() * 1000;
            } else {
                // Default to 64MB for dynamic messages
                buffer_size = 64 * 1024 * 1024;
            }
        }
        
        std::string uri = "tcp://" + host + ":" + std::to_string(port);
        return std::make_unique<ZeroCopyTCPChannel>(
            uri, buffer_size, ChannelMode::SPSC, ChannelType::SingleType, host, port
        );
    }
    
    /**
     * @brief Create GPU-optimized TCP channel
     */
    template<NetworkSerializable MessageType>
    static std::unique_ptr<ZeroCopyTCPChannel>
    create_gpu_channel(const std::string& host, uint16_t port) {
        // Use GPU-visible memory for ring buffer
        auto channel = create_for_workload<MessageType>(host, port);
        
#ifdef PSYNE_GPU_ENABLED
        // Initialize ring buffer in GPU-visible memory
        auto& ring_buffer = channel->get_ring_buffer();
        
#ifdef PSYNE_CUDA_ENABLED
        // For CUDA, use pinned memory for zero-copy transfers
        cudaHostAlloc(&ring_buffer.base_ptr(), ring_buffer.capacity(), 
                      cudaHostAllocMapped | cudaHostAllocWriteCombined);
#elif defined(PSYNE_METAL_ENABLED)
        // For Metal, use shared memory
        // Metal automatically handles CPU-GPU memory coherency
#elif defined(PSYNE_VULKAN_ENABLED)
        // For Vulkan, use host-visible coherent memory
        // This would be set up through the Vulkan memory allocator
#endif
#endif
        
        return channel;
    }
};

} // namespace detail
} // namespace psyne