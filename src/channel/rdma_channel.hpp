#pragma once

/**
 * @file rdma_channel.hpp
 * @brief RDMA/InfiniBand channel implementation for high-performance computing
 * 
 * This header provides RDMA (Remote Direct Memory Access) support for ultra-low latency
 * and high-bandwidth communication, commonly used in HPC clusters and data centers.
 * 
 * Features:
 * - Sub-microsecond latency communication
 * - High bandwidth (up to 200+ Gbps)
 * - CPU bypass with kernel bypass
 * - Zero-copy memory transfers
 * - Reliable Connection (RC) and Unreliable Datagram (UD) modes
 * 
 * Note: This implementation provides the interface and basic structure.
 * Production use requires linking against actual RDMA libraries (ibverbs, rdma-core).
 */

#include "channel_impl.hpp"
#include "../compression/compression.hpp"
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace psyne {
namespace detail {

/**
 * @struct RDMAHeader
 * @brief Header for RDMA messages with minimal overhead
 */
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
struct RDMAHeader {
    uint32_t length;          ///< Message length in bytes
    uint32_t message_type;    ///< Message type identifier
    uint64_t sequence;        ///< Sequence number for ordering
    uint64_t timestamp;       ///< Timestamp (nanoseconds)
    uint64_t checksum;        ///< xxHash64 checksum for validation
    uint32_t flags;           ///< Control flags
}
#ifdef _MSC_VER
#pragma pack(pop)
#else
__attribute__((packed))
#endif
;

static_assert(sizeof(RDMAHeader) == 36, "RDMAHeader must be 36 bytes for optimal alignment");

/**
 * @enum RDMATransportMode
 * @brief RDMA transport modes
 */
enum class RDMATransportMode {
    RC,    ///< Reliable Connection - guaranteed delivery, in-order
    UC,    ///< Unreliable Connection - no guarantees but faster
    UD     ///< Unreliable Datagram - connectionless, lowest latency
};

/**
 * @enum RDMARole
 * @brief RDMA endpoint role
 */
enum class RDMARole {
    Server,   ///< Listens for connections
    Client    ///< Initiates connections
};

/**
 * @struct RDMAConfig
 * @brief Configuration for RDMA connections
 */
struct RDMAConfig {
    RDMATransportMode transport_mode = RDMATransportMode::RC;
    uint32_t queue_depth = 256;           ///< Send/receive queue depth
    uint32_t max_inline_data = 64;        ///< Max inline data size (bytes)
    uint32_t max_sge = 16;                ///< Max scatter-gather elements
    bool use_odp = false;                 ///< Use On-Demand Paging
    std::string device_name = "";         ///< RDMA device name (auto-detect if empty)
    uint8_t port_num = 1;                 ///< RDMA port number
    uint8_t gid_index = 0;                ///< GID index for addressing
};

/**
 * @struct RDMAStats
 * @brief RDMA performance statistics
 */
struct RDMAStats {
    uint64_t messages_sent = 0;
    uint64_t messages_received = 0;
    uint64_t bytes_sent = 0;
    uint64_t bytes_received = 0;
    uint64_t send_completions = 0;
    uint64_t recv_completions = 0;
    uint64_t send_errors = 0;
    uint64_t recv_errors = 0;
    uint64_t retransmissions = 0;
    double avg_latency_ns = 0.0;
    double max_latency_ns = 0.0;
    double min_latency_ns = 0.0;
};

/**
 * @class RDMAChannel
 * @brief High-performance RDMA channel implementation
 * 
 * Provides ultra-low latency communication using RDMA/InfiniBand technology.
 * Supports both reliable and unreliable transport modes with zero-copy semantics.
 */
class RDMAChannel : public ChannelImpl {
public:
    /**
     * @brief Constructor for RDMA channel
     * @param uri Channel URI (e.g., "rdma://192.168.1.100:4791")
     * @param buffer_size Total buffer size for RDMA operations
     * @param mode Channel synchronization mode
     * @param type Channel message type support
     * @param role Server or client role
     * @param config RDMA-specific configuration
     */
    RDMAChannel(const std::string& uri, size_t buffer_size,
               ChannelMode mode, ChannelType type,
               RDMARole role, const RDMAConfig& config = {});
    
    /**
     * @brief Destructor - cleans up RDMA resources
     */
    ~RDMAChannel();
    
    // ChannelImpl interface
    void* reserve_space(size_t size) override;
    void commit_message(void* handle) override;
    void* receive_message(size_t& size, uint32_t& type) override;
    void release_message(void* handle) override;
    
    /**
     * @brief Get RDMA-specific statistics
     */
    RDMAStats get_rdma_stats() const;
    
    /**
     * @brief Reset RDMA statistics
     */
    void reset_rdma_stats();
    
    /**
     * @brief Set RDMA quality of service parameters
     * @param traffic_class Traffic class (0-7)
     * @param service_level Service level (0-15)
     */
    void set_qos(uint8_t traffic_class, uint8_t service_level);
    
    /**
     * @brief Enable/disable adaptive routing
     */
    void set_adaptive_routing(bool enable);
    
    /**
     * @brief Perform RDMA memory registration for zero-copy operations
     * @param addr Memory address to register
     * @param length Memory region length
     * @return Memory registration handle (opaque pointer)
     */
    void* register_memory(void* addr, size_t length);
    
    /**
     * @brief Unregister RDMA memory
     * @param handle Memory registration handle
     */
    void unregister_memory(void* handle);

    /**
     * @brief Calculate checksum for RDMA payload
     * @param data Pointer to data
     * @param size Size of data in bytes
     * @return 64-bit checksum
     */
    uint64_t calculate_checksum(const uint8_t* data, size_t size);

private:
    // RDMA configuration and state
    RDMARole role_;
    RDMAConfig config_;
    std::string remote_address_;
    uint16_t remote_port_;
    
    // Connection state
    std::atomic<bool> connected_;
    std::atomic<bool> stopping_;
    
    // Message handling
    std::vector<uint8_t> send_buffer_;
    std::vector<uint8_t> recv_buffer_;
    uint64_t sequence_number_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    RDMAStats stats_;
    
    // Threading
    std::thread completion_thread_;
    std::thread recv_thread_;
    
    // Simulated RDMA operations (in real implementation, these would use ibverbs)
    struct MockRDMADevice {
        std::string name = "mlx5_0";
        uint8_t port_count = 2;
        bool active = true;
        uint64_t max_mr_size = 1ULL << 63;  // Max memory registration size
    };
    
    struct MockQueuePair {
        uint32_t qp_num = 12345;
        RDMATransportMode transport = RDMATransportMode::RC;
        uint32_t sq_depth = 256;
        uint32_t rq_depth = 256;
        bool connected = false;
    };
    
    struct MockMemoryRegion {
        void* addr = nullptr;
        size_t length = 0;
        uint32_t lkey = 0x12345;  // Local key
        uint32_t rkey = 0x54321;  // Remote key
    };
    
    MockRDMADevice device_;
    MockQueuePair qp_;
    std::vector<MockMemoryRegion> memory_regions_;
    
    // Helper methods
    void parse_uri(const std::string& uri);
    bool setup_rdma_device();
    bool create_queue_pair();
    bool establish_connection();
    void run_completion_handler();
    void run_receive_handler();
    void update_latency_stats(uint64_t latency_ns);
    
    // Simulated RDMA operations
    bool mock_post_send(const void* data, size_t length);
    bool mock_post_recv();
    void mock_poll_completions();
};

} // namespace detail

// Factory functions for RDMA channels
namespace rdma {

/**
 * @brief Create an RDMA server channel
 * @param port Port to listen on
 * @param buffer_size Buffer size for operations
 * @param config RDMA configuration
 * @return Unique pointer to the channel
 */
std::unique_ptr<Channel> create_server(uint16_t port, 
                                      size_t buffer_size);

/**
 * @brief Create an RDMA client channel
 * @param host Remote host address
 * @param port Remote port
 * @param buffer_size Buffer size for operations  
 * @param config RDMA configuration
 * @return Unique pointer to the channel
 */
std::unique_ptr<Channel> create_client(const std::string& host, uint16_t port,
                                      size_t buffer_size);

/**
 * @brief Create an RDMA channel with specific role
 * @param uri RDMA URI (rdma://host:port)
 * @param role Server or client role
 * @param buffer_size Buffer size for operations
 * @param config RDMA configuration
 * @return Unique pointer to the channel
 */
std::unique_ptr<Channel> create_channel(const std::string& uri,
                                       detail::RDMARole role,
                                       size_t buffer_size);

} // namespace rdma

} // namespace psyne