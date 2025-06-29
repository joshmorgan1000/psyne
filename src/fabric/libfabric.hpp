/**
 * @file libfabric.hpp
 * @brief Unified high-performance fabric interface using libfabric
 * 
 * This provides a unified API for various high-performance networking fabrics
 * including InfiniBand, Ethernet (RoCE), Intel Omni-Path, and others.
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#pragma once

#include <psyne/channel.hpp>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

// Forward declarations for libfabric types
struct fid_fabric;
struct fid_domain;
struct fid_ep;
struct fid_cq;
struct fid_mr;
struct fid_av;
struct fi_info;
struct fi_context;

namespace psyne {
namespace fabric {

/**
 * @brief Fabric provider types
 */
enum class ProviderType {
    VERBS,      ///< InfiniBand Verbs provider
    SOCKETS,    ///< Berkeley sockets provider
    UDP,        ///< UDP provider
    TCP,        ///< TCP provider
    SHM,        ///< Shared memory provider
    PSM2,       ///< Intel PSM2 provider (Omni-Path)
    MLX,        ///< Mellanox provider
    BGQ,        ///< IBM Blue Gene/Q provider
    AUTO        ///< Auto-select best provider
};

/**
 * @brief Fabric endpoint types
 */
enum class EndpointType {
    MSG,        ///< Message endpoint (connection-oriented)
    RDM,        ///< Reliable datagram endpoint
    DGRAM       ///< Datagram endpoint
};

/**
 * @brief Fabric capabilities
 */
struct FabricCapabilities {
    std::string provider_name;
    std::string fabric_name;
    std::string domain_name;
    uint64_t max_msg_size;
    uint64_t max_rma_size;
    uint32_t max_ep_cnt;
    uint32_t max_cq_cnt;
    bool supports_rma;
    bool supports_atomic;
    bool supports_multicast;
    bool supports_tagged;
};

/**
 * @brief Fabric memory region for zero-copy operations
 */
class FabricMemoryRegion {
public:
    FabricMemoryRegion(fid_domain* domain, void* addr, size_t length, uint64_t access);
    ~FabricMemoryRegion();
    
    void* addr() const { return addr_; }
    size_t length() const { return length_; }
    uint64_t key() const;
    fid_mr* mr() const { return mr_; }
    
private:
    void* addr_;
    size_t length_;
    fid_mr* mr_;
};

/**
 * @brief Fabric completion queue wrapper
 */
class FabricCompletionQueue {
public:
    FabricCompletionQueue(fid_domain* domain, size_t size);
    ~FabricCompletionQueue();
    
    int read(void* buf, size_t count);
    int read_error(void* buf, size_t count);
    void signal();
    fid_cq* cq() const { return cq_; }
    
private:
    fid_cq* cq_;
};

/**
 * @brief Fabric endpoint wrapper
 */
class FabricEndpoint {
public:
    FabricEndpoint(fid_domain* domain, fi_info* info, 
                   EndpointType type = EndpointType::MSG);
    ~FabricEndpoint();
    
    bool bind_cq(FabricCompletionQueue* cq, uint64_t flags);
    bool enable();
    
    // Message operations
    ssize_t send(const void* buf, size_t len, void* context = nullptr);
    ssize_t recv(void* buf, size_t len, void* context = nullptr);
    ssize_t post_recv(void* buf, size_t len, void* context = nullptr);
    
    // RMA operations
    ssize_t read(void* buf, size_t len, uint64_t addr, uint64_t key, void* context = nullptr);
    ssize_t write(const void* buf, size_t len, uint64_t addr, uint64_t key, void* context = nullptr);
    
    // Atomic operations
    ssize_t atomic_compare_swap(void* result, const void* compare, const void* swap,
                               uint64_t addr, uint64_t key, void* context = nullptr);
    ssize_t atomic_fetch_add(void* result, const void* operand,
                            uint64_t addr, uint64_t key, void* context = nullptr);
    
    fid_ep* ep() const { return ep_; }
    
private:
    fid_ep* ep_;
    EndpointType type_;
};

/**
 * @brief High-performance fabric channel using libfabric
 */
class LibfabricChannel : public Channel {
public:
    LibfabricChannel(const std::string& provider, const std::string& node,
                     const std::string& service, EndpointType ep_type = EndpointType::MSG,
                     size_t buffer_size = 1024 * 1024);
    ~LibfabricChannel() override;
    
    // Channel interface
    size_t send(const void* data, size_t size, uint32_t type_id = 0) override;
    size_t receive(void* buffer, size_t buffer_size, uint32_t* type_id = nullptr) override;
    bool try_send(const void* data, size_t size, uint32_t type_id = 0) override;
    bool try_receive(void* buffer, size_t buffer_size, size_t* received_size = nullptr, 
                     uint32_t* type_id = nullptr) override;
    
    // Fabric-specific operations
    
    /**
     * @brief Register memory for fabric operations
     */
    std::shared_ptr<FabricMemoryRegion> register_memory(void* addr, size_t length);
    
    /**
     * @brief Remote memory access (RMA) read
     */
    bool rma_read(void* local_addr, size_t length, uint64_t remote_addr, uint64_t remote_key);
    
    /**
     * @brief Remote memory access (RMA) write
     */
    bool rma_write(const void* local_addr, size_t length, uint64_t remote_addr, uint64_t remote_key);
    
    /**
     * @brief Atomic compare and swap
     */
    bool atomic_compare_swap(uint64_t* result, uint64_t compare, uint64_t swap,
                            uint64_t remote_addr, uint64_t remote_key);
    
    /**
     * @brief Atomic fetch and add
     */
    bool atomic_fetch_add(uint64_t* result, uint64_t operand,
                         uint64_t remote_addr, uint64_t remote_key);
    
    /**
     * @brief Connect to remote endpoint
     */
    bool connect(const std::string& remote_node, const std::string& remote_service);
    
    /**
     * @brief Listen for incoming connections
     */
    bool listen();
    
    /**
     * @brief Accept incoming connection
     */
    std::unique_ptr<LibfabricChannel> accept();
    
    /**
     * @brief Get fabric capabilities
     */
    FabricCapabilities get_capabilities() const;
    
    /**
     * @brief Get performance statistics
     */
    struct Stats {
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        uint64_t messages_sent = 0;
        uint64_t messages_received = 0;
        uint64_t rma_reads = 0;
        uint64_t rma_writes = 0;
        uint64_t atomic_ops = 0;
        uint64_t completion_errors = 0;
        double avg_latency_us = 0.0;
    };
    
    Stats get_stats() const;
    
protected:
    // Fabric resources
    fid_fabric* fabric_;
    fid_domain* domain_;
    std::unique_ptr<FabricEndpoint> endpoint_;
    std::unique_ptr<FabricCompletionQueue> tx_cq_;
    std::unique_ptr<FabricCompletionQueue> rx_cq_;
    fid_av* address_vector_;
    
    // Memory management
    std::vector<std::shared_ptr<FabricMemoryRegion>> memory_regions_;
    std::shared_ptr<FabricMemoryRegion> send_mr_;
    std::shared_ptr<FabricMemoryRegion> recv_mr_;
    
    // Connection state
    std::atomic<bool> connected_;
    std::string provider_name_;
    std::string node_;
    std::string service_;
    EndpointType endpoint_type_;
    
    // Threading
    std::thread completion_thread_;
    std::atomic<bool> running_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    Stats stats_;
    
    // Helper methods
    bool init_fabric(const std::string& provider);
    bool create_domain();
    bool create_endpoint();
    bool create_completion_queues();
    bool setup_memory_regions();
    void cleanup_resources();
    void completion_handler();
    void process_tx_completion();
    void process_rx_completion();
    uint64_t get_timestamp_ns() const;
};

/**
 * @brief Fabric provider discovery and selection
 */
class FabricProvider {
public:
    /**
     * @brief List available fabric providers
     */
    static std::vector<FabricCapabilities> list_providers();
    
    /**
     * @brief Get best provider for given requirements
     */
    static std::string select_provider(bool requires_rma = false, 
                                      bool requires_atomic = false,
                                      uint64_t min_msg_size = 0);
    
    /**
     * @brief Check if libfabric is available
     */
    static bool is_available();
    
    /**
     * @brief Get provider-specific optimizations
     */
    static std::unordered_map<std::string, std::string> get_provider_hints(const std::string& provider);
};

/**
 * @brief Factory functions for fabric channels
 */

/**
 * @brief Create a fabric server channel
 */
std::unique_ptr<LibfabricChannel> create_fabric_server(
    const std::string& service,
    const std::string& provider = "auto",
    EndpointType ep_type = EndpointType::MSG,
    size_t buffer_size = 1024 * 1024);

/**
 * @brief Create a fabric client channel
 */
std::unique_ptr<LibfabricChannel> create_fabric_client(
    const std::string& node,
    const std::string& service,
    const std::string& provider = "auto",
    EndpointType ep_type = EndpointType::MSG,
    size_t buffer_size = 1024 * 1024);

/**
 * @brief Create fabric channel with automatic provider selection
 */
std::unique_ptr<LibfabricChannel> create_auto_fabric_channel(
    const std::string& node,
    const std::string& service,
    bool require_rma = false,
    bool require_atomic = false,
    size_t buffer_size = 1024 * 1024);

} // namespace fabric
} // namespace psyne