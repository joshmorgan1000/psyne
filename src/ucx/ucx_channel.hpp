/**
 * @file ucx_channel.hpp
 * @brief UCX (Unified Communication X) channel implementation for automatic transport selection
 * 
 * UCX provides a unified communication framework that automatically selects
 * the best available transport (TCP, InfiniBand, shared memory, etc.) and
 * optimizes for the specific workload characteristics.
 * 
 * Key features:
 * - Automatic transport selection and optimization
 * - Multi-rail communication support
 * - Built-in GPU memory support (CUDA/ROCm)
 * - MPI and ML framework integration
 * - Runtime performance tuning
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

// Forward declarations for UCX types
extern "C" {
    typedef struct ucp_context ucp_context_t;
    typedef struct ucp_worker ucp_worker_t;
    typedef struct ucp_ep ucp_ep_t;
    typedef struct ucp_mem ucp_mem_t;
    typedef struct ucp_request ucp_request_t;
    typedef struct ucp_config ucp_config_t;
    typedef struct ucp_address ucp_address_t;
    typedef struct ucp_tag_recv_info ucp_tag_recv_info_t;
    typedef uint64_t ucp_tag_t;
    typedef void* ucp_rkey_h;
}

namespace psyne {
namespace ucx {

/**
 * @brief UCX transport selection modes
 */
enum class TransportMode {
    AUTO,           ///< Automatic transport selection
    TCP_ONLY,       ///< TCP transport only
    RDMA_ONLY,      ///< RDMA/InfiniBand only
    SHM_ONLY,       ///< Shared memory only
    MULTI_RAIL,     ///< Multiple transports simultaneously
    GPU_DIRECT      ///< Optimized for GPU-to-GPU transfers
};

/**
 * @brief UCX memory types for GPU integration
 */
enum class MemoryType {
    HOST,           ///< Host/CPU memory
    CUDA,           ///< CUDA GPU memory
    ROCM,           ///< ROCm GPU memory
    HOST_PINNED     ///< Pinned host memory for GPU transfers
};

/**
 * @brief UCX transport information
 */
struct TransportInfo {
    std::string name;
    std::string device;
    double bandwidth_mbps;
    double latency_us;
    bool supports_rma;
    bool supports_atomic;
    bool supports_gpu;
    bool active;
};

/**
 * @brief UCX communication capabilities
 */
struct UCXCapabilities {
    std::vector<TransportInfo> available_transports;
    bool supports_tag_matching;
    bool supports_stream_api;
    bool supports_rma;
    bool supports_atomic;
    bool supports_gpu_memory;
    bool supports_multi_rail;
    uint64_t max_message_size;
    uint64_t max_iov_count;
};

/**
 * @brief UCX memory registration for zero-copy operations
 */
class UCXMemoryRegion {
public:
    UCXMemoryRegion(ucp_context_t* context, void* addr, size_t length, MemoryType type);
    ~UCXMemoryRegion();
    
    void* addr() const { return addr_; }
    size_t length() const { return length_; }
    MemoryType type() const { return type_; }
    ucp_mem_t* mem_handle() const { return mem_handle_; }
    ucp_rkey_h remote_key() const { return rkey_; }
    
    // Pack remote key for transmission
    std::vector<uint8_t> pack_rkey() const;
    
private:
    void* addr_;
    size_t length_;
    MemoryType type_;
    ucp_mem_t* mem_handle_;
    ucp_rkey_h rkey_;
    ucp_context_t* context_;
};

/**
 * @brief UCX endpoint for peer-to-peer communication
 */
class UCXEndpoint {
public:
    UCXEndpoint(ucp_worker_t* worker, const std::string& peer_address);
    ~UCXEndpoint();
    
    // Blocking send/receive operations
    bool send(const void* buffer, size_t length, ucp_tag_t tag = 0);
    bool receive(void* buffer, size_t length, ucp_tag_t tag = 0, size_t* received_length = nullptr);
    
    // Non-blocking send/receive operations
    ucp_request_t* send_nb(const void* buffer, size_t length, ucp_tag_t tag = 0);
    ucp_request_t* receive_nb(void* buffer, size_t length, ucp_tag_t tag = 0);
    
    // RMA operations
    bool put(const void* local_buffer, size_t length, uint64_t remote_addr, ucp_rkey_h rkey);
    bool get(void* local_buffer, size_t length, uint64_t remote_addr, ucp_rkey_h rkey);
    
    // Atomic operations
    bool atomic_add(uint64_t* result, uint64_t operand, uint64_t remote_addr, ucp_rkey_h rkey);
    bool atomic_cas(uint64_t* result, uint64_t compare, uint64_t swap, uint64_t remote_addr, ucp_rkey_h rkey);
    
    // Stream API (for reliable ordered delivery)
    bool stream_send(const void* buffer, size_t length);
    bool stream_receive(void* buffer, size_t length, size_t* received_length = nullptr);
    
    // Connection management
    bool is_connected() const { return connected_; }
    void close();
    
    // Progress communication
    void progress();
    
private:
    ucp_worker_t* worker_;
    ucp_ep_t* endpoint_;
    std::atomic<bool> connected_;
    std::string peer_address_;
};

/**
 * @brief UCX worker for handling communication operations
 */
class UCXWorker {
public:
    UCXWorker(ucp_context_t* context, bool enable_wakeup = true);
    ~UCXWorker();
    
    // Endpoint management
    std::shared_ptr<UCXEndpoint> create_endpoint(const std::string& peer_address);
    void remove_endpoint(const std::string& peer_address);
    
    // Address exchange
    std::vector<uint8_t> get_address() const;
    
    // Progress communication
    void progress();
    void wait_for_events(int timeout_ms = -1);
    void signal();
    
    // Request management
    bool test_request(ucp_request_t* request);
    void wait_request(ucp_request_t* request);
    void cancel_request(ucp_request_t* request);
    
    ucp_worker_t* worker() const { return worker_; }
    
private:
    ucp_context_t* context_;
    ucp_worker_t* worker_;
    std::unordered_map<std::string, std::shared_ptr<UCXEndpoint>> endpoints_;
    std::mutex endpoints_mutex_;
    bool wakeup_enabled_;
};

/**
 * @brief High-level UCX channel implementation
 */
class UCXChannel : public Channel {
public:
    UCXChannel(const std::string& name, TransportMode mode = TransportMode::AUTO,
               size_t buffer_size = 1024 * 1024, bool enable_gpu = false);
    ~UCXChannel() override;
    
    // Channel interface
    size_t send(const void* data, size_t size, uint32_t type_id = 0) override;
    size_t receive(void* buffer, size_t buffer_size, uint32_t* type_id = nullptr) override;
    bool try_send(const void* data, size_t size, uint32_t type_id = 0) override;
    bool try_receive(void* buffer, size_t buffer_size, size_t* received_size = nullptr, 
                     uint32_t* type_id = nullptr) override;
    
    // UCX-specific operations
    
    /**
     * @brief Register memory for zero-copy operations
     */
    std::shared_ptr<UCXMemoryRegion> register_memory(void* addr, size_t length, 
                                                     MemoryType type = MemoryType::HOST);
    
    /**
     * @brief Send registered memory with zero-copy
     */
    bool send_memory_region(const UCXMemoryRegion& region, size_t offset = 0, size_t length = 0);
    
    /**
     * @brief Receive into registered memory with zero-copy
     */
    bool receive_memory_region(const UCXMemoryRegion& region, size_t offset = 0, size_t length = 0);
    
    /**
     * @brief Remote memory access - put operation
     */
    bool rma_put(const void* local_buffer, size_t length, const std::string& peer,
                 uint64_t remote_addr, const std::vector<uint8_t>& remote_key);
    
    /**
     * @brief Remote memory access - get operation
     */
    bool rma_get(void* local_buffer, size_t length, const std::string& peer,
                 uint64_t remote_addr, const std::vector<uint8_t>& remote_key);
    
    /**
     * @brief Atomic fetch and add
     */
    bool atomic_fadd(uint64_t* result, uint64_t operand, const std::string& peer,
                     uint64_t remote_addr, const std::vector<uint8_t>& remote_key);
    
    /**
     * @brief Connect to remote peer
     */
    bool connect(const std::string& peer_address);
    
    /**
     * @brief Listen for incoming connections
     */
    bool listen(const std::string& address);
    
    /**
     * @brief Get UCX capabilities
     */
    UCXCapabilities get_capabilities() const;
    
    /**
     * @brief Get transport statistics
     */
    struct UCXStats {
        uint64_t messages_sent = 0;
        uint64_t messages_received = 0;
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        uint64_t rma_operations = 0;
        uint64_t atomic_operations = 0;
        uint64_t zero_copy_sends = 0;
        uint64_t zero_copy_receives = 0;
        double avg_latency_us = 0.0;
        double avg_bandwidth_mbps = 0.0;
        std::vector<TransportInfo> active_transports;
    };
    
    UCXStats get_stats() const;
    
    /**
     * @brief Force progress on communication
     */
    void progress();
    
protected:
    // UCX resources
    ucp_context_t* context_;
    std::unique_ptr<UCXWorker> worker_;
    
    // Memory management
    std::vector<std::shared_ptr<UCXMemoryRegion>> memory_regions_;
    
    // Connection management
    std::unordered_map<std::string, std::shared_ptr<UCXEndpoint>> peers_;
    std::mutex peers_mutex_;
    
    // Configuration
    TransportMode transport_mode_;
    bool gpu_enabled_;
    
    // Threading
    std::thread progress_thread_;
    std::atomic<bool> running_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    UCXStats stats_;
    
    // Helper methods
    bool init_ucx_context(TransportMode mode, bool enable_gpu);
    void cleanup_resources();
    void progress_worker();
    ucp_config_t* create_ucx_config(TransportMode mode, bool enable_gpu);
    void update_transport_stats();
    uint64_t get_timestamp_us() const;
};

/**
 * @brief UCX context manager for sharing resources
 */
class UCXContextManager {
public:
    static UCXContextManager& instance();
    
    ucp_context_t* get_context(TransportMode mode, bool enable_gpu);
    void release_context(ucp_context_t* context);
    
    UCXCapabilities get_system_capabilities();
    std::vector<TransportInfo> discover_transports();
    std::string select_optimal_transport(bool require_gpu = false, bool require_rma = false);
    
private:
    UCXContextManager() = default;
    ~UCXContextManager();
    
    struct ContextInfo {
        ucp_context_t* context;
        TransportMode mode;
        bool gpu_enabled;
        int reference_count;
    };
    
    std::vector<ContextInfo> contexts_;
    std::mutex contexts_mutex_;
    
    UCXCapabilities cached_capabilities_;
    bool capabilities_cached_ = false;
};

/**
 * @brief Factory functions for UCX channels
 */

/**
 * @brief Create UCX server channel
 */
std::unique_ptr<UCXChannel> create_ucx_server(
    const std::string& address,
    TransportMode mode = TransportMode::AUTO,
    size_t buffer_size = 1024 * 1024,
    bool enable_gpu = false);

/**
 * @brief Create UCX client channel
 */
std::unique_ptr<UCXChannel> create_ucx_client(
    const std::string& server_address,
    TransportMode mode = TransportMode::AUTO,
    size_t buffer_size = 1024 * 1024,
    bool enable_gpu = false);

/**
 * @brief Create UCX channel with automatic optimization
 */
std::unique_ptr<UCXChannel> create_auto_ucx_channel(
    const std::string& peer_address,
    bool require_gpu = false,
    bool require_rma = false,
    size_t buffer_size = 1024 * 1024);

/**
 * @brief Create UCX channel optimized for ML workloads
 */
std::unique_ptr<UCXChannel> create_ml_ucx_channel(
    const std::string& peer_address,
    bool enable_cuda = true,
    size_t buffer_size = 64 * 1024 * 1024);  // Larger buffer for ML

} // namespace ucx
} // namespace psyne