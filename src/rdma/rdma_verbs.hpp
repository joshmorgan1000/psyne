/**
 * @file rdma_verbs.hpp
 * @brief Real RDMA/InfiniBand implementation using Verbs API
 * 
 * This provides actual RDMA functionality using the ibverbs library
 * for ultra-low latency communication on InfiniBand and RoCE networks.
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

// Forward declarations for ibverbs types
struct ibv_device;
struct ibv_context;
struct ibv_pd;
struct ibv_mr;
struct ibv_cq;
struct ibv_qp;
struct ibv_comp_channel;
struct ibv_ah;
struct ibv_wc;
struct ibv_send_wr;
struct ibv_recv_wr;
struct ibv_sge;

namespace psyne {
namespace rdma {

/**
 * @brief RDMA transport types
 */
enum class TransportType {
    RC,  ///< Reliable Connection
    UC,  ///< Unreliable Connection  
    UD,  ///< Unreliable Datagram
    XRC  ///< Extended Reliable Connection
};

/**
 * @brief RDMA device capabilities
 */
struct DeviceCapabilities {
    std::string name;
    uint64_t max_mr_size;
    uint32_t max_qp;
    uint32_t max_cq;
    uint32_t max_qp_wr;
    uint32_t max_sge;
    uint32_t max_inline_data;
    bool atomic_cap;
    bool odp_support;  // On-Demand Paging
};

/**
 * @brief RDMA connection parameters
 */
struct ConnectionParams {
    uint32_t qp_num;
    uint16_t lid;
    uint8_t gid[16];
    uint32_t psn;  // Packet sequence number
    uint8_t port_num;
    uint8_t sl;    // Service level
};

/**
 * @brief RDMA memory region wrapper
 */
class MemoryRegion {
public:
    MemoryRegion(ibv_pd* pd, void* addr, size_t length, int access_flags);
    ~MemoryRegion();
    
    void* addr() const { return addr_; }
    size_t length() const { return length_; }
    uint32_t lkey() const;
    uint32_t rkey() const;
    ibv_mr* mr() const { return mr_; }
    
private:
    void* addr_;
    size_t length_;
    ibv_mr* mr_;
};

/**
 * @brief RDMA completion queue wrapper
 */
class CompletionQueue {
public:
    CompletionQueue(ibv_context* context, int cq_size, 
                    ibv_comp_channel* channel = nullptr);
    ~CompletionQueue();
    
    int poll(ibv_wc* wc, int num_entries);
    void request_notify(bool solicited_only = false);
    ibv_cq* cq() const { return cq_; }
    
private:
    ibv_cq* cq_;
    ibv_comp_channel* channel_;
};

/**
 * @brief RDMA queue pair wrapper
 */
class QueuePair {
public:
    QueuePair(ibv_pd* pd, ibv_cq* send_cq, ibv_cq* recv_cq, 
              TransportType type, uint32_t max_send_wr, 
              uint32_t max_recv_wr, uint32_t max_inline_data);
    ~QueuePair();
    
    bool modify_to_init(uint8_t port_num);
    bool modify_to_rtr(const ConnectionParams& remote_params);
    bool modify_to_rts();
    
    uint32_t qp_num() const;
    ibv_qp* qp() const { return qp_; }
    
    bool post_send(ibv_send_wr* wr);
    bool post_recv(ibv_recv_wr* wr);
    
private:
    ibv_qp* qp_;
    TransportType type_;
};

/**
 * @brief High-performance RDMA channel using real Verbs API
 */
class RDMAVerbsChannel : public Channel {
public:
    RDMAVerbsChannel(const std::string& device_name, uint8_t port_num,
                     size_t buffer_size, TransportType transport = TransportType::RC);
    ~RDMAVerbsChannel() override;
    
    // Channel interface
    size_t send(const void* data, size_t size, uint32_t type_id = 0) override;
    size_t receive(void* buffer, size_t buffer_size, uint32_t* type_id = nullptr) override;
    bool try_send(const void* data, size_t size, uint32_t type_id = 0) override;
    bool try_receive(void* buffer, size_t buffer_size, size_t* received_size = nullptr, 
                     uint32_t* type_id = nullptr) override;
    
    // RDMA-specific operations
    
    /**
     * @brief Register memory for RDMA operations
     */
    std::shared_ptr<MemoryRegion> register_memory(void* addr, size_t length);
    
    /**
     * @brief RDMA Write operation (zero-copy remote write)
     */
    bool rdma_write(const void* local_addr, size_t length, 
                    uint64_t remote_addr, uint32_t rkey);
    
    /**
     * @brief RDMA Read operation (zero-copy remote read)
     */
    bool rdma_read(void* local_addr, size_t length,
                   uint64_t remote_addr, uint32_t rkey);
    
    /**
     * @brief RDMA Compare and Swap atomic operation
     */
    bool rdma_compare_swap(uint64_t* local_addr, uint64_t compare, 
                          uint64_t swap, uint64_t remote_addr, uint32_t rkey);
    
    /**
     * @brief RDMA Fetch and Add atomic operation
     */
    bool rdma_fetch_add(uint64_t* local_addr, uint64_t add,
                       uint64_t remote_addr, uint32_t rkey);
    
    /**
     * @brief Connect to remote peer
     */
    bool connect(const std::string& remote_addr, uint16_t remote_port);
    
    /**
     * @brief Listen for incoming connections
     */
    bool listen(uint16_t port);
    
    /**
     * @brief Accept incoming connection
     */
    std::unique_ptr<RDMAVerbsChannel> accept();
    
    /**
     * @brief Get local connection parameters
     */
    ConnectionParams get_local_params() const;
    
    /**
     * @brief Exchange connection parameters with remote
     */
    bool exchange_params(const ConnectionParams& local, ConnectionParams& remote);
    
    /**
     * @brief Get device capabilities
     */
    DeviceCapabilities get_capabilities() const;
    
    /**
     * @brief Enable/disable inline data optimization
     */
    void set_inline_threshold(uint32_t threshold) { inline_threshold_ = threshold; }
    
    /**
     * @brief Get performance statistics
     */
    struct Stats {
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        uint64_t rdma_writes = 0;
        uint64_t rdma_reads = 0;
        uint64_t send_completions = 0;
        uint64_t recv_completions = 0;
        uint64_t completion_errors = 0;
        double avg_latency_us = 0.0;
    };
    
    Stats get_stats() const;
    
protected:
    // RDMA resources
    ibv_device* device_;
    ibv_context* context_;
    ibv_pd* pd_;
    std::unique_ptr<CompletionQueue> send_cq_;
    std::unique_ptr<CompletionQueue> recv_cq_;
    std::unique_ptr<QueuePair> qp_;
    
    // Memory management
    std::vector<std::shared_ptr<MemoryRegion>> memory_regions_;
    std::shared_ptr<MemoryRegion> send_mr_;
    std::shared_ptr<MemoryRegion> recv_mr_;
    
    // Connection state
    std::atomic<bool> connected_;
    ConnectionParams local_params_;
    ConnectionParams remote_params_;
    uint8_t port_num_;
    TransportType transport_;
    
    // Performance tuning
    uint32_t inline_threshold_;
    uint32_t max_send_wr_;
    uint32_t max_recv_wr_;
    uint32_t max_sge_;
    
    // Threading
    std::thread completion_thread_;
    std::atomic<bool> running_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    Stats stats_;
    
    // Helper methods
    bool init_device(const std::string& device_name);
    bool create_resources();
    void destroy_resources();
    void completion_handler();
    void process_completion(const ibv_wc& wc);
    bool post_receive_buffers();
    uint64_t get_timestamp_ns() const;
};

/**
 * @brief RDMA channel with GPUDirect support
 */
class RDMAGPUDirectChannel : public RDMAVerbsChannel {
public:
    RDMAGPUDirectChannel(const std::string& device_name, uint8_t port_num,
                         size_t buffer_size, TransportType transport = TransportType::RC);
    
    /**
     * @brief Register GPU memory for GPUDirect RDMA
     */
    std::shared_ptr<MemoryRegion> register_gpu_memory(void* gpu_addr, size_t length);
    
    /**
     * @brief RDMA Write from GPU memory
     */
    bool rdma_write_from_gpu(const void* local_gpu_addr, size_t length,
                            uint64_t remote_addr, uint32_t rkey);
    
    /**
     * @brief RDMA Read to GPU memory
     */
    bool rdma_read_to_gpu(void* local_gpu_addr, size_t length,
                         uint64_t remote_addr, uint32_t rkey);
    
private:
    bool gpu_direct_enabled_;
};

/**
 * @brief Factory functions for RDMA channels
 */

/**
 * @brief Create an RDMA server channel
 */
std::unique_ptr<RDMAVerbsChannel> create_rdma_server(
    uint16_t port, 
    size_t buffer_size = 1024 * 1024,
    const std::string& device = "",
    TransportType transport = TransportType::RC);

/**
 * @brief Create an RDMA client channel
 */
std::unique_ptr<RDMAVerbsChannel> create_rdma_client(
    const std::string& server_addr,
    uint16_t server_port,
    size_t buffer_size = 1024 * 1024,
    const std::string& device = "",
    TransportType transport = TransportType::RC);

/**
 * @brief List available RDMA devices
 */
std::vector<DeviceCapabilities> list_rdma_devices();

/**
 * @brief Check if RDMA is available on the system
 */
bool is_rdma_available();

} // namespace rdma
} // namespace psyne