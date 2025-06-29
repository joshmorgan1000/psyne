/**
 * @file gpudirect_message.hpp
 * @brief GPUDirect RDMA-aware message types for zero-copy GPU-to-GPU transfers
 * 
 * Provides high-level integration between CUDA GPU buffers and RDMA
 * for direct GPU-to-GPU communication across the network.
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#pragma once

#if defined(PSYNE_CUDA_ENABLED) && defined(PSYNE_RDMA_SUPPORT)

#include "gpu_message.hpp"
#include "../rdma/rdma_verbs.hpp"
#include <memory>
#include <unordered_map>

namespace psyne {
namespace gpu {

/**
 * @brief GPUDirect RDMA-aware channel that bridges GPU and RDMA operations
 */
class GPUDirectChannel {
public:
    /**
     * @brief Constructor
     * @param device_name RDMA device name
     * @param port_num RDMA port number
     * @param buffer_size Default buffer size
     */
    GPUDirectChannel(const std::string& device_name, uint8_t port_num,
                     size_t buffer_size = 1024 * 1024);
    
    ~GPUDirectChannel() = default;
    
    /**
     * @brief Register a CUDA buffer for GPUDirect RDMA
     * @param cuda_buffer The CUDA buffer to register
     * @return Memory region handle for RDMA operations
     */
    template<typename T>
    std::shared_ptr<rdma::MemoryRegion> register_gpu_vector(GPUVector<T>& gpu_vector, GPUContext& context);
    
    /**
     * @brief Send GPU vector directly via RDMA (zero-copy)
     * @param gpu_vector The GPU vector to send
     * @param context GPU context
     * @param remote_addr Remote memory address
     * @param rkey Remote key
     * @return True if successful
     */
    template<typename T>
    bool send_gpu_vector(const GPUVector<T>& gpu_vector, GPUContext& context,
                        uint64_t remote_addr, uint32_t rkey);
    
    /**
     * @brief Receive data directly into GPU vector via RDMA (zero-copy)
     * @param gpu_vector The GPU vector to receive into
     * @param context GPU context
     * @param remote_addr Remote memory address
     * @param rkey Remote key
     * @return True if successful
     */
    template<typename T>
    bool receive_gpu_vector(GPUVector<T>& gpu_vector, GPUContext& context,
                           uint64_t remote_addr, uint32_t rkey);
    
    /**
     * @brief Connect to remote GPUDirect peer
     */
    bool connect(const std::string& remote_addr, uint16_t remote_port);
    
    /**
     * @brief Listen for incoming GPUDirect connections
     */
    bool listen(uint16_t port);
    
    /**
     * @brief Get the underlying RDMA channel
     */
    rdma::RDMAGPUDirectChannel& rdma_channel() { return *rdma_channel_; }
    
    /**
     * @brief Get performance statistics
     */
    struct GPUDirectStats {
        uint64_t gpu_to_gpu_transfers = 0;
        uint64_t bytes_transferred = 0;
        uint64_t registration_cache_hits = 0;
        uint64_t registration_cache_misses = 0;
        double avg_transfer_time_us = 0.0;
    };
    
    GPUDirectStats get_stats() const { return stats_; }

private:
    std::unique_ptr<rdma::RDMAGPUDirectChannel> rdma_channel_;
    
    // Cache of registered GPU memory regions
    std::unordered_map<void*, std::shared_ptr<rdma::MemoryRegion>> memory_cache_;
    
    // Statistics
    mutable GPUDirectStats stats_;
    
    // Helper methods
    std::shared_ptr<rdma::MemoryRegion> get_or_register_memory(void* gpu_addr, size_t length);
    void update_stats(size_t bytes_transferred, double time_us);
};

/**
 * @brief High-level GPU-to-GPU messaging with automatic registration
 */
template<typename T>
class GPUDirectVector : public GPUVector<T> {
public:
    /**
     * @brief Constructor
     * @param channel GPUDirect channel
     * @param psyne_channel Regular Psyne channel for fallback
     */
    GPUDirectVector(GPUDirectChannel& gpu_direct_channel, Channel& psyne_channel)
        : GPUVector<T>(psyne_channel)
        , gpu_direct_channel_(gpu_direct_channel)
        , is_registered_(false) {}
    
    /**
     * @brief Ensure GPU memory is registered for RDMA
     */
    void ensure_registered(GPUContext& context) {
        if (!is_registered_ && this->is_on_gpu()) {
            auto gpu_buffer = this->to_gpu_buffer(context);
            memory_region_ = gpu_direct_channel_.register_gpu_vector(*this, context);
            is_registered_ = true;
        }
    }
    
    /**
     * @brief Send this vector directly via GPUDirect RDMA
     */
    bool send_direct(GPUContext& context, uint64_t remote_addr, uint32_t rkey) {
        ensure_registered(context);
        return gpu_direct_channel_.send_gpu_vector(*this, context, remote_addr, rkey);
    }
    
    /**
     * @brief Receive data directly into this vector via GPUDirect RDMA
     */
    bool receive_direct(GPUContext& context, uint64_t remote_addr, uint32_t rkey) {
        ensure_registered(context);
        return gpu_direct_channel_.receive_gpu_vector(*this, context, remote_addr, rkey);
    }
    
    /**
     * @brief Get RDMA memory region info for this vector
     */
    std::shared_ptr<rdma::MemoryRegion> memory_region() const { return memory_region_; }

private:
    GPUDirectChannel& gpu_direct_channel_;
    std::shared_ptr<rdma::MemoryRegion> memory_region_;
    bool is_registered_;
};

// Template implementations

template<typename T>
std::shared_ptr<rdma::MemoryRegion> GPUDirectChannel::register_gpu_vector(GPUVector<T>& gpu_vector, GPUContext& context) {
    // Ensure vector is on GPU
    auto gpu_buffer = gpu_vector.to_gpu_buffer(context);
    void* gpu_addr = gpu_buffer->native_handle();
    size_t length = gpu_vector.size() * sizeof(T);
    
    return get_or_register_memory(gpu_addr, length);
}

template<typename T>
bool GPUDirectChannel::send_gpu_vector(const GPUVector<T>& gpu_vector, GPUContext& context,
                                       uint64_t remote_addr, uint32_t rkey) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get GPU buffer
    auto gpu_buffer = const_cast<GPUVector<T>&>(gpu_vector).to_gpu_buffer(context);
    void* gpu_addr = gpu_buffer->native_handle();
    size_t length = gpu_vector.size() * sizeof(T);
    
    // Perform RDMA write from GPU memory
    bool success = rdma_channel_->rdma_write_from_gpu(gpu_addr, length, remote_addr, rkey);
    
    if (success) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        update_stats(length, duration.count());
        stats_.gpu_to_gpu_transfers++;
    }
    
    return success;
}

template<typename T>
bool GPUDirectChannel::receive_gpu_vector(GPUVector<T>& gpu_vector, GPUContext& context,
                                          uint64_t remote_addr, uint32_t rkey) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get GPU buffer
    auto gpu_buffer = gpu_vector.to_gpu_buffer(context);
    void* gpu_addr = gpu_buffer->native_handle();
    size_t length = gpu_vector.size() * sizeof(T);
    
    // Perform RDMA read to GPU memory
    bool success = rdma_channel_->rdma_read_to_gpu(gpu_addr, length, remote_addr, rkey);
    
    if (success) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        update_stats(length, duration.count());
        stats_.gpu_to_gpu_transfers++;
    }
    
    return success;
}

// Convenience aliases
using GPUDirectFloatVector = GPUDirectVector<float>;
using GPUDirectDoubleVector = GPUDirectVector<double>;
using GPUDirectIntVector = GPUDirectVector<int32_t>;

} // namespace gpu
} // namespace psyne

#endif // PSYNE_CUDA_ENABLED && PSYNE_RDMA_SUPPORT