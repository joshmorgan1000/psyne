/**
 * @file gpudirect_message.cpp
 * @brief GPUDirect RDMA-aware message implementation
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#if defined(PSYNE_CUDA_ENABLED) && defined(PSYNE_RDMA_SUPPORT)

#include "gpudirect_message.hpp"
#include <iostream>
#include <chrono>
#include <atomic>

namespace psyne {
namespace gpu {

// GPUDirectChannel implementation

GPUDirectChannel::GPUDirectChannel(const std::string& device_name, uint8_t port_num, size_t buffer_size)
    : stats_{} {
    
    try {
        // Create the underlying RDMA GPUDirect channel
        rdma_channel_ = std::make_unique<rdma::RDMAGPUDirectChannel>(
            device_name, port_num, buffer_size, rdma::TransportType::RC);
        
        std::cout << "GPUDirect channel created successfully on device: " 
                  << device_name << " port: " << static_cast<int>(port_num) << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to create GPUDirect channel: " << e.what() << std::endl;
        throw;
    }
}

bool GPUDirectChannel::connect(const std::string& remote_addr, uint16_t remote_port) {
    if (!rdma_channel_) {
        std::cerr << "RDMA channel not initialized" << std::endl;
        return false;
    }
    
    std::cout << "Connecting to GPUDirect peer: " << remote_addr << ":" << remote_port << std::endl;
    return rdma_channel_->connect(remote_addr, remote_port);
}

bool GPUDirectChannel::listen(uint16_t port) {
    if (!rdma_channel_) {
        std::cerr << "RDMA channel not initialized" << std::endl;
        return false;
    }
    
    std::cout << "Listening for GPUDirect connections on port: " << port << std::endl;
    return rdma_channel_->listen(port);
}

std::shared_ptr<rdma::MemoryRegion> GPUDirectChannel::get_or_register_memory(void* gpu_addr, size_t length) {
    // Check if already registered
    auto it = memory_cache_.find(gpu_addr);
    if (it != memory_cache_.end()) {
        stats_.registration_cache_hits++;
        return it->second;
    }
    
    // Register new memory region
    auto mr = rdma_channel_->register_gpu_memory(gpu_addr, length);
    if (mr) {
        memory_cache_[gpu_addr] = mr;
        stats_.registration_cache_misses++;
        
        std::cout << "Registered new GPU memory region: " << gpu_addr 
                  << " size: " << length << " bytes" << std::endl;
    }
    
    return mr;
}

void GPUDirectChannel::update_stats(size_t bytes_transferred, double time_us) {
    stats_.bytes_transferred += bytes_transferred;
    
    // Update running average of transfer time
    static std::atomic<uint64_t> transfer_count{0};
    uint64_t count = transfer_count.fetch_add(1) + 1;
    
    stats_.avg_transfer_time_us = 
        (stats_.avg_transfer_time_us * (count - 1) + time_us) / count;
}

// Explicit template instantiations for common types
template std::shared_ptr<rdma::MemoryRegion> GPUDirectChannel::register_gpu_vector<float>(GPUVector<float>&, GPUContext&);
template std::shared_ptr<rdma::MemoryRegion> GPUDirectChannel::register_gpu_vector<double>(GPUVector<double>&, GPUContext&);
template std::shared_ptr<rdma::MemoryRegion> GPUDirectChannel::register_gpu_vector<int32_t>(GPUVector<int32_t>&, GPUContext&);

template bool GPUDirectChannel::send_gpu_vector<float>(const GPUVector<float>&, GPUContext&, uint64_t, uint32_t);
template bool GPUDirectChannel::send_gpu_vector<double>(const GPUVector<double>&, GPUContext&, uint64_t, uint32_t);
template bool GPUDirectChannel::send_gpu_vector<int32_t>(const GPUVector<int32_t>&, GPUContext&, uint64_t, uint32_t);

template bool GPUDirectChannel::receive_gpu_vector<float>(GPUVector<float>&, GPUContext&, uint64_t, uint32_t);
template bool GPUDirectChannel::receive_gpu_vector<double>(GPUVector<double>&, GPUContext&, uint64_t, uint32_t);
template bool GPUDirectChannel::receive_gpu_vector<int32_t>(GPUVector<int32_t>&, GPUContext&, uint64_t, uint32_t);

} // namespace gpu
} // namespace psyne

#endif // PSYNE_CUDA_ENABLED && PSYNE_RDMA_SUPPORT
