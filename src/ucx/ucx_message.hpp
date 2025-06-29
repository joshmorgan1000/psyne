/**
 * @file ucx_message.hpp
 * @brief UCX-aware message types with automatic transport selection
 * 
 * Provides high-level message abstractions that leverage UCX's automatic
 * transport selection and optimization capabilities. Integrates with GPU
 * memory management for zero-copy operations.
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#pragma once

#if defined(PSYNE_UCX_SUPPORT)

#include "ucx_channel.hpp"
#include "../gpu/gpu_message.hpp"
#include <memory>
#include <vector>
#include <string>
#include <type_traits>

namespace psyne {
namespace ucx {

/**
 * @brief Communication patterns supported by UCX messages
 */
enum class MessagePattern {
    POINT_TO_POINT,     ///< Direct peer-to-peer communication
    BROADCAST,          ///< One-to-many communication
    SCATTER,            ///< Distribute data across peers
    GATHER,             ///< Collect data from peers
    ALL_TO_ALL,         ///< All peers exchange data
    REDUCE,             ///< Collective reduction operation
    ALLREDUCE           ///< Reduction with result to all peers
};

/**
 * @brief UCX message delivery modes
 */
enum class DeliveryMode {
    EAGER,              ///< Small messages sent immediately
    RENDEZVOUS,         ///< Large messages with handshake
    ZERO_COPY,          ///< Direct memory access
    STREAM              ///< Ordered delivery stream
};

/**
 * @brief Base class for UCX-aware messages
 */
class UCXMessage {
public:
    UCXMessage(std::shared_ptr<UCXChannel> channel, MessagePattern pattern = MessagePattern::POINT_TO_POINT)
        : channel_(channel), pattern_(pattern), delivery_mode_(DeliveryMode::EAGER) {}
    
    virtual ~UCXMessage() = default;
    
    // Message properties
    MessagePattern pattern() const { return pattern_; }
    DeliveryMode delivery_mode() const { return delivery_mode_; }
    void set_delivery_mode(DeliveryMode mode) { delivery_mode_ = mode; }
    
    // Tags for message matching
    virtual uint64_t get_tag() const { return 0; }
    virtual void set_tag(uint64_t tag) { tag_ = tag; }
    
    // Virtual interface for sending/receiving
    virtual bool send(const std::string& peer = "") = 0;
    virtual bool receive(const std::string& peer = "") = 0;
    virtual bool broadcast(const std::vector<std::string>& peers) = 0;
    
    // Size and capacity management
    virtual size_t size() const = 0;
    virtual size_t capacity() const = 0;
    virtual void resize(size_t new_size) = 0;
    virtual void reserve(size_t new_capacity) = 0;
    
protected:
    std::shared_ptr<UCXChannel> channel_;
    MessagePattern pattern_;
    DeliveryMode delivery_mode_;
    uint64_t tag_ = 0;
};

/**
 * @brief Template for typed UCX messages
 */
template<typename T>
class UCXVector : public UCXMessage {
public:
    using value_type = T;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    
    UCXVector(std::shared_ptr<UCXChannel> channel, MessagePattern pattern = MessagePattern::POINT_TO_POINT)
        : UCXMessage(channel, pattern), registered_(false) {}
    
    UCXVector(std::shared_ptr<UCXChannel> channel, size_t size, const T& value = T{})
        : UCXMessage(channel), data_(size, value), registered_(false) {}
    
    ~UCXVector() = default;
    
    // Container interface
    size_t size() const override { return data_.size(); }
    size_t capacity() const override { return data_.capacity(); }
    bool empty() const { return data_.empty(); }
    
    void resize(size_t new_size) override { 
        data_.resize(new_size);
        invalidate_registration();
    }
    
    void reserve(size_t new_capacity) override { 
        data_.reserve(new_capacity);
        invalidate_registration();
    }
    
    void clear() { 
        data_.clear();
        invalidate_registration();
    }
    
    // Element access
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    
    T& at(size_t index) { return data_.at(index); }
    const T& at(size_t index) const { return data_.at(index); }
    
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    
    // Iterators
    iterator begin() { return data_.begin(); }
    const_iterator begin() const { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator end() const { return data_.end(); }
    
    // Modifiers
    void push_back(const T& value) { 
        data_.push_back(value);
        invalidate_registration();
    }
    
    void pop_back() { 
        data_.pop_back();
        invalidate_registration();
    }
    
    template<typename... Args>
    void emplace_back(Args&&... args) {
        data_.emplace_back(std::forward<Args>(args)...);
        invalidate_registration();
    }
    
    // UCX-specific operations
    
    /**
     * @brief Send vector to specified peer
     */
    bool send(const std::string& peer = "") override {
        ensure_registered();
        
        if (delivery_mode_ == DeliveryMode::ZERO_COPY && memory_region_) {
            return channel_->send_memory_region(*memory_region_);
        } else {
            return channel_->send(data(), size() * sizeof(T), get_tag()) > 0;
        }
    }
    
    /**
     * @brief Receive vector from specified peer
     */
    bool receive(const std::string& peer = "") override {
        ensure_registered();
        
        if (delivery_mode_ == DeliveryMode::ZERO_COPY && memory_region_) {
            return channel_->receive_memory_region(*memory_region_);
        } else {
            size_t received = channel_->receive(data(), capacity() * sizeof(T));
            if (received > 0) {
                resize(received / sizeof(T));
                return true;
            }
            return false;
        }
    }
    
    /**
     * @brief Broadcast vector to multiple peers
     */
    bool broadcast(const std::vector<std::string>& peers) override {
        ensure_registered();
        
        bool success = true;
        for (const auto& peer : peers) {
            if (!channel_->connect(peer)) {
                success = false;
                continue;
            }
            
            if (delivery_mode_ == DeliveryMode::ZERO_COPY && memory_region_) {
                success &= channel_->send_memory_region(*memory_region_);
            } else {
                success &= (channel_->send(data(), size() * sizeof(T), get_tag()) > 0);
            }
        }
        return success;
    }
    
    /**
     * @brief Scatter vector elements across peers
     */
    bool scatter(const std::vector<std::string>& peers, size_t chunk_size = 0) {
        if (peers.empty()) return false;
        
        size_t elements_per_peer = chunk_size > 0 ? chunk_size : size() / peers.size();
        size_t offset = 0;
        
        for (size_t i = 0; i < peers.size() && offset < size(); ++i) {
            if (!channel_->connect(peers[i])) {
                continue;
            }
            
            size_t send_size = std::min(elements_per_peer, size() - offset);
            const T* send_data = data() + offset;
            
            bool success = channel_->send(send_data, send_size * sizeof(T), get_tag() + i) > 0;
            if (!success) return false;
            
            offset += send_size;
        }
        
        return true;
    }
    
    /**
     * @brief Gather vector elements from peers
     */
    bool gather(const std::vector<std::string>& peers) {
        clear();
        
        for (size_t i = 0; i < peers.size(); ++i) {
            if (!channel_->connect(peers[i])) {
                continue;
            }
            
            // Receive size first
            size_t incoming_size;
            if (channel_->receive(&incoming_size, sizeof(size_t)) == 0) {
                continue;
            }
            
            // Resize to accommodate new data
            size_t old_size = size();
            resize(old_size + incoming_size);
            
            // Receive data
            size_t received = channel_->receive(data() + old_size, incoming_size * sizeof(T));
            if (received != incoming_size * sizeof(T)) {
                resize(old_size); // Rollback on failure
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * @brief All-to-all exchange with peers
     */
    bool all_to_all(const std::vector<std::string>& peers) {
        // Send to all peers
        bool send_success = broadcast(peers);
        
        // Receive from all peers
        bool recv_success = gather(peers);
        
        return send_success && recv_success;
    }
    
    /**
     * @brief Register memory for zero-copy operations
     */
    void ensure_registered() {
        if (!registered_ && !data_.empty()) {
            memory_region_ = channel_->register_memory(data(), size() * sizeof(T), MemoryType::HOST);
            registered_ = (memory_region_ != nullptr);
        }
    }
    
    /**
     * @brief Get memory region for advanced operations
     */
    std::shared_ptr<UCXMemoryRegion> memory_region() const { return memory_region_; }
    
    /**
     * @brief Check if memory is registered for zero-copy
     */
    bool is_registered() const { return registered_; }
    
private:
    std::vector<T> data_;
    std::shared_ptr<UCXMemoryRegion> memory_region_;
    bool registered_;
    
    void invalidate_registration() {
        registered_ = false;
        memory_region_.reset();
    }
};

/**
 * @brief UCX GPU-aware vector for CUDA/ROCm memory
 */
template<typename T>
class UCXGPUVector : public UCXMessage {
public:
    UCXGPUVector(std::shared_ptr<UCXChannel> channel, gpu::GPUContext& gpu_context,
                 MemoryType mem_type = MemoryType::CUDA)
        : UCXMessage(channel), gpu_context_(gpu_context), memory_type_(mem_type), registered_(false) {}
    
    ~UCXGPUVector() = default;
    
    // Container interface
    size_t size() const override { return size_; }
    size_t capacity() const override { return capacity_; }
    bool empty() const { return size_ == 0; }
    
    void resize(size_t new_size) override {
        if (new_size > capacity_) {
            reserve(new_size * 2); // Exponential growth
        }
        size_ = new_size;
    }
    
    void reserve(size_t new_capacity) override {
        if (new_capacity > capacity_) {
            allocate_gpu_memory(new_capacity);
            capacity_ = new_capacity;
            invalidate_registration();
        }
    }
    
    void clear() { size_ = 0; }
    
    // GPU memory access
    T* gpu_data() { return gpu_data_; }
    const T* gpu_data() const { return gpu_data_; }
    
    // Host-GPU synchronization
    bool copy_to_host(std::vector<T>& host_vector) const {
        host_vector.resize(size_);
        return gpu_context_.copy_from_gpu(host_vector.data(), gpu_data_, size_ * sizeof(T));
    }
    
    bool copy_from_host(const std::vector<T>& host_vector) {
        resize(host_vector.size());
        return gpu_context_.copy_to_gpu(gpu_data_, host_vector.data(), size_ * sizeof(T));
    }
    
    // UCX operations with GPU memory
    bool send(const std::string& peer = "") override {
        ensure_registered();
        
        if (memory_region_) {
            return channel_->send_memory_region(*memory_region_);
        }
        return false;
    }
    
    bool receive(const std::string& peer = "") override {
        ensure_registered();
        
        if (memory_region_) {
            return channel_->receive_memory_region(*memory_region_);
        }
        return false;
    }
    
    bool broadcast(const std::vector<std::string>& peers) override {
        ensure_registered();
        
        if (!memory_region_) return false;
        
        bool success = true;
        for (const auto& peer : peers) {
            if (channel_->connect(peer)) {
                success &= channel_->send_memory_region(*memory_region_);
            } else {
                success = false;
            }
        }
        return success;
    }
    
    /**
     * @brief Direct GPU-to-GPU transfer via RMA
     */
    bool gpu_to_gpu_transfer(const std::string& peer, uint64_t remote_addr, 
                            const std::vector<uint8_t>& remote_key) {
        ensure_registered();
        
        if (!memory_region_) return false;
        
        return channel_->rma_put(gpu_data_, size_ * sizeof(T), peer, remote_addr, remote_key);
    }
    
    void ensure_registered() {
        if (!registered_ && gpu_data_ && size_ > 0) {
            memory_region_ = channel_->register_memory(gpu_data_, size_ * sizeof(T), memory_type_);
            registered_ = (memory_region_ != nullptr);
        }
    }
    
    std::shared_ptr<UCXMemoryRegion> memory_region() const { return memory_region_; }
    bool is_registered() const { return registered_; }
    
private:
    gpu::GPUContext& gpu_context_;
    MemoryType memory_type_;
    T* gpu_data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
    std::shared_ptr<UCXMemoryRegion> memory_region_;
    bool registered_;
    
    void allocate_gpu_memory(size_t elements) {
        if (gpu_data_) {
            gpu_context_.free_gpu_memory(gpu_data_);
        }
        
        gpu_data_ = static_cast<T*>(gpu_context_.allocate_gpu_memory(elements * sizeof(T)));
        if (!gpu_data_) {
            throw std::runtime_error("Failed to allocate GPU memory");
        }
    }
    
    void invalidate_registration() {
        registered_ = false;
        memory_region_.reset();
    }
};

/**
 * @brief High-level collective operations using UCX
 */
class UCXCollectives {
public:
    UCXCollectives(std::shared_ptr<UCXChannel> channel) : channel_(channel) {}
    
    /**
     * @brief Allreduce operation for numeric types
     */
    template<typename T>
    bool allreduce(UCXVector<T>& data, const std::vector<std::string>& peers,
                   std::function<T(T, T)> reduction_op = std::plus<T>{}) {
        
        // Gather all data
        std::vector<UCXVector<T>> peer_data(peers.size());
        for (size_t i = 0; i < peers.size(); ++i) {
            peer_data[i] = UCXVector<T>(channel_);
            peer_data[i].resize(data.size());
            
            if (!channel_->connect(peers[i])) continue;
            peer_data[i].receive(peers[i]);
        }
        
        // Perform reduction
        for (size_t i = 0; i < data.size(); ++i) {
            T result = data[i];
            for (const auto& peer_vec : peer_data) {
                if (i < peer_vec.size()) {
                    result = reduction_op(result, peer_vec[i]);
                }
            }
            data[i] = result;
        }
        
        // Broadcast result back to all peers
        return data.broadcast(peers);
    }
    
    /**
     * @brief Barrier synchronization
     */
    bool barrier(const std::vector<std::string>& peers) {
        uint32_t sync_value = 1;
        
        // Send sync signal to all peers
        for (const auto& peer : peers) {
            if (channel_->connect(peer)) {
                channel_->send(&sync_value, sizeof(sync_value), 0xBARRIER);
            }
        }
        
        // Wait for sync signals from all peers
        for (const auto& peer : peers) {
            uint32_t received_value;
            size_t received = channel_->receive(&received_value, sizeof(received_value));
            if (received != sizeof(received_value) || received_value != sync_value) {
                return false;
            }
        }
        
        return true;
    }
    
private:
    std::shared_ptr<UCXChannel> channel_;
};

// Convenience type aliases
using UCXFloatVector = UCXVector<float>;
using UCXDoubleVector = UCXVector<double>;
using UCXIntVector = UCXVector<int32_t>;
using UCXLongVector = UCXVector<int64_t>;

using UCXGPUFloatVector = UCXGPUVector<float>;
using UCXGPUDoubleVector = UCXGPUVector<double>;
using UCXGPUIntVector = UCXGPUVector<int32_t>;

} // namespace ucx
} // namespace psyne

#endif // PSYNE_UCX_SUPPORT