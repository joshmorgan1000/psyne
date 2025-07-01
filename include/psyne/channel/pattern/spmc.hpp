/**
 * @file spmc.hpp
 * @brief Single-Producer, Multi-Consumer pattern implementation
 * 
 * SPMC = Single producer sends, multiple consumers can receive
 * Requires synchronization on the consumer side for message distribution
 */

#pragma once

#include "../../core/behaviors.hpp"
#include <atomic>
#include <mutex>
#include <cstddef>

namespace psyne::patterns {

/**
 * @brief Single-Producer, Multi-Consumer pattern
 * 
 * Only one producer thread should send messages.
 * Multiple consumer threads can safely receive messages concurrently.
 * 
 * Synchronization:
 * - Producer allocation: Lock-free (single producer)
 * - Consumer receive: Atomic counter for message distribution
 */
class SPMC : public psyne::behaviors::PatternBehavior {
public:
    explicit SPMC(size_t max_messages = 1024 * 1024) : max_messages_(max_messages) {}
    
    /**
     * @brief Coordinate allocation for single producer (lock-free)
     */
    void* coordinate_allocation(void* slab_memory, size_t message_size) override {
        // Store slab info on first allocation
        if (!slab_memory_) {
            slab_memory_ = slab_memory;
            message_size_ = message_size;
        }
        
        // Single producer, no contention
        size_t slot = write_pos_.load(std::memory_order_relaxed) % max_messages_;
        write_pos_.fetch_add(1, std::memory_order_release);
        
        return static_cast<char*>(slab_memory) + (slot * message_size);
    }
    
    /**
     * @brief Coordinate receive for multiple consumers (thread-safe)
     */
    void* coordinate_receive() override {
        // Thread-safe message consumption
        size_t current_read = read_pos_.fetch_add(1, std::memory_order_acq_rel);
        size_t current_write = write_pos_.load(std::memory_order_acquire);
        
        if (current_read >= current_write) {
            // No message available, rollback the read position
            read_pos_.fetch_sub(1, std::memory_order_acq_rel);
            return nullptr;
        }
        
        size_t slot = current_read % max_messages_;
        return static_cast<char*>(slab_memory_) + (slot * message_size_);
    }
    
    /**
     * @brief Producer synchronization (none needed - single producer)
     */
    void producer_sync() override {
        // Single producer, no sync needed
    }
    
    /**
     * @brief Consumer synchronization (atomic operations)
     */
    void consumer_sync() override {
        // Consumers coordinate via atomic read_pos_
    }
    
    /**
     * @brief Pattern capabilities
     */
    const char* pattern_name() const override { return "SPMC"; }
    bool needs_locks() const override { return true; } // For consumer coordination
    size_t max_producers() const override { return 1; } // Exactly 1
    size_t max_consumers() const override { return SIZE_MAX; } // Unlimited
    
    /**
     * @brief Pattern state
     */
    size_t size() const {
        return write_pos_.load(std::memory_order_acquire) - read_pos_.load(std::memory_order_relaxed);
    }
    
    bool empty() const {
        return size() == 0;
    }
    
    bool full() const {
        return size() >= max_messages_;
    }
    
    /**
     * @brief Get statistics
     */
    size_t get_write_pos() const { return write_pos_.load(); }
    size_t get_read_pos() const { return read_pos_.load(); }
    size_t get_max_messages() const { return max_messages_; }

private:
    // Producer state (single producer, relaxed ordering)
    std::atomic<size_t> write_pos_{0};
    
    // Consumer coordination (atomic for multiple consumers)
    std::atomic<size_t> read_pos_{0};
    
    // Ring buffer configuration
    size_t max_messages_;
    void* slab_memory_ = nullptr;
    size_t message_size_ = 0;
};

} // namespace psyne::patterns