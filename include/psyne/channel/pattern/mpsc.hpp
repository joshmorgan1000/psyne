/**
 * @file mpsc.hpp
 * @brief Multi-Producer, Single-Consumer pattern implementation
 * 
 * MPSC = Multiple producers can send, only 1 consumer receives
 * Requires synchronization on the producer side for slot allocation
 */

#pragma once

#include "../../core/behaviors.hpp"
#include <atomic>
#include <mutex>
#include <cstddef>

namespace psyne::patterns {

/**
 * @brief Multi-Producer, Single-Consumer pattern
 * 
 * Multiple producer threads can safely send messages concurrently.
 * Only one consumer thread should receive messages.
 * 
 * Synchronization:
 * - Producer allocation: Atomic counter + mutex for slot assignment
 * - Consumer receive: Lock-free (single consumer)
 */
class MPSC : public psyne::behaviors::PatternBehavior {
public:
    explicit MPSC(size_t max_messages = 1024 * 1024) : max_messages_(max_messages) {}
    
    /**
     * @brief Coordinate allocation for multiple producers (thread-safe)
     */
    void* coordinate_allocation(void* slab_memory, size_t message_size) override {
        // Store slab info on first allocation
        if (!slab_memory_) {
            std::lock_guard<std::mutex> lock(slab_mutex_);
            if (!slab_memory_) {
                slab_memory_ = slab_memory;
                message_size_ = message_size;
            }
        }
        
        // Thread-safe slot allocation for producers
        size_t slot = write_pos_.fetch_add(1, std::memory_order_acq_rel) % max_messages_;
        
        return static_cast<char*>(slab_memory) + (slot * message_size);
    }
    
    /**
     * @brief Coordinate receive for single consumer (lock-free)
     */
    void* coordinate_receive() override {
        size_t current_read = read_pos_.load(std::memory_order_relaxed);
        size_t current_write = write_pos_.load(std::memory_order_acquire);
        
        if (current_read >= current_write) {
            return nullptr; // No messages available
        }
        
        size_t slot = current_read % max_messages_;
        read_pos_.fetch_add(1, std::memory_order_release);
        
        return static_cast<char*>(slab_memory_) + (slot * message_size_);
    }
    
    /**
     * @brief Producer synchronization (atomic operations)
     */
    void producer_sync() override {
        // Producers coordinate via atomic write_pos_
    }
    
    /**
     * @brief Consumer synchronization (none needed - single consumer)
     */
    void consumer_sync() override {
        // Single consumer, no sync needed
    }
    
    /**
     * @brief Pattern capabilities
     */
    const char* pattern_name() const override { return "MPSC"; }
    bool needs_locks() const override { return true; } // For producer coordination
    size_t max_producers() const override { return SIZE_MAX; } // Unlimited
    size_t max_consumers() const override { return 1; } // Exactly 1
    
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
    // Producer coordination (atomic for multiple producers)
    std::atomic<size_t> write_pos_{0};
    
    // Consumer state (single consumer, relaxed ordering)
    std::atomic<size_t> read_pos_{0};
    
    // Ring buffer configuration
    size_t max_messages_;
    void* slab_memory_ = nullptr;
    size_t message_size_ = 0;
    
    // Slab initialization synchronization
    std::mutex slab_mutex_;
};

} // namespace psyne::patterns