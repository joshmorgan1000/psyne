/**
 * @file mpmc.hpp
 * @brief Multi-Producer, Multi-Consumer pattern implementation
 *
 * MPMC = Multiple producers send, multiple consumers receive
 * Requires synchronization on both producer and consumer sides
 */

#pragma once

#include "../../core/behaviors.hpp"
#include <atomic>
#include <cstddef>
#include <mutex>

namespace psyne::patterns {

/**
 * @brief Multi-Producer, Multi-Consumer pattern
 *
 * Multiple producer threads can send messages concurrently.
 * Multiple consumer threads can receive messages concurrently.
 *
 * Synchronization:
 * - Producer allocation: Atomic counter for slot assignment
 * - Consumer receive: Atomic counter for message distribution
 * - Both sides need coordination to avoid conflicts
 */
class MPMC : public psyne::behaviors::PatternBehavior {
public:
    explicit MPMC(size_t max_messages = 1024 * 1024)
        : max_messages_(max_messages) {}

    /**
     * @brief Coordinate allocation for multiple producers (thread-safe)
     */
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        // Store slab info on first allocation
        if (!slab_memory_) {
            std::lock_guard<std::mutex> lock(slab_mutex_);
            if (!slab_memory_) {
                slab_memory_ = slab_memory;
                message_size_ = message_size;
            }
        }

        // Atomic slot reservation with overflow check
        size_t slot_index;
        size_t current_write, current_read;

        do {
            current_write = write_pos_.load(std::memory_order_acquire);
            current_read = read_pos_.load(std::memory_order_acquire);

            // Check if ring buffer is full
            if (current_write - current_read >= max_messages_) {
                allocation_failures_.fetch_add(1, std::memory_order_relaxed);
                return nullptr; // Ring buffer full
            }

            slot_index = current_write;

            // Try to reserve the slot atomically
        } while (!write_pos_.compare_exchange_weak(
            current_write, current_write + 1, std::memory_order_acq_rel,
            std::memory_order_acquire));

        // Compute actual slot position
        size_t slot = slot_index % max_messages_;

        // Bounds check
        if (slot * message_size >= max_messages_ * message_size) {
            // This should never happen, but check anyway
            allocation_failures_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }

        return static_cast<char *>(slab_memory) + (slot * message_size);
    }

    /**
     * @brief Coordinate receive for multiple consumers (thread-safe)
     */
    void *coordinate_receive() override {
        // Thread-safe message consumption
        size_t current_read = read_pos_.fetch_add(1, std::memory_order_acq_rel);
        size_t current_write = write_pos_.load(std::memory_order_acquire);

        if (current_read >= current_write) {
            // No message available, rollback the read position
            read_pos_.fetch_sub(1, std::memory_order_acq_rel);
            return nullptr;
        }

        size_t slot = current_read % max_messages_;
        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }

    /**
     * @brief Producer synchronization (atomic operations)
     */
    void producer_sync() override {
        // Producers coordinate via atomic write_pos_
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
    const char *pattern_name() const override {
        return "MPMC";
    }
    bool needs_locks() const override {
        return true;
    } // For both producer and consumer coordination
    size_t max_producers() const override {
        return SIZE_MAX;
    } // Unlimited
    size_t max_consumers() const override {
        return SIZE_MAX;
    } // Unlimited

    /**
     * @brief Pattern state
     */
    size_t size() const {
        return write_pos_.load(std::memory_order_acquire) -
               read_pos_.load(std::memory_order_acquire);
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
    size_t get_write_pos() const {
        return write_pos_.load();
    }
    size_t get_read_pos() const {
        return read_pos_.load();
    }
    size_t get_max_messages() const {
        return max_messages_;
    }

    /**
     * @brief Get contention statistics
     */
    size_t get_allocation_failures() const {
        return allocation_failures_.load();
    }
    size_t get_receive_failures() const {
        return receive_failures_.load();
    }

private:
    // Producer coordination (atomic for multiple producers)
    std::atomic<size_t> write_pos_{0};

    // Consumer coordination (atomic for multiple consumers)
    std::atomic<size_t> read_pos_{0};

    // Ring buffer configuration
    size_t max_messages_;
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;

    // Slab initialization synchronization
    std::mutex slab_mutex_;

    // Contention tracking
    std::atomic<size_t> allocation_failures_{0};
    std::atomic<size_t> receive_failures_{0};
};

} // namespace psyne::patterns