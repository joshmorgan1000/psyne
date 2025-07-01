#pragma once

/**
 * @file simple_patterns.hpp
 * @brief Simplified pattern implementations without boost dependencies
 * 
 * These patterns provide basic functionality for CI testing without
 * requiring boost::asio for coroutine support.
 */

#include "behaviors.hpp"
#include <atomic>
#include <vector>
#include <mutex>

namespace psyne::simple_patterns {

/**
 * @brief Simplified SPSC pattern without boost dependencies
 */
class SimpleSPSC : public psyne::behaviors::PatternBehavior {
private:
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
public:
    explicit SimpleSPSC(size_t ring_size = 1024) 
        : ring_size_(ring_size), ring_mask_(ring_size - 1) {
        if ((ring_size & (ring_size - 1)) != 0) {
            throw std::invalid_argument("Ring size must be power of 2");
        }
        ring_ = new std::atomic<void*>[ring_size];
        for (size_t i = 0; i < ring_size; ++i) {
            ring_[i].store(nullptr, std::memory_order_relaxed);
        }
    }
    
    ~SimpleSPSC() {
        delete[] ring_;
    }
    
    void* coordinate_allocation(void* slab_memory, size_t message_size) override {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next_head = head + 1;
        
        size_t cached_tail = tail_.load(std::memory_order_acquire);
        
        if (next_head - cached_tail > ring_size_) {
            return nullptr; // Ring full
        }
        
        size_t slot = head % (ring_size_ * 10); // Arbitrary max messages
        void* ptr = static_cast<char*>(slab_memory) + (slot * message_size);
        
        ring_[head & ring_mask_].store(ptr, std::memory_order_relaxed);
        head_.store(next_head, std::memory_order_release);
        
        return ptr;
    }
    
    void* coordinate_receive() override {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t cached_head = head_.load(std::memory_order_acquire);
        
        if (tail >= cached_head) {
            return nullptr; // Ring empty
        }
        
        void* msg = ring_[tail & ring_mask_].load(std::memory_order_acquire);
        if (!msg) {
            return nullptr;
        }
        
        ring_[tail & ring_mask_].store(nullptr, std::memory_order_relaxed);
        tail_.store(tail + 1, std::memory_order_release);
        
        return msg;
    }
    
    void producer_sync() override {}
    void consumer_sync() override {}
    
    const char* pattern_name() const override { return "SimpleSPSC"; }
    bool needs_locks() const override { return false; }
    size_t max_producers() const override { return 1; }
    size_t max_consumers() const override { return 1; }

private:
    size_t ring_size_;
    size_t ring_mask_;
    std::atomic<void*>* ring_;
    
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_{0};
};

/**
 * @brief Simplified InProcess substrate without boost dependencies
 */
class SimpleInProcess : public psyne::behaviors::SubstrateBehavior {
public:
    void* allocate_memory_slab(size_t size_bytes) override {
        return std::aligned_alloc(64, size_bytes);
    }
    
    void deallocate_memory_slab(void* memory) override {
        std::free(memory);
    }
    
    void transport_send(void* data, size_t size) override {
        // InProcess doesn't need network transport
    }
    
    void transport_receive(void* buffer, size_t buffer_size) override {
        // InProcess doesn't need network transport
    }
    
    const char* substrate_name() const override { return "SimpleInProcess"; }
    bool is_zero_copy() const override { return true; }
    bool is_cross_process() const override { return false; }
};

} // namespace psyne::simple_patterns