#pragma once

#include <cstddef>
#include <memory>

namespace psyne {

/**
 * @brief Base allocator interface that messages use
 * 
 * This breaks the circular dependency:
 * - Messages take Allocator& (not Channel&)
 * - Channels create Allocators (using their Substrate)
 * - Clean separation of concerns
 */
class AllocatorBase {
public:
    virtual ~AllocatorBase() = default;
    
    /**
     * @brief Allocate memory for a message
     */
    virtual void* allocate(size_t size_bytes, size_t alignment) = 0;
    
    /**
     * @brief Deallocate memory (optional - many allocators don't need this)
     */
    virtual void deallocate(void* ptr) {}
    
    /**
     * @brief Get allocator info
     */
    virtual const char* name() const = 0;
    virtual bool is_zero_copy() const = 0;
    virtual size_t total_allocated() const { return 0; }
};

/**
 * @brief Slab allocator that allocates from a pre-allocated memory region
 * 
 * This is what channels create using their substrate
 */
template<typename T>
class SlabAllocator : public AllocatorBase {
public:
    SlabAllocator(T* slab_memory, size_t slab_size, const char* name = "Slab")
        : slab_memory_(slab_memory), slab_size_(slab_size), 
          allocated_bytes_(0), allocator_name_(name) {}
    
    void* allocate(size_t size_bytes, size_t alignment) override {
        // Align current position
        size_t aligned_pos = (allocated_bytes_ + alignment - 1) & ~(alignment - 1);
        
        if (aligned_pos + size_bytes > slab_size_) {
            return nullptr; // Out of space
        }
        
        void* ptr = static_cast<char*>(slab_memory_) + aligned_pos;
        allocated_bytes_ = aligned_pos + size_bytes;
        
        return ptr;
    }
    
    const char* name() const override { return allocator_name_; }
    bool is_zero_copy() const override { return true; }
    size_t total_allocated() const override { return allocated_bytes_; }
    
    size_t available() const { return slab_size_ - allocated_bytes_; }
    double utilization() const { return double(allocated_bytes_) / double(slab_size_); }

private:
    T* slab_memory_;
    size_t slab_size_;
    size_t allocated_bytes_;
    const char* allocator_name_;
};

/**
 * @brief Ring allocator for channel patterns
 * 
 * Allocates messages in a ring buffer for producer-consumer patterns
 */
template<typename T>
class RingAllocator : public AllocatorBase {
public:
    RingAllocator(T* slab_memory, size_t slab_size, size_t ring_size)
        : slab_memory_(slab_memory), slab_size_(slab_size), 
          ring_size_(ring_size), current_pos_(0) {
        
        message_size_ = sizeof(T);
        max_messages_ = slab_size / message_size_;
        
        if (ring_size > max_messages_) {
            ring_size_ = max_messages_;
        }
    }
    
    void* allocate(size_t size_bytes, size_t alignment) override {
        if (size_bytes != message_size_) {
            return nullptr; // Ring allocator only supports fixed-size messages
        }
        
        T* slot = &slab_memory_[current_pos_ % max_messages_];
        current_pos_++;
        
        return slot;
    }
    
    const char* name() const override { return "Ring"; }
    bool is_zero_copy() const override { return true; }
    
    size_t get_ring_position() const { return current_pos_; }

private:
    T* slab_memory_;
    size_t slab_size_;
    size_t ring_size_;
    size_t message_size_;
    size_t max_messages_;
    size_t current_pos_;
};

/**
 * @brief Pool allocator for dynamic message sizes
 */
class PoolAllocator : public AllocatorBase {
public:
    PoolAllocator(void* memory, size_t total_size) 
        : memory_(memory), total_size_(total_size), allocated_(0) {}
    
    void* allocate(size_t size_bytes, size_t alignment) override {
        // Simple bump allocator
        size_t aligned_pos = (allocated_ + alignment - 1) & ~(alignment - 1);
        
        if (aligned_pos + size_bytes > total_size_) {
            return nullptr;
        }
        
        void* ptr = static_cast<char*>(memory_) + aligned_pos;
        allocated_ = aligned_pos + size_bytes;
        
        return ptr;
    }
    
    const char* name() const override { return "Pool"; }
    bool is_zero_copy() const override { return true; }
    size_t total_allocated() const override { return allocated_; }

private:
    void* memory_;
    size_t total_size_;
    size_t allocated_;
};

} // namespace psyne