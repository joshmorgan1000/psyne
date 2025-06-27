#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace psyne {

class SlabAllocator {
public:
    static constexpr size_t kAlignment = 8;
    
    struct SlabHeader {
        uint32_t len;          // Total bytes after this u32
        uint32_t reserved;     // Future: xxhash32 for tcp
        
        void* data() { return reinterpret_cast<uint8_t*>(this) + sizeof(SlabHeader); }
        const void* data() const { return reinterpret_cast<const uint8_t*>(this) + sizeof(SlabHeader); }
    };
    
    explicit SlabAllocator(size_t slab_size)
        : slab_size_(align_up(slab_size, kAlignment))
        , memory_(static_cast<uint8_t*>(std::aligned_alloc(kAlignment, slab_size_)))
        , head_(0)
        , tail_(0) {
        if (!memory_) {
            throw std::bad_alloc();
        }
    }
    
    ~SlabAllocator() {
        std::free(memory_);
    }
    
    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;
    
    SlabAllocator(SlabAllocator&& other) noexcept
        : slab_size_(other.slab_size_)
        , memory_(std::exchange(other.memory_, nullptr))
        , head_(other.head_.load())
        , tail_(other.tail_.load()) {}
    
    void* allocate(size_t size) {
        size_t total_size = align_up(sizeof(SlabHeader) + size, kAlignment);
        
        size_t current_head = head_.load(std::memory_order_relaxed);
        size_t new_head;
        
        do {
            new_head = current_head + total_size;
            if (new_head > slab_size_) {
                return nullptr;  // Out of memory
            }
        } while (!head_.compare_exchange_weak(
            current_head, new_head,
            std::memory_order_release,
            std::memory_order_relaxed));
        
        auto* header = reinterpret_cast<SlabHeader*>(memory_ + current_head);
        header->len = static_cast<uint32_t>(size);
        header->reserved = 0;
        
        return header->data();
    }
    
    void deallocate(void* ptr) {
        // For ring buffer semantics, deallocation happens in order
        // This is a simplified version - real implementation would validate ptr
    }
    
    size_t available() const {
        return slab_size_ - head_.load(std::memory_order_acquire);
    }
    
    size_t capacity() const { return slab_size_; }
    
    void* base() { return memory_; }
    const void* base() const { return memory_; }
    
private:
    static constexpr size_t align_up(size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }
    
    size_t slab_size_;
    uint8_t* memory_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};

}  // namespace psyne