#include "ring_buffer.hpp"
#include <cstdlib>
#include <stdexcept>
#include <cstdio>

// Platform-specific aligned allocation (duplicated for now - should be moved to common header)
static void *aligned_alloc_portable(size_t alignment, size_t size) {
    // Ensure alignment is a power of 2 and at least sizeof(void*)
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }
    
    // Ensure size is aligned properly for std::aligned_alloc
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    
#if defined(_WIN32)
    return _aligned_malloc(aligned_size, alignment);
#elif defined(__APPLE__) ||                                                    \
    (defined(__GLIBC__) &&                                                     \
     (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 16)))
    return std::aligned_alloc(alignment, aligned_size);
#else
    // Linux fallback - posix_memalign is more robust
    void *ptr = nullptr;
    // posix_memalign requires alignment to be power of 2 and multiple of sizeof(void*)
    if ((alignment & (alignment - 1)) != 0) {
        // Not a power of 2, round up to next power of 2
        size_t power = sizeof(void*);
        while (power < alignment) power <<= 1;
        alignment = power;
    }
    
    int result = posix_memalign(&ptr, alignment, aligned_size);
    if (result != 0) {
        fprintf(stderr, "posix_memalign failed in ring_buffer_zero_copy: alignment=%zu, size=%zu, error=%d\n", 
                alignment, aligned_size, result);
        return nullptr;
    }
    return ptr;
#endif
}

static void aligned_free_portable(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

namespace psyne {

// Utility function to get next power of 2
size_t SPSCRingBuffer::next_power_of_2(size_t n) {
    if (n == 0) return 1;
    if ((n & (n - 1)) == 0) return n; // Already power of 2
    
    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

// SPSC Ring Buffer Implementation
SPSCRingBuffer::SPSCRingBuffer(size_t size) 
    : buffer_size_(next_power_of_2(size))
    , mask_(buffer_size_ - 1)
    , head_(0)
    , tail_(0) {
    
    // Allocate aligned memory for ring buffer slab
    slab_ = static_cast<uint8_t*>(aligned_alloc_portable(64, buffer_size_));
    if (!slab_) {
        throw std::runtime_error("Failed to allocate ring buffer memory");
    }
}

SPSCRingBuffer::~SPSCRingBuffer() {
    if (slab_) {
        aligned_free_portable(slab_);
    }
}

uint32_t SPSCRingBuffer::reserve_slot(size_t size) noexcept {
    // SPSC: No atomics needed - producer owns head
    const uint64_t current_head = head_;
    const uint64_t current_tail = tail_;
    
    // Check if we have space (head-hits-tail backpressure)
    if ((current_head + size - current_tail) >= buffer_size_) {
        return BUFFER_FULL;
    }
    
    // Return offset (will wrap around with mask)
    return static_cast<uint32_t>(current_head & mask_);
}

void SPSCRingBuffer::advance_write_pointer(size_t new_write_pos) noexcept {
    // SPSC: Just advance head - no atomic needed
    head_ = new_write_pos;
}

void SPSCRingBuffer::advance_read_pointer(size_t size) noexcept {
    // SPSC: Consumer advances tail - no atomic needed
    tail_ += size;
}

bool SPSCRingBuffer::has_space(size_t size) const noexcept {
    return (head_ + size - tail_) < buffer_size_;
}

// MPSC Ring Buffer Implementation
size_t MPSCRingBuffer::next_power_of_2(size_t n) {
    return SPSCRingBuffer::next_power_of_2(n);
}

MPSCRingBuffer::MPSCRingBuffer(size_t size)
    : buffer_size_(next_power_of_2(size))
    , mask_(buffer_size_ - 1)
    , head_(0)
    , tail_(0) {
    
    slab_ = static_cast<uint8_t*>(std::aligned_alloc(64, buffer_size_));
    if (!slab_) {
        throw std::runtime_error("Failed to allocate ring buffer memory");
    }
}

MPSCRingBuffer::~MPSCRingBuffer() {
    if (slab_) {
        std::free(slab_);
    }
}

uint32_t MPSCRingBuffer::reserve_slot(size_t size) noexcept {
    // MPSC: Multiple producers need atomic CAS
    uint64_t current_head = head_.load(std::memory_order_acquire);
    uint64_t current_tail = tail_;
    
    do {
        // Check if we have space
        if ((current_head + size - current_tail) >= buffer_size_) {
            return BUFFER_FULL;
        }
        
        // Try to atomically advance head
    } while (!head_.compare_exchange_weak(current_head, current_head + size, 
                                         std::memory_order_release, 
                                         std::memory_order_acquire));
    
    return static_cast<uint32_t>(current_head & mask_);
}

void MPSCRingBuffer::advance_write_pointer(size_t new_write_pos) noexcept {
    // For MPSC, write pointer is already advanced in reserve_slot
    // This is a no-op since we advance atomically during reservation
}

void MPSCRingBuffer::advance_read_pointer(size_t size) noexcept {
    // MPSC: Single consumer, no atomic needed for tail
    tail_ += size;
}

bool MPSCRingBuffer::has_space(size_t size) const noexcept {
    uint64_t current_head = head_.load(std::memory_order_acquire);
    return (current_head + size - tail_) < buffer_size_;
}

// SPMC Ring Buffer Implementation
size_t SPMCRingBuffer::next_power_of_2(size_t n) {
    return SPSCRingBuffer::next_power_of_2(n);
}

SPMCRingBuffer::SPMCRingBuffer(size_t size)
    : buffer_size_(next_power_of_2(size))
    , mask_(buffer_size_ - 1)
    , head_(0)
    , tail_(0) {
    
    slab_ = static_cast<uint8_t*>(std::aligned_alloc(64, buffer_size_));
    if (!slab_) {
        throw std::runtime_error("Failed to allocate ring buffer memory");
    }
}

SPMCRingBuffer::~SPMCRingBuffer() {
    if (slab_) {
        std::free(slab_);
    }
}

uint32_t SPMCRingBuffer::reserve_slot(size_t size) {
    // SPMC: Single producer, no atomic needed for head
    const uint64_t current_head = head_;
    const uint64_t current_tail = tail_.load(std::memory_order_acquire);
    
    if ((current_head + size - current_tail) >= buffer_size_) {
        return BUFFER_FULL;
    }
    
    return static_cast<uint32_t>(current_head & mask_);
}

void SPMCRingBuffer::advance_write_pointer(size_t new_write_pos) {
    // SPMC: Single producer, no atomic needed
    head_ = new_write_pos;
}

void SPMCRingBuffer::advance_read_pointer(size_t size) {
    // SPMC: Multiple consumers need atomic CAS
    uint64_t current_tail = tail_.load(std::memory_order_acquire);
    while (!tail_.compare_exchange_weak(current_tail, current_tail + size,
                                       std::memory_order_release,
                                       std::memory_order_acquire)) {
        // Retry CAS
    }
}

bool SPMCRingBuffer::has_space(size_t size) const {
    uint64_t current_tail = tail_.load(std::memory_order_acquire);
    return (head_ + size - current_tail) < buffer_size_;
}

// MPMC Ring Buffer Implementation
size_t MPMCRingBuffer::next_power_of_2(size_t n) {
    return SPSCRingBuffer::next_power_of_2(n);
}

MPMCRingBuffer::MPMCRingBuffer(size_t size)
    : buffer_size_(next_power_of_2(size))
    , mask_(buffer_size_ - 1)
    , head_(0)
    , tail_(0) {
    
    slab_ = static_cast<uint8_t*>(std::aligned_alloc(64, buffer_size_));
    if (!slab_) {
        throw std::runtime_error("Failed to allocate ring buffer memory");
    }
}

MPMCRingBuffer::~MPMCRingBuffer() {
    if (slab_) {
        std::free(slab_);
    }
}

uint32_t MPMCRingBuffer::reserve_slot(size_t size) {
    // MPMC: Both producers and consumers use atomics
    uint64_t current_head = head_.load(std::memory_order_acquire);
    uint64_t current_tail = tail_.load(std::memory_order_acquire);
    
    do {
        if ((current_head + size - current_tail) >= buffer_size_) {
            return BUFFER_FULL;
        }
        
        // Retry if tail changed (another consumer)
        current_tail = tail_.load(std::memory_order_acquire);
        
    } while (!head_.compare_exchange_weak(current_head, current_head + size,
                                         std::memory_order_release,
                                         std::memory_order_acquire));
    
    return static_cast<uint32_t>(current_head & mask_);
}

void MPMCRingBuffer::advance_write_pointer(size_t new_write_pos) {
    // For MPMC, write pointer is already advanced in reserve_slot
    // This is a no-op since we advance atomically during reservation
}

void MPMCRingBuffer::advance_read_pointer(size_t size) {
    // MPMC: Multiple consumers need atomic CAS
    uint64_t current_tail = tail_.load(std::memory_order_acquire);
    while (!tail_.compare_exchange_weak(current_tail, current_tail + size,
                                       std::memory_order_release,
                                       std::memory_order_acquire)) {
        // Retry CAS
    }
}

bool MPMCRingBuffer::has_space(size_t size) const {
    uint64_t current_head = head_.load(std::memory_order_acquire);
    uint64_t current_tail = tail_.load(std::memory_order_acquire);
    return (current_head + size - current_tail) < buffer_size_;
}

} // namespace psyne