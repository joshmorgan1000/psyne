#pragma once

#include "slab_allocator.hpp"
#include <atomic>
#include <optional>
#include <cstring>

namespace psyne {

// Base template - not implemented
template <typename ProducerType, typename ConsumerType>
class RingBuffer;

// SPSC - Single Producer, Single Consumer (wait-free)
template <>
class RingBuffer<struct SingleProducer, struct SingleConsumer> {
public:
    using Header = SlabAllocator::SlabHeader;
    
    explicit RingBuffer(size_t size)
        : buffer_size_(next_power_of_2(size))
        , mask_(buffer_size_ - 1)
        , buffer_(static_cast<uint8_t*>(std::aligned_alloc(64, buffer_size_)))
        , write_pos_(0)
        , read_pos_(0) {
        if (!buffer_) throw std::bad_alloc();
        std::memset(buffer_, 0, buffer_size_);
    }
    
    ~RingBuffer() { std::free(buffer_); }
    
    struct WriteHandle {
        Header* header;
        void* data;
        size_t size;
        
        WriteHandle(Header* h, size_t s) : header(h), data(h->data()), size(s) {}
        template<typename T> T* as() { return static_cast<T*>(data); }
        void commit() { header->len = static_cast<uint32_t>(size); }
    };
    
    struct ReadHandle {
        const Header* header;
        const void* data;
        size_t size;
        
        ReadHandle(const Header* h) : header(h), data(h->data()), size(h->len) {}
        template<typename T> const T* as() const { return static_cast<const T*>(data); }
    };
    
    std::optional<WriteHandle> reserve(size_t size) {
        const size_t total_size = align_up(sizeof(Header) + size);
        size_t current_write = write_pos_;
        size_t current_read = read_pos_;
        
        if (buffer_size_ - (current_write - current_read) < total_size) {
            return std::nullopt;
        }
        
        auto* header = reinterpret_cast<Header*>(buffer_ + (current_write & mask_));
        write_pos_ = current_write + total_size;
        
        return WriteHandle(header, size);
    }
    
    std::optional<ReadHandle> read() {
        size_t current_read = read_pos_;
        if (current_read == write_pos_) {
            return std::nullopt;
        }
        
        auto* header = reinterpret_cast<Header*>(buffer_ + (current_read & mask_));
        read_pos_ = current_read + align_up(sizeof(Header) + header->len);
        
        return ReadHandle(header);
    }
    
    bool empty() const { return read_pos_ == write_pos_; }
    void* base() { return buffer_; }
    size_t capacity() const { return buffer_size_; }
    
private:
    static constexpr size_t align_up(size_t value) {
        return (value + 7) & ~7;
    }
    
    static size_t next_power_of_2(size_t n) {
        n--;
        n |= n >> 1; n |= n >> 2; n |= n >> 4;
        n |= n >> 8; n |= n >> 16; n |= n >> 32;
        return n + 1;
    }
    
    const size_t buffer_size_;
    const size_t mask_;
    uint8_t* buffer_;
    
    alignas(64) size_t write_pos_;
    alignas(64) size_t read_pos_;
};

// MPSC - Multiple Producer, Single Consumer
template <>
class RingBuffer<struct MultiProducer, struct SingleConsumer> {
public:
    using Header = SlabAllocator::SlabHeader;
    
    explicit RingBuffer(size_t size)
        : buffer_size_(next_power_of_2(size))
        , mask_(buffer_size_ - 1)
        , buffer_(static_cast<uint8_t*>(std::aligned_alloc(64, buffer_size_)))
        , write_pos_(0)
        , read_pos_(0) {
        if (!buffer_) throw std::bad_alloc();
        std::memset(buffer_, 0, buffer_size_);
    }
    
    ~RingBuffer() { std::free(buffer_); }
    
    struct WriteHandle {
        Header* header;
        void* data;
        size_t size;
        
        WriteHandle(Header* h, size_t s) : header(h), data(h->data()), size(s) {}
        template<typename T> T* as() { return static_cast<T*>(data); }
        void commit() { 
            header->len = static_cast<uint32_t>(size);
            std::atomic_thread_fence(std::memory_order_release);
        }
    };
    
    struct ReadHandle {
        const Header* header;
        const void* data;
        size_t size;
        
        ReadHandle(const Header* h) : header(h), data(h->data()), size(h->len) {}
        template<typename T> const T* as() const { return static_cast<const T*>(data); }
    };
    
    std::optional<WriteHandle> reserve(size_t size) {
        const size_t total_size = align_up(sizeof(Header) + size);
        
        // SPSC: No atomics needed for single producer
        size_t current_write = write_pos_;
        size_t current_read = read_pos_;
        
        if (buffer_size_ - (current_write - current_read) < total_size) {
            return std::nullopt;
        }
        
        auto* header = reinterpret_cast<Header*>(buffer_ + (current_write & mask_));
        header->len = 0;  // Mark as not ready
        
        write_pos_ = current_write + total_size;
        
        return WriteHandle(header, size);
    }
    
    std::optional<ReadHandle> read() {
        // SPSC: No atomics needed for single consumer
        size_t current_read = read_pos_;
        std::atomic_thread_fence(std::memory_order_acquire); // Ensure we see writes
        size_t current_write = write_pos_;
        
        if (current_read == current_write) {
            return std::nullopt;
        }
        
        auto* header = reinterpret_cast<Header*>(buffer_ + (current_read & mask_));
        
        // Wait for commit
        while (header->len == 0) {
            std::atomic_thread_fence(std::memory_order_acquire);
        }
        
        read_pos_ = current_read + align_up(sizeof(Header) + header->len);
        return ReadHandle(header);
    }
    
    bool empty() const { 
        std::atomic_thread_fence(std::memory_order_acquire);
        return read_pos_ == write_pos_;
    }
    
    void* base() { return buffer_; }
    size_t capacity() const { return buffer_size_; }
    
private:
    static constexpr size_t align_up(size_t value) {
        return (value + 7) & ~7;
    }
    
    static size_t next_power_of_2(size_t n) {
        n--;
        n |= n >> 1; n |= n >> 2; n |= n >> 4;
        n |= n >> 8; n |= n >> 16; n |= n >> 32;
        return n + 1;
    }
    
    const size_t buffer_size_;
    const size_t mask_;
    uint8_t* buffer_;
    
    // Cache line padding to prevent false sharing
    // SPSC doesn't need atomics - just proper memory ordering
    alignas(64) size_t write_pos_;
    char padding1_[64 - sizeof(size_t)];
    
    alignas(64) size_t read_pos_;
    char padding2_[64 - sizeof(size_t)];
};

// SPMC - Single Producer, Multiple Consumer
template <>
class RingBuffer<struct SingleProducer, struct MultiConsumer> {
public:
    using Header = SlabAllocator::SlabHeader;
    
    explicit RingBuffer(size_t size)
        : buffer_size_(next_power_of_2(size))
        , mask_(buffer_size_ - 1)
        , buffer_(static_cast<uint8_t*>(std::aligned_alloc(64, buffer_size_)))
        , write_pos_(0)
        , read_pos_(0) {
        if (!buffer_) throw std::bad_alloc();
        std::memset(buffer_, 0, buffer_size_);
    }
    
    ~RingBuffer() { std::free(buffer_); }
    
    struct WriteHandle {
        Header* header;
        void* data;
        size_t size;
        
        WriteHandle(Header* h, size_t s) : header(h), data(h->data()), size(s) {}
        template<typename T> T* as() { return static_cast<T*>(data); }
        void commit() { 
            header->len = static_cast<uint32_t>(size);
            std::atomic_thread_fence(std::memory_order_release);
        }
    };
    
    struct ReadHandle {
        const Header* header;
        const void* data;
        size_t size;
        
        ReadHandle(const Header* h) : header(h), data(h->data()), size(h->len) {}
        template<typename T> const T* as() const { return static_cast<const T*>(data); }
    };
    
    std::optional<WriteHandle> reserve(size_t size) {
        const size_t total_size = align_up(sizeof(Header) + size);
        size_t current_write = write_pos_;
        size_t current_read = read_pos_.load(std::memory_order_acquire);
        
        if (buffer_size_ - (current_write - current_read) < total_size) {
            return std::nullopt;
        }
        
        auto* header = reinterpret_cast<Header*>(buffer_ + (current_write & mask_));
        write_pos_ = current_write + total_size;
        
        return WriteHandle(header, size);
    }
    
    std::optional<ReadHandle> read() {
        size_t current_read = read_pos_.load(std::memory_order_relaxed);
        size_t next_read;
        
        do {
            if (current_read == write_pos_) {
                return std::nullopt;
            }
            
            auto* header = reinterpret_cast<Header*>(buffer_ + (current_read & mask_));
            
            // Wait for commit
            while (header->len == 0) {
                std::atomic_thread_fence(std::memory_order_acquire);
            }
            
            next_read = current_read + align_up(sizeof(Header) + header->len);
        } while (!read_pos_.compare_exchange_weak(
            current_read, next_read,
            std::memory_order_relaxed,
            std::memory_order_relaxed));
        
        auto* header = reinterpret_cast<Header*>(buffer_ + (current_read & mask_));
        return ReadHandle(header);
    }
    
    bool empty() const { 
        return read_pos_.load(std::memory_order_acquire) == write_pos_;
    }
    
    void* base() { return buffer_; }
    size_t capacity() const { return buffer_size_; }
    
private:
    static constexpr size_t align_up(size_t value) {
        return (value + 7) & ~7;
    }
    
    static size_t next_power_of_2(size_t n) {
        n--;
        n |= n >> 1; n |= n >> 2; n |= n >> 4;
        n |= n >> 8; n |= n >> 16; n |= n >> 32;
        return n + 1;
    }
    
    const size_t buffer_size_;
    const size_t mask_;
    uint8_t* buffer_;
    
    // Cache line padding to prevent false sharing
    alignas(64) size_t write_pos_;
    char padding1_[64 - sizeof(size_t)];
    
    alignas(64) std::atomic<size_t> read_pos_;
    char padding2_[64 - sizeof(std::atomic<size_t>)];
};

// MPMC - Multiple Producer, Multiple Consumer
template <>
class RingBuffer<struct MultiProducer, struct MultiConsumer> {
public:
    using Header = SlabAllocator::SlabHeader;
    
    explicit RingBuffer(size_t size)
        : buffer_size_(next_power_of_2(size))
        , mask_(buffer_size_ - 1)
        , buffer_(static_cast<uint8_t*>(std::aligned_alloc(64, buffer_size_)))
        , write_pos_(0)
        , read_pos_(0)
        , commit_pos_(0) {
        if (!buffer_) throw std::bad_alloc();
        std::memset(buffer_, 0, buffer_size_);
    }
    
    ~RingBuffer() { std::free(buffer_); }
    
    struct WriteHandle {
        Header* header;
        void* data;
        size_t size;
        std::atomic<size_t>* commit_pos;
        size_t pos;
        size_t total_size;
        
        WriteHandle(Header* h, size_t s, std::atomic<size_t>* cp, size_t p, size_t ts) 
            : header(h), data(h->data()), size(s), commit_pos(cp), pos(p), total_size(ts) {}
        
        template<typename T> T* as() { return static_cast<T*>(data); }
        
        void commit() { 
            header->len = static_cast<uint32_t>(size);
            std::atomic_thread_fence(std::memory_order_release);
            
            // Update commit position
            size_t expected = pos;
            while (!commit_pos->compare_exchange_weak(expected, pos + total_size)) {
                expected = pos;
            }
        }
    };
    
    struct ReadHandle {
        const Header* header;
        const void* data;
        size_t size;
        
        ReadHandle(const Header* h) : header(h), data(h->data()), size(h->len) {}
        template<typename T> const T* as() const { return static_cast<const T*>(data); }
    };
    
    std::optional<WriteHandle> reserve(size_t size) {
        const size_t total_size = align_up(sizeof(Header) + size);
        size_t current_write = write_pos_.load(std::memory_order_relaxed);
        size_t next_write;
        
        do {
            size_t current_read = read_pos_.load(std::memory_order_acquire);
            if (buffer_size_ - (current_write - current_read) < total_size) {
                return std::nullopt;
            }
            next_write = current_write + total_size;
        } while (!write_pos_.compare_exchange_weak(
            current_write, next_write,
            std::memory_order_relaxed,
            std::memory_order_relaxed));
        
        auto* header = reinterpret_cast<Header*>(buffer_ + (current_write & mask_));
        header->len = 0;  // Mark as not ready
        
        return WriteHandle(header, size, &commit_pos_, current_write, total_size);
    }
    
    std::optional<ReadHandle> read() {
        size_t current_read = read_pos_.load(std::memory_order_relaxed);
        size_t next_read;
        
        do {
            size_t commit = commit_pos_.load(std::memory_order_acquire);
            if (current_read >= commit) {
                return std::nullopt;
            }
            
            auto* header = reinterpret_cast<Header*>(buffer_ + (current_read & mask_));
            
            // Wait for commit
            while (header->len == 0) {
                std::atomic_thread_fence(std::memory_order_acquire);
            }
            
            next_read = current_read + align_up(sizeof(Header) + header->len);
        } while (!read_pos_.compare_exchange_weak(
            current_read, next_read,
            std::memory_order_relaxed,
            std::memory_order_relaxed));
        
        auto* header = reinterpret_cast<Header*>(buffer_ + (current_read & mask_));
        return ReadHandle(header);
    }
    
    bool empty() const { 
        return read_pos_.load(std::memory_order_acquire) >= 
               commit_pos_.load(std::memory_order_acquire);
    }
    
    void* base() { return buffer_; }
    size_t capacity() const { return buffer_size_; }
    
private:
    static constexpr size_t align_up(size_t value) {
        return (value + 7) & ~7;
    }
    
    static size_t next_power_of_2(size_t n) {
        n--;
        n |= n >> 1; n |= n >> 2; n |= n >> 4;
        n |= n >> 8; n |= n >> 16; n |= n >> 32;
        return n + 1;
    }
    
    const size_t buffer_size_;
    const size_t mask_;
    uint8_t* buffer_;
    
    // Cache line padding to prevent false sharing
    alignas(64) std::atomic<size_t> write_pos_;
    char padding1_[64 - sizeof(std::atomic<size_t>)];
    
    alignas(64) std::atomic<size_t> read_pos_;
    char padding2_[64 - sizeof(std::atomic<size_t>)];
    
    alignas(64) std::atomic<size_t> commit_pos_;
    char padding3_[64 - sizeof(std::atomic<size_t>)];
};

using SPSCRingBuffer = RingBuffer<SingleProducer, SingleConsumer>;
using SPMCRingBuffer = RingBuffer<SingleProducer, MultiConsumer>;
using MPSCRingBuffer = RingBuffer<MultiProducer, SingleConsumer>;
using MPMCRingBuffer = RingBuffer<MultiProducer, MultiConsumer>;

}  // namespace psyne