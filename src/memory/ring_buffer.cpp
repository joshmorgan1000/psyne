#include "ring_buffer_impl.hpp"
#include <cstdlib>
#include <new>

namespace psyne {
namespace detail {

// Utility function
static size_t next_power_of_2(size_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

// Round up to next multiple of alignment
static size_t align_size(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

// SPSC Implementation
SPSCRingBuffer::SPSCRingBuffer(size_t size)
    : buffer_size_(next_power_of_2(size)), mask_(buffer_size_ - 1),
      buffer_(static_cast<uint8_t *>(std::aligned_alloc(64, align_size(buffer_size_, 64)))),
      write_pos_(0), read_pos_(0) {
    if (!buffer_)
        throw std::bad_alloc();
    std::memset(buffer_, 0, buffer_size_);
}

SPSCRingBuffer::~SPSCRingBuffer() {
    std::free(buffer_);
}

std::optional<SPSCRingBuffer::WriteHandle>
SPSCRingBuffer::reserve(size_t size) {
    const size_t total_size = align_up(sizeof(SlabHeader) + size);
    size_t current_write = write_pos_;
    size_t current_read = read_pos_;

    // Check if we have enough space
    size_t used = current_write - current_read;
    if (used + total_size > buffer_size_) {
        return std::nullopt;
    }

    // Check if this allocation would wrap around the buffer
    size_t write_offset = current_write & mask_;
    if (write_offset + total_size > buffer_size_) {
        return std::nullopt;
    }

    auto *header = reinterpret_cast<SlabHeader *>(buffer_ + write_offset);
    write_pos_ = current_write + total_size;

    return WriteHandle(header, size);
}

std::optional<SPSCRingBuffer::ReadHandle> SPSCRingBuffer::read() {
    size_t current_read = read_pos_;
    if (current_read == write_pos_) {
        return std::nullopt;
    }

    auto *header =
        reinterpret_cast<SlabHeader *>(buffer_ + (current_read & mask_));
    read_pos_ = current_read + align_up(sizeof(SlabHeader) + header->len);

    return ReadHandle(header);
}

bool SPSCRingBuffer::empty() const {
    return read_pos_ == write_pos_;
}

size_t SPSCRingBuffer::next_power_of_2(size_t n) {
    return ::psyne::detail::next_power_of_2(n);
}

// MPSC Implementation
MPSCRingBuffer::MPSCRingBuffer(size_t size)
    : buffer_size_(next_power_of_2(size)), mask_(buffer_size_ - 1),
      buffer_(static_cast<uint8_t *>(std::aligned_alloc(64, align_size(buffer_size_, 64)))),
      write_pos_(0), read_pos_(0) {
    if (!buffer_)
        throw std::bad_alloc();
    std::memset(buffer_, 0, buffer_size_);
}

MPSCRingBuffer::~MPSCRingBuffer() {
    std::free(buffer_);
}

std::optional<MPSCRingBuffer::WriteHandle>
MPSCRingBuffer::reserve(size_t size) {
    const size_t total_size = align_up(sizeof(SlabHeader) + size);

    // MPSC: Atomic CAS for multiple producers
    while (true) {
        size_t current_write = write_pos_.load(std::memory_order_relaxed);
        size_t current_read = read_pos_;

        // Check if we have enough space
        size_t used = current_write - current_read;
        if (used + total_size > buffer_size_) {
            return std::nullopt;
        }

        // Check if this allocation would wrap around the buffer
        size_t write_offset = current_write & mask_;
        if (write_offset + total_size > buffer_size_) {
            return std::nullopt;
        }

        size_t new_write = current_write + total_size;
        if (write_pos_.compare_exchange_weak(current_write, new_write,
                                             std::memory_order_release,
                                             std::memory_order_relaxed)) {
            auto *header =
                reinterpret_cast<SlabHeader *>(buffer_ + write_offset);
            header->len = 0; // Mark as not ready
            return WriteHandle(header, size);
        }
    }
}

std::optional<MPSCRingBuffer::ReadHandle> MPSCRingBuffer::read() {
    size_t current_read = read_pos_;
    std::atomic_thread_fence(std::memory_order_acquire);
    size_t current_write = write_pos_.load(std::memory_order_acquire);

    if (current_read == current_write) {
        return std::nullopt;
    }

    auto *header =
        reinterpret_cast<SlabHeader *>(buffer_ + (current_read & mask_));

    // Wait for commit
    while (header->len == 0) {
        std::atomic_thread_fence(std::memory_order_acquire);
    }

    read_pos_ = current_read + align_up(sizeof(SlabHeader) + header->len);
    return ReadHandle(header);
}

bool MPSCRingBuffer::empty() const {
    std::atomic_thread_fence(std::memory_order_acquire);
    return read_pos_ == write_pos_.load(std::memory_order_acquire);
}

size_t MPSCRingBuffer::next_power_of_2(size_t n) {
    return ::psyne::detail::next_power_of_2(n);
}

// SPMC Implementation
SPMCRingBuffer::SPMCRingBuffer(size_t size)
    : buffer_size_(next_power_of_2(size)), mask_(buffer_size_ - 1),
      buffer_(static_cast<uint8_t *>(std::aligned_alloc(64, align_size(buffer_size_, 64)))),
      write_pos_(0), read_pos_(0) {
    if (!buffer_)
        throw std::bad_alloc();
    std::memset(buffer_, 0, buffer_size_);
}

SPMCRingBuffer::~SPMCRingBuffer() {
    std::free(buffer_);
}

std::optional<SPMCRingBuffer::WriteHandle>
SPMCRingBuffer::reserve(size_t size) {
    const size_t total_size = align_up(sizeof(SlabHeader) + size);
    size_t current_write = write_pos_;
    size_t current_read = read_pos_.load(std::memory_order_acquire);

    // Check if we have enough space
    size_t used = current_write - current_read;
    if (used + total_size > buffer_size_) {
        return std::nullopt;
    }

    // Check if this allocation would wrap around the buffer
    size_t write_offset = current_write & mask_;
    if (write_offset + total_size > buffer_size_) {
        return std::nullopt;
    }

    auto *header = reinterpret_cast<SlabHeader *>(buffer_ + write_offset);
    write_pos_ = current_write + total_size;

    return WriteHandle(header, size);
}

std::optional<SPMCRingBuffer::ReadHandle> SPMCRingBuffer::read() {
    size_t current_read = read_pos_.load(std::memory_order_relaxed);
    size_t next_read;

    do {
        if (current_read == write_pos_) {
            return std::nullopt;
        }

        auto *header =
            reinterpret_cast<SlabHeader *>(buffer_ + (current_read & mask_));

        // Wait for commit
        while (header->len == 0) {
            std::atomic_thread_fence(std::memory_order_acquire);
        }

        next_read = current_read + align_up(sizeof(SlabHeader) + header->len);
    } while (!read_pos_.compare_exchange_weak(current_read, next_read,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed));

    auto *header =
        reinterpret_cast<SlabHeader *>(buffer_ + (current_read & mask_));
    return ReadHandle(header);
}

bool SPMCRingBuffer::empty() const {
    return read_pos_.load(std::memory_order_acquire) == write_pos_;
}

size_t SPMCRingBuffer::next_power_of_2(size_t n) {
    return ::psyne::detail::next_power_of_2(n);
}

// MPMC Implementation
MPMCRingBuffer::MPMCRingBuffer(size_t size)
    : buffer_size_(next_power_of_2(size)), mask_(buffer_size_ - 1),
      buffer_(static_cast<uint8_t *>(std::aligned_alloc(64, align_size(buffer_size_, 64)))),
      write_pos_(0), read_pos_(0), commit_pos_(0) {
    if (!buffer_)
        throw std::bad_alloc();
    std::memset(buffer_, 0, buffer_size_);
}

MPMCRingBuffer::~MPMCRingBuffer() {
    std::free(buffer_);
}

std::optional<MPMCRingBuffer::WriteHandle>
MPMCRingBuffer::reserve(size_t size) {
    const size_t total_size = align_up(sizeof(SlabHeader) + size);
    size_t current_write = write_pos_.load(std::memory_order_relaxed);
    size_t next_write;

    do {
        size_t current_read = read_pos_.load(std::memory_order_acquire);

        // Check if we have enough space
        size_t used = current_write - current_read;
        if (used + total_size > buffer_size_) {
            return std::nullopt;
        }

        // Check if this allocation would wrap around the buffer
        size_t write_offset = current_write & mask_;
        if (write_offset + total_size > buffer_size_) {
            return std::nullopt;
        }

        next_write = current_write + total_size;
    } while (!write_pos_.compare_exchange_weak(current_write, next_write,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed));

    auto *header =
        reinterpret_cast<SlabHeader *>(buffer_ + (current_write & mask_));
    header->len = 0; // Mark as not ready

    // For MPMC, we need to track commit position
    return WriteHandle(header, size, this);
}

std::optional<MPMCRingBuffer::ReadHandle> MPMCRingBuffer::read() {
    size_t current_read = read_pos_.load(std::memory_order_relaxed);
    size_t next_read;

    do {
        size_t commit = commit_pos_.load(std::memory_order_acquire);
        if (current_read >= commit) {
            return std::nullopt;
        }

        auto *header =
            reinterpret_cast<SlabHeader *>(buffer_ + (current_read & mask_));

        // Wait for commit
        while (header->len == 0) {
            std::atomic_thread_fence(std::memory_order_acquire);
        }

        next_read = current_read + align_up(sizeof(SlabHeader) + header->len);
    } while (!read_pos_.compare_exchange_weak(current_read, next_read,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed));

    auto *header =
        reinterpret_cast<SlabHeader *>(buffer_ + (current_read & mask_));
    return ReadHandle(header);
}

bool MPMCRingBuffer::empty() const {
    return read_pos_.load(std::memory_order_acquire) >=
           commit_pos_.load(std::memory_order_acquire);
}

void MPMCRingBuffer::on_commit(WriteHandle* handle) {
    if (!handle || !handle->header) return;
    
    // Calculate the position after this message
    size_t message_size = align_up(sizeof(SlabHeader) + handle->size);
    size_t message_start = reinterpret_cast<uint8_t*>(handle->header) - buffer_;
    size_t message_end = message_start + message_size;
    
    // Update commit position to indicate this message is available for reading
    // For MPMC, we need to ensure commit_pos_ advances sequentially
    size_t expected_commit = message_start;
    while (!commit_pos_.compare_exchange_weak(expected_commit, message_end,
                                              std::memory_order_release,
                                              std::memory_order_relaxed)) {
        // If another thread is ahead of us, we're done
        if (expected_commit >= message_end) {
            break;
        }
    }
}

size_t MPMCRingBuffer::next_power_of_2(size_t n) {
    return ::psyne::detail::next_power_of_2(n);
}

} // namespace detail
} // namespace psyne