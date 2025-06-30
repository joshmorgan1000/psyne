/**
 * @file dynamic_ring_buffer.hpp
 * @brief Ring buffer with dynamic slab allocation
 *
 * This ring buffer uses the dynamic slab allocator to automatically
 * grow capacity based on usage patterns.
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <psyne/memory/dynamic_slab_allocator.hpp>

namespace psyne {
namespace memory {

/**
 * @brief Ring buffer with dynamic slab growth
 */
class DynamicRingBuffer {
public:
    /**
     * @brief Configuration for dynamic ring buffer
     */
    struct Config {
        DynamicSlabConfig slab_config;
        size_t max_message_size =
            16 * 1024 * 1024;  ///< Maximum single message size (16MB)
        size_t alignment = 64; ///< Message alignment (cache line)
        bool thread_local_allocator = false; ///< Use thread-local allocators
    };

    /**
     * @brief Message handle for writing
     */
    struct WriteHandle {
        void *data;
        size_t size;
        DynamicRingBuffer *buffer;
        size_t sequence;

        WriteHandle(void *d, size_t s, DynamicRingBuffer *b, size_t seq)
            : data(d), size(s), buffer(b), sequence(seq) {}

        void commit() {
            if (buffer) {
                buffer->commit_write(this);
            }
        }
    };

    /**
     * @brief Message handle for reading
     */
    struct ReadHandle {
        const void *data;
        size_t size;
        uint32_t type_id;
        size_t sequence;

        ReadHandle(const void *d, size_t s, uint32_t t, size_t seq)
            : data(d), size(s), type_id(t), sequence(seq) {}
    };

    explicit DynamicRingBuffer(const Config &config = {});
    ~DynamicRingBuffer();

    /**
     * @brief Reserve space for writing
     * @param size Size in bytes
     * @param type_id Message type ID
     * @return Write handle or nullptr if failed
     */
    std::unique_ptr<WriteHandle> reserve(size_t size, uint32_t type_id = 0);

    /**
     * @brief Try to read next message
     * @return Read handle or nullptr if no message available
     */
    std::unique_ptr<ReadHandle> try_read();

    /**
     * @brief Release a read message
     * @param handle Read handle to release
     */
    void release(std::unique_ptr<ReadHandle> handle);

    /**
     * @brief Get buffer statistics
     */
    struct Stats {
        DynamicSlabStats slab_stats;
        size_t messages_written = 0;
        size_t messages_read = 0;
        size_t bytes_written = 0;
        size_t bytes_read = 0;
        size_t current_messages = 0;
        double avg_message_size = 0.0;
    };

    Stats get_stats() const;

    /**
     * @brief Force maintenance (slab growth/shrink check)
     */
    void perform_maintenance();

    /**
     * @brief Reset buffer (clear all messages)
     */
    void reset();

private:
    struct MessageHeader {
        uint32_t size;
        uint32_t type_id;
        uint64_t sequence;
        uint64_t timestamp;

        void *data() {
            return reinterpret_cast<uint8_t *>(this) + sizeof(MessageHeader);
        }

        const void *data() const {
            return reinterpret_cast<const uint8_t *>(this) +
                   sizeof(MessageHeader);
        }
    };

    Config config_;
    std::unique_ptr<DynamicSlabAllocator> allocator_;
    std::unique_ptr<ThreadLocalSlabAllocator> tls_allocator_;

    // Message tracking
    mutable std::mutex mutex_;
    std::atomic<uint64_t> write_sequence_{0};
    std::atomic<uint64_t> read_sequence_{0};
    std::atomic<size_t> messages_written_{0};
    std::atomic<size_t> messages_read_{0};
    std::atomic<size_t> bytes_written_{0};
    std::atomic<size_t> bytes_read_{0};

    // Message queue (simple implementation - could be optimized)
    struct QueuedMessage {
        MessageHeader *header;
        size_t alloc_size;
    };
    std::vector<QueuedMessage> message_queue_;
    size_t read_index_ = 0;

    // Helper methods
    void commit_write(WriteHandle *handle);
    DynamicSlabAllocator *get_allocator();
};

/**
 * @brief Thread-safe dynamic ring buffer
 */
class ThreadSafeDynamicRingBuffer : public DynamicRingBuffer {
public:
    using DynamicRingBuffer::DynamicRingBuffer;

    std::unique_ptr<WriteHandle> reserve(size_t size, uint32_t type_id = 0) {
        std::lock_guard<std::mutex> lock(write_mutex_);
        return DynamicRingBuffer::reserve(size, type_id);
    }

    std::unique_ptr<ReadHandle> try_read() {
        std::lock_guard<std::mutex> lock(read_mutex_);
        return DynamicRingBuffer::try_read();
    }

private:
    mutable std::mutex write_mutex_;
    mutable std::mutex read_mutex_;
};

} // namespace memory
} // namespace psyne