#pragma once

/**
 * @file ring_buffer_impl.hpp
 * @brief Implementation details for lock-free ring buffers
 * @author Psyne Contributors
 * @date 2025
 *
 * This file contains the core ring buffer implementations that power
 * Psyne's zero-copy message passing. Different implementations are
 * provided for different synchronization modes (SPSC, MPSC, etc.).
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>

namespace psyne {
namespace detail {

/**
 * @struct SlabHeader
 * @brief Header for each message in the ring buffer
 *
 * Each message in the ring buffer is prefixed with this header,
 * which contains the message length and alignment padding.
 */
struct SlabHeader {
    uint32_t len;      ///< Length of the message data in bytes
    uint32_t reserved; ///< Reserved for alignment and future use

    /**
     * @brief Get pointer to message data
     * @return Pointer to data immediately following the header
     */
    void *data() {
        return reinterpret_cast<uint8_t *>(this) + sizeof(SlabHeader);
    }

    /**
     * @brief Get const pointer to message data
     * @return Const pointer to data immediately following the header
     */
    const void *data() const {
        return reinterpret_cast<const uint8_t *>(this) + sizeof(SlabHeader);
    }
};

/**
 * @class RingBufferBase
 * @brief Abstract base class for ring buffer implementations
 *
 * This class defines the interface that all ring buffer implementations
 * must provide. Different implementations handle different synchronization
 * modes (SPSC, MPSC, SPMC, MPMC) with varying performance characteristics.
 */
class RingBufferBase {
public:
    virtual ~RingBufferBase() = default;

    /**
     * @struct WriteHandle
     * @brief Handle for writing data into the ring buffer
     *
     * Represents a reserved slot in the ring buffer. The user writes
     * data through this handle and then calls commit() to make it
     * available to readers.
     */
    struct WriteHandle {
        SlabHeader *header; ///< Pointer to the message header
        void *data;         ///< Pointer to the data area
        size_t size;        ///< Size of the reserved space
        RingBufferBase
            *ring_buffer; ///< Back-reference for MPMC commit tracking

        /**
         * @brief Construct a write handle
         * @param h Pointer to the slab header
         * @param s Size of the reserved space
         * @param rb Ring buffer reference (for MPMC mode)
         */
        WriteHandle(SlabHeader *h, size_t s, RingBufferBase *rb = nullptr)
            : header(h), data(h->data()), size(s), ring_buffer(rb) {}

        /**
         * @brief Commit the written data
         *
         * Updates the header with the actual data size and notifies
         * the ring buffer that the write is complete. After commit(),
         * the data becomes visible to readers.
         */
        void commit() {
            header->len = static_cast<uint32_t>(size);
            if (ring_buffer) {
                ring_buffer->on_commit(this);
            }
        }
    };

    /**
     * @struct ReadHandle
     * @brief Handle for reading data from the ring buffer
     *
     * Represents a message that is ready to be read. The handle
     * provides access to the message data and its size.
     */
    struct ReadHandle {
        const SlabHeader *header; ///< Pointer to the message header
        const void *data;         ///< Pointer to the message data
        size_t size;              ///< Size of the message data

        /**
         * @brief Construct a read handle from a slab header
         * @param h Pointer to the slab header
         */
        ReadHandle(const SlabHeader *h)
            : header(h), data(h->data()), size(h->len) {}
    };

    /**
     * @brief Reserve space for writing
     * @param size Number of bytes to reserve
     * @return WriteHandle if space is available, std::nullopt if buffer is full
     *
     * This method is thread-safe according to the ring buffer's synchronization
     * mode.
     */
    virtual std::optional<WriteHandle> reserve(size_t size) = 0;

    /**
     * @brief Read the next available message
     * @return ReadHandle if a message is available, std::nullopt if buffer is
     * empty
     *
     * This method is thread-safe according to the ring buffer's synchronization
     * mode.
     */
    virtual std::optional<ReadHandle> read() = 0;

    /**
     * @brief Check if the buffer is empty
     * @return true if no messages are available, false otherwise
     */
    virtual bool empty() const = 0;

    /**
     * @brief Get the total capacity of the ring buffer
     * @return Capacity in bytes
     */
    virtual size_t capacity() const = 0;

    /**
     * @brief Callback for when a write is committed (MPMC mode)
     * @param handle The write handle that was committed
     *
     * This is used by MPMC implementations to track committed writes.
     * Other modes can use the default empty implementation.
     */
    virtual void on_commit(WriteHandle *handle) {} // Override in MPMC
};

// SPSC implementation
class SPSCRingBuffer : public RingBufferBase {
public:
    explicit SPSCRingBuffer(size_t size);
    ~SPSCRingBuffer();

    std::optional<WriteHandle> reserve(size_t size) override;
    std::optional<ReadHandle> read() override;
    bool empty() const override;
    size_t capacity() const override {
        return buffer_size_;
    }

private:
    static size_t next_power_of_2(size_t n);
    static constexpr size_t align_up(size_t value) {
        return (value + 7) & ~7;
    }

    const size_t buffer_size_;
    const size_t mask_;
    uint8_t *buffer_;

    alignas(64) size_t write_pos_;
    alignas(64) size_t read_pos_;
};

// MPSC implementation
class MPSCRingBuffer : public RingBufferBase {
public:
    explicit MPSCRingBuffer(size_t size);
    ~MPSCRingBuffer();

    std::optional<WriteHandle> reserve(size_t size) override;
    std::optional<ReadHandle> read() override;
    bool empty() const override;
    size_t capacity() const override {
        return buffer_size_;
    }

private:
    static size_t next_power_of_2(size_t n);
    static constexpr size_t align_up(size_t value) {
        return (value + 7) & ~7;
    }

    const size_t buffer_size_;
    const size_t mask_;
    uint8_t *buffer_;

    alignas(64) std::atomic<size_t> write_pos_;
    char padding_[64 - sizeof(std::atomic<size_t>)];
    alignas(64) size_t read_pos_;
};

// SPMC implementation
class SPMCRingBuffer : public RingBufferBase {
public:
    explicit SPMCRingBuffer(size_t size);
    ~SPMCRingBuffer();

    std::optional<WriteHandle> reserve(size_t size) override;
    std::optional<ReadHandle> read() override;
    bool empty() const override;
    size_t capacity() const override {
        return buffer_size_;
    }

private:
    static size_t next_power_of_2(size_t n);
    static constexpr size_t align_up(size_t value) {
        return (value + 7) & ~7;
    }

    const size_t buffer_size_;
    const size_t mask_;
    uint8_t *buffer_;

    alignas(64) size_t write_pos_;
    char padding_[64 - sizeof(size_t)];
    alignas(64) std::atomic<size_t> read_pos_;
};

// MPMC implementation
class MPMCRingBuffer : public RingBufferBase {
public:
    explicit MPMCRingBuffer(size_t size);
    ~MPMCRingBuffer();

    std::optional<WriteHandle> reserve(size_t size) override;
    std::optional<ReadHandle> read() override;
    bool empty() const override;
    size_t capacity() const override {
        return buffer_size_;
    }
    void on_commit(WriteHandle *handle) override;

private:
    static size_t next_power_of_2(size_t n);
    static constexpr size_t align_up(size_t value) {
        return (value + 7) & ~7;
    }

    const size_t buffer_size_;
    const size_t mask_;
    uint8_t *buffer_;

    alignas(64) std::atomic<size_t> write_pos_;
    alignas(64) std::atomic<size_t> read_pos_;
    alignas(64) std::atomic<size_t> commit_pos_;
};

} // namespace detail
} // namespace psyne