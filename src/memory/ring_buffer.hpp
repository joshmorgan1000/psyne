#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace psyne {

/**
 * @brief Zero-copy ring buffer interface
 *
 * This interface implements the true zero-copy design from CORE_DESIGN.md:
 * - Messages are views, not objects
 * - Ring buffer = pre-allocated slab
 * - No handles, no commits, just pointer arithmetic
 */
class RingBuffer {
public:
    virtual ~RingBuffer() = default;

    /**
     * @brief Get base pointer to ring buffer memory
     * @return Pointer to start of ring buffer slab
     */
    virtual uint8_t *base_ptr() = 0;
    virtual const uint8_t *base_ptr() const = 0;

    /**
     * @brief Reserve slot in ring buffer and return offset
     * @param size Size of message to reserve
     * @return Offset within buffer, or BUFFER_FULL if no space
     */
    virtual uint32_t reserve_slot(size_t size) = 0;

    /**
     * @brief Advance write pointer to make message available for reading
     * @param new_write_pos New write position
     */
    virtual void advance_write_pointer(size_t new_write_pos) = 0;

    /**
     * @brief Advance read pointer after consuming message
     * @param size Size of message consumed
     */
    virtual void advance_read_pointer(size_t size) = 0;

    /**
     * @brief Get current write position
     */
    virtual size_t write_position() const = 0;

    /**
     * @brief Get current read position
     */
    virtual size_t read_position() const = 0;

    /**
     * @brief Check if buffer has available space
     * @param size Size needed
     * @return True if space available
     */
    virtual bool has_space(size_t size) const = 0;

    /**
     * @brief Get buffer capacity
     */
    virtual size_t capacity() const = 0;

    /**
     * @brief Buffer full constant
     */
    static constexpr uint32_t BUFFER_FULL = 0xFFFFFFFF;
};

/**
 * @brief SPSC (Single Producer Single Consumer) Ring Buffer
 *
 * Optimized for maximum performance with zero atomics.
 * Follows the original v1.0.0 design from CORE_DESIGN.md lines 226-262.
 */
class SPSCRingBuffer : public RingBuffer {
public:
    explicit SPSCRingBuffer(size_t size);
    ~SPSCRingBuffer();

    uint8_t *base_ptr() noexcept override {
        return slab_;
    }
    const uint8_t *base_ptr() const noexcept override {
        return slab_;
    }

    uint32_t reserve_slot(size_t size) noexcept override;
    void advance_write_pointer(size_t new_write_pos) noexcept override;
    void advance_read_pointer(size_t size) noexcept override;

    size_t write_position() const noexcept override {
        return head_;
    }
    size_t read_position() const noexcept override {
        return tail_;
    }
    bool has_space(size_t size) const noexcept override;
    size_t capacity() const noexcept override {
        return buffer_size_;
    }

private:
    static size_t next_power_of_2(size_t n);

    // Ring buffer data
    uint8_t *slab_;
    size_t buffer_size_;
    size_t mask_; // size - 1 for power-of-2 modulo

    // SPSC: Zero atomics design for maximum performance
    alignas(64) uint64_t head_; // Producer-owned (cache line 0)
    uint8_t padding1_[56];      // Cache line separation
    alignas(64) uint64_t tail_; // Consumer-owned (cache line 1)
    uint8_t padding2_[56];      // Cache line separation
};

/**
 * @brief MPSC (Multi Producer Single Consumer) Ring Buffer
 *
 * Multiple producers need atomic head CAS operations.
 */
class MPSCRingBuffer : public RingBuffer {
public:
    explicit MPSCRingBuffer(size_t size);
    ~MPSCRingBuffer();

    uint8_t *base_ptr() noexcept override {
        return slab_;
    }
    const uint8_t *base_ptr() const noexcept override {
        return slab_;
    }

    uint32_t reserve_slot(size_t size) override;
    void advance_write_pointer(size_t new_write_pos) override;
    void advance_read_pointer(size_t size) override;

    size_t write_position() const override {
        return head_.load(std::memory_order_acquire);
    }
    size_t read_position() const override {
        return tail_;
    }
    bool has_space(size_t size) const override;
    size_t capacity() const override {
        return buffer_size_;
    }

private:
    static size_t next_power_of_2(size_t n);

    uint8_t *slab_;
    size_t buffer_size_;
    size_t mask_;

    alignas(64) std::atomic<uint64_t> head_; // Multiple producers need atomic
    uint8_t padding1_[56];
    alignas(64) uint64_t tail_; // Single consumer, no atomic needed
    uint8_t padding2_[56];
};

/**
 * @brief SPMC (Single Producer Multi Consumer) Ring Buffer
 *
 * Multiple consumers need atomic tail CAS operations.
 */
class SPMCRingBuffer : public RingBuffer {
public:
    explicit SPMCRingBuffer(size_t size);
    ~SPMCRingBuffer();

    uint8_t *base_ptr() noexcept override {
        return slab_;
    }
    const uint8_t *base_ptr() const noexcept override {
        return slab_;
    }

    uint32_t reserve_slot(size_t size) override;
    void advance_write_pointer(size_t new_write_pos) override;
    void advance_read_pointer(size_t size) override;

    size_t write_position() const override {
        return head_;
    }
    size_t read_position() const override {
        return tail_.load(std::memory_order_acquire);
    }
    bool has_space(size_t size) const override;
    size_t capacity() const override {
        return buffer_size_;
    }

private:
    static size_t next_power_of_2(size_t n);

    uint8_t *slab_;
    size_t buffer_size_;
    size_t mask_;

    alignas(64) uint64_t head_; // Single producer, no atomic needed
    uint8_t padding1_[56];
    alignas(64) std::atomic<uint64_t> tail_; // Multiple consumers need atomic
    uint8_t padding2_[56];
};

/**
 * @brief MPMC (Multi Producer Multi Consumer) Ring Buffer
 *
 * Both head and tail need atomic CAS operations.
 */
class MPMCRingBuffer : public RingBuffer {
public:
    explicit MPMCRingBuffer(size_t size);
    ~MPMCRingBuffer();

    uint8_t *base_ptr() noexcept override {
        return slab_;
    }
    const uint8_t *base_ptr() const noexcept override {
        return slab_;
    }

    uint32_t reserve_slot(size_t size) override;
    void advance_write_pointer(size_t new_write_pos) override;
    void advance_read_pointer(size_t size) override;

    size_t write_position() const override {
        return head_.load(std::memory_order_acquire);
    }
    size_t read_position() const override {
        return tail_.load(std::memory_order_acquire);
    }
    bool has_space(size_t size) const override;
    size_t capacity() const override {
        return buffer_size_;
    }

private:
    static size_t next_power_of_2(size_t n);

    uint8_t *slab_;
    size_t buffer_size_;
    size_t mask_;

    alignas(64) std::atomic<uint64_t> head_; // Multiple producers need atomic
    uint8_t padding1_[56];
    alignas(64) std::atomic<uint64_t> tail_; // Multiple consumers need atomic
    uint8_t padding2_[56];
};

} // namespace psyne