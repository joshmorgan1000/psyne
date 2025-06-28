#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>

namespace psyne {
namespace detail {

// Slab header for messages
struct SlabHeader {
    uint32_t len;
    uint32_t reserved;

    void *data() {
        return reinterpret_cast<uint8_t *>(this) + sizeof(SlabHeader);
    }
    const void *data() const {
        return reinterpret_cast<const uint8_t *>(this) + sizeof(SlabHeader);
    }
};

// Base ring buffer interface
class RingBufferBase {
public:
    virtual ~RingBufferBase() = default;

    struct WriteHandle {
        SlabHeader *header;
        void *data;
        size_t size;

        WriteHandle(SlabHeader *h, size_t s)
            : header(h), data(h->data()), size(s) {}
        void commit() {
            header->len = static_cast<uint32_t>(size);
        }
    };

    struct ReadHandle {
        const SlabHeader *header;
        const void *data;
        size_t size;

        ReadHandle(const SlabHeader *h)
            : header(h), data(h->data()), size(h->len) {}
    };

    virtual std::optional<WriteHandle> reserve(size_t size) = 0;
    virtual std::optional<ReadHandle> read() = 0;
    virtual bool empty() const = 0;
    virtual size_t capacity() const = 0;
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