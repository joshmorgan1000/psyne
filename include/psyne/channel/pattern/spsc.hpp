#pragma once

#include <psyne/global/logger.hpp>
#include "pattern_base.hpp"
#include <atomic>
#include <vector>

namespace psyne::pattern {

/**
 * @brief Single Producer Single Consumer pattern - lock-free ring buffer
 */
template <typename T, typename Substrate>
class SPSC : public PatternBase<T, Substrate> {
private:
    static constexpr size_t CACHE_LINE_SIZE = 64;

public:
    explicit SPSC(size_t ring_size = 1024) : ring_size_(ring_size) {
        if ((ring_size & (ring_size - 1)) != 0) {
            throw std::invalid_argument("Ring size must be power of 2");
        }
        ring_mask_ = ring_size - 1;
        ring_.resize(ring_size);

        log_info("SPSC pattern created with ring size ", ring_size);
    }

    T *try_allocate(T *slab, size_t max_messages) override {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next_head = head + 1;

        // Acquire fence to ensure we see the latest tail value
        size_t cached_tail = tail_.load(std::memory_order_acquire);

        if (next_head - cached_tail > ring_size_) {
            return nullptr; // Ring full
        }

        // Bounds check to prevent overflow
        if (head >= max_messages * ring_size_) {
            // Reset counters to prevent integer overflow
            size_t current_size = head - cached_tail;
            head_.store(current_size, std::memory_order_relaxed);
            tail_.store(0, std::memory_order_relaxed);
            head = current_size;
            next_head = head + 1;
        }

        size_t slab_pos = head % max_messages;
        T *slot = &slab[slab_pos];

        // Store in ring for consumer
        ring_[head & ring_mask_].store(slot, std::memory_order_relaxed);
        head_.store(next_head, std::memory_order_release);

        return slot;
    }

    T *try_receive() override {
        size_t tail = tail_.load(std::memory_order_relaxed);

        // Acquire fence to ensure we see the latest head value
        size_t cached_head = head_.load(std::memory_order_acquire);

        if (tail >= cached_head) {
            return nullptr; // Ring empty
        }

        // Load with acquire to ensure we see the complete message
        T *msg = ring_[tail & ring_mask_].load(std::memory_order_acquire);
        if (!msg) {
            // This should not happen in correct SPSC usage
            return nullptr;
        }

        // Clear the slot to help detect use-after-free
        ring_[tail & ring_mask_].store(nullptr, std::memory_order_relaxed);

        // Release fence to ensure the slot is cleared before advancing tail
        tail_.store(tail + 1, std::memory_order_release);

        return msg;
    }

    boost::asio::awaitable<T *>
    async_receive(boost::asio::io_context &io_context,
                  std::chrono::milliseconds timeout =
                      std::chrono::milliseconds(1000)) override {
        auto deadline = std::chrono::steady_clock::now() + timeout;

        while (std::chrono::steady_clock::now() < deadline) {
            T *msg = try_receive();
            if (msg) {
                co_return msg;
            }

            // Short wait before retry
            boost::asio::steady_timer timer(io_context);
            timer.expires_after(std::chrono::milliseconds(1));
            co_await timer.async_wait(boost::asio::use_awaitable);
        }

        co_return nullptr; // Timeout
    }

    T *receive_blocking(std::chrono::milliseconds timeout =
                            std::chrono::milliseconds(1000)) override {
        auto deadline = std::chrono::steady_clock::now() + timeout;

        while (std::chrono::steady_clock::now() < deadline) {
            T *msg = try_receive();
            if (msg) {
                return msg;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        return nullptr; // Timeout
    }

    bool needs_locks() const override {
        return false;
    }
    size_t max_producers() const override {
        return 1;
    }
    size_t max_consumers() const override {
        return 1;
    }
    const char *name() const override {
        return "SPSC";
    }

    size_t size() const override {
        return head_.load(std::memory_order_acquire) -
               tail_.load(std::memory_order_acquire);
    }

    bool empty() const override {
        return size() == 0;
    }

    bool full() const override {
        return size() >= ring_size_;
    }

private:
    size_t ring_size_;
    size_t ring_mask_;
    std::vector<std::atomic<T *>> ring_;

    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_{0};
};

} // namespace psyne::pattern