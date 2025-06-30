#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace psyne {
namespace debug {

/**
 * @struct ChannelMetrics
 * @brief Lightweight performance metrics for channel debugging
 *
 * Tracks basic counters for message throughput and blocking behavior.
 * Designed to have minimal overhead in the hot path.
 */
struct ChannelMetrics {
    // Core counters - non-atomic for SPSC, atomic for multi-threaded channels
    uint64_t messages_sent = 0;     ///< Number of messages sent
    uint64_t bytes_sent = 0;        ///< Total bytes sent
    uint64_t messages_received = 0; ///< Number of messages received
    uint64_t bytes_received = 0;    ///< Total bytes received
    uint64_t send_blocks = 0;       ///< Times send() blocked waiting for space
    uint64_t receive_blocks = 0; ///< Times receive() blocked waiting for data

    ChannelMetrics() = default;
    ChannelMetrics(const ChannelMetrics &) = default;
    ChannelMetrics &operator=(const ChannelMetrics &) = default;
    ChannelMetrics(ChannelMetrics &&) = default;
    ChannelMetrics &operator=(ChannelMetrics &&) = default;
};

/**
 * @struct AtomicChannelMetrics
 * @brief Thread-safe version of ChannelMetrics for multi-threaded channels
 *
 * Uses atomic counters for thread safety in MPSC/SPMC/MPMC modes.
 */
struct AtomicChannelMetrics {
    std::atomic<uint64_t> messages_sent{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> messages_received{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> send_blocks{0};
    std::atomic<uint64_t> receive_blocks{0};

    AtomicChannelMetrics() = default;

    /**
     * @brief Get current metrics as non-atomic struct for calculations
     */
    ChannelMetrics current() const {
        ChannelMetrics result;
        result.messages_sent = messages_sent.load(std::memory_order_relaxed);
        result.bytes_sent = bytes_sent.load(std::memory_order_relaxed);
        result.messages_received =
            messages_received.load(std::memory_order_relaxed);
        result.bytes_received = bytes_received.load(std::memory_order_relaxed);
        result.send_blocks = send_blocks.load(std::memory_order_relaxed);
        result.receive_blocks = receive_blocks.load(std::memory_order_relaxed);
        return result;
    }
};

} // namespace debug
} // namespace psyne