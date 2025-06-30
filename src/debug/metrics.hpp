#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include "../utils/logger.hpp"

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
        log_trace("Collecting current channel metrics");
        ChannelMetrics result;
        result.messages_sent = messages_sent.load(std::memory_order_relaxed);
        result.bytes_sent = bytes_sent.load(std::memory_order_relaxed);
        result.messages_received =
            messages_received.load(std::memory_order_relaxed);
        result.bytes_received = bytes_received.load(std::memory_order_relaxed);
        result.send_blocks = send_blocks.load(std::memory_order_relaxed);
        result.receive_blocks = receive_blocks.load(std::memory_order_relaxed);
        
        log_debug("Metrics collected - sent: ", result.messages_sent, " msgs (", result.bytes_sent, " bytes), ",
                 "received: ", result.messages_received, " msgs (", result.bytes_received, " bytes), ",
                 "blocks: ", result.send_blocks, "/", result.receive_blocks);
        return result;
    }
    
    /**
     * @brief Update send metrics with logging
     */
    void record_send(uint64_t message_size) {
        messages_sent.fetch_add(1, std::memory_order_relaxed);
        bytes_sent.fetch_add(message_size, std::memory_order_relaxed);
        log_trace("Recorded send: size=", message_size, ", total_msgs=", messages_sent.load(), ", total_bytes=", bytes_sent.load());
    }
    
    /**
     * @brief Update receive metrics with logging
     */
    void record_receive(uint64_t message_size) {
        messages_received.fetch_add(1, std::memory_order_relaxed);
        bytes_received.fetch_add(message_size, std::memory_order_relaxed);
        log_trace("Recorded receive: size=", message_size, ", total_msgs=", messages_received.load(), ", total_bytes=", bytes_received.load());
    }
    
    /**
     * @brief Record a send block event
     */
    void record_send_block() {
        auto count = send_blocks.fetch_add(1, std::memory_order_relaxed) + 1;
        log_warn("Send blocked (buffer full) - total send blocks: ", count);
    }
    
    /**
     * @brief Record a receive block event
     */
    void record_receive_block() {
        auto count = receive_blocks.fetch_add(1, std::memory_order_relaxed) + 1;
        log_debug("Receive blocked (no data) - total receive blocks: ", count);
    }
};

} // namespace debug
} // namespace psyne