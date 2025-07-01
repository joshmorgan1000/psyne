#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>

/**
 * @file backpressure.hpp
 * @brief Backpressure policies for Psyne channels
 *
 * Provides different strategies for handling channel capacity limits:
 * - Drop: Drop new messages when full
 * - Block: Block producer until space available
 * - Retry: Retry with exponential backoff
 * - Callback: Notify application when pressure detected
 */

namespace psyne::backpressure {

/**
 * @brief Base backpressure policy interface
 */
class PolicyBase {
public:
    virtual ~PolicyBase() = default;

    /**
     * @brief Handle allocation failure due to full channel
     * @param retry_fn Function to retry allocation
     * @return Allocated memory or nullptr if policy decides to drop
     */
    virtual void *handle_full(std::function<void *()> retry_fn) = 0;

    /**
     * @brief Get policy name
     */
    virtual const char *name() const = 0;

    /**
     * @brief Record metrics about backpressure events
     */
    virtual void record_pressure_event() {
        pressure_events_.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Get total pressure events
     */
    size_t get_pressure_events() const {
        return pressure_events_.load(std::memory_order_relaxed);
    }

protected:
    std::atomic<size_t> pressure_events_{0};
};

/**
 * @brief Drop policy - drop messages when channel is full
 */
class DropPolicy : public PolicyBase {
public:
    void *handle_full(std::function<void *()> retry_fn) override {
        record_pressure_event();
        dropped_messages_.fetch_add(1, std::memory_order_relaxed);
        return nullptr; // Drop the message
    }

    const char *name() const override {
        return "Drop";
    }

    /**
     * @brief Get number of dropped messages
     */
    size_t get_dropped_messages() const {
        return dropped_messages_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<size_t> dropped_messages_{0};
};

/**
 * @brief Block policy - block producer until space is available
 */
class BlockPolicy : public PolicyBase {
public:
    explicit BlockPolicy(
        std::chrono::milliseconds max_wait = std::chrono::milliseconds(1000))
        : max_wait_(max_wait) {}

    void *handle_full(std::function<void *()> retry_fn) override {
        record_pressure_event();

        auto start = std::chrono::steady_clock::now();

        while (true) {
            void *result = retry_fn();
            if (result) {
                return result;
            }

            auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed >= max_wait_) {
                timeout_count_.fetch_add(1, std::memory_order_relaxed);
                return nullptr; // Timeout
            }

            // Yield to avoid spinning
            std::this_thread::yield();
        }
    }

    const char *name() const override {
        return "Block";
    }

    /**
     * @brief Get number of timeouts
     */
    size_t get_timeout_count() const {
        return timeout_count_.load(std::memory_order_relaxed);
    }

private:
    std::chrono::milliseconds max_wait_;
    std::atomic<size_t> timeout_count_{0};
};

/**
 * @brief Retry policy - exponential backoff retry
 */
class RetryPolicy : public PolicyBase {
public:
    explicit RetryPolicy(
        size_t max_retries = 10,
        std::chrono::microseconds initial_delay = std::chrono::microseconds(10))
        : max_retries_(max_retries), initial_delay_(initial_delay) {}

    void *handle_full(std::function<void *()> retry_fn) override {
        record_pressure_event();

        auto delay = initial_delay_;

        for (size_t i = 0; i < max_retries_; ++i) {
            void *result = retry_fn();
            if (result) {
                return result;
            }

            retry_count_.fetch_add(1, std::memory_order_relaxed);

            // Exponential backoff with jitter
            std::this_thread::sleep_for(delay);
            delay = delay * 2;

            // Add jitter (0-25% of delay)
            auto jitter = std::chrono::duration_cast<std::chrono::microseconds>(
                delay * (rand() % 250) / 1000);
            delay += jitter;
        }

        failed_retries_.fetch_add(1, std::memory_order_relaxed);
        return nullptr; // All retries exhausted
    }

    const char *name() const override {
        return "Retry";
    }

    /**
     * @brief Get retry statistics
     */
    size_t get_retry_count() const {
        return retry_count_.load(std::memory_order_relaxed);
    }

    size_t get_failed_retries() const {
        return failed_retries_.load(std::memory_order_relaxed);
    }

private:
    size_t max_retries_;
    std::chrono::microseconds initial_delay_;
    std::atomic<size_t> retry_count_{0};
    std::atomic<size_t> failed_retries_{0};
};

/**
 * @brief Callback policy - notify application of backpressure
 */
class CallbackPolicy : public PolicyBase {
public:
    using PressureCallback = std::function<bool()>;

    explicit CallbackPolicy(
        PressureCallback callback,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(100))
        : callback_(callback), timeout_(timeout) {}

    void *handle_full(std::function<void *()> retry_fn) override {
        record_pressure_event();

        // Notify application
        bool should_retry = callback_();
        if (!should_retry) {
            rejected_count_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }

        // Wait for timeout then retry once
        auto deadline = std::chrono::steady_clock::now() + timeout_;

        while (std::chrono::steady_clock::now() < deadline) {
            void *result = retry_fn();
            if (result) {
                return result;
            }
            std::this_thread::yield();
        }

        timeout_count_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    const char *name() const override {
        return "Callback";
    }

    /**
     * @brief Get callback statistics
     */
    size_t get_rejected_count() const {
        return rejected_count_.load(std::memory_order_relaxed);
    }

    size_t get_timeout_count() const {
        return timeout_count_.load(std::memory_order_relaxed);
    }

private:
    PressureCallback callback_;
    std::chrono::milliseconds timeout_;
    std::atomic<size_t> rejected_count_{0};
    std::atomic<size_t> timeout_count_{0};
};

/**
 * @brief Adaptive policy - switch strategies based on pressure
 */
class AdaptivePolicy : public PolicyBase {
public:
    AdaptivePolicy() {
        // Start with retry, escalate to block, then drop
        retry_policy_ = std::make_unique<RetryPolicy>(3);
        block_policy_ =
            std::make_unique<BlockPolicy>(std::chrono::milliseconds(50));
        drop_policy_ = std::make_unique<DropPolicy>();
    }

    void *handle_full(std::function<void *()> retry_fn) override {
        record_pressure_event();

        size_t pressure = pressure_events_.load(std::memory_order_relaxed);

        // Adaptive strategy based on pressure level
        if (pressure < 100) {
            // Low pressure: retry with backoff
            return retry_policy_->handle_full(retry_fn);
        } else if (pressure < 1000) {
            // Medium pressure: brief blocking
            return block_policy_->handle_full(retry_fn);
        } else {
            // High pressure: drop messages
            return drop_policy_->handle_full(retry_fn);
        }
    }

    const char *name() const override {
        return "Adaptive";
    }

private:
    std::unique_ptr<RetryPolicy> retry_policy_;
    std::unique_ptr<BlockPolicy> block_policy_;
    std::unique_ptr<DropPolicy> drop_policy_;
};

} // namespace psyne::backpressure