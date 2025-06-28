#pragma once

#include <chrono>
#include <functional>
#include <vector>
#include <random>
#include <cstdint>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace psyne {

// Forward declarations
class Channel;

// MessageID type definition 
using MessageID = uint64_t;

namespace reliability {

// Retry strategy types
enum class RetryStrategy : uint8_t {
    FixedDelay = 0,      // Fixed delay between retries
    ExponentialBackoff = 1, // Exponential backoff with jitter
    LinearBackoff = 2,   // Linear increase in delay
    Custom = 3           // Custom retry strategy
};

// Retry configuration
struct RetryConfig {
    RetryStrategy strategy = RetryStrategy::ExponentialBackoff;
    size_t max_attempts = 3;
    std::chrono::milliseconds initial_delay = std::chrono::milliseconds(100);
    std::chrono::milliseconds max_delay = std::chrono::seconds(30);
    double backoff_multiplier = 2.0;
    double jitter_factor = 0.1; // 10% jitter
    bool jitter_enabled = true;
    
    // Custom retry function (for RetryStrategy::Custom)
    std::function<std::chrono::milliseconds(size_t attempt, std::chrono::milliseconds base_delay)> custom_delay_func;
};

// Retry context for tracking retry state
struct RetryContext {
    MessageID message_id;
    size_t attempt_count = 0;
    std::chrono::steady_clock::time_point last_attempt;
    std::chrono::steady_clock::time_point next_retry_time;
    RetryConfig config;
    std::function<bool()> retry_func; // Function to call for retry
    std::function<void(bool success, size_t attempts)> completion_callback;
    bool completed = false;
};

// Circuit breaker states for integration with retry mechanism
enum class CircuitState : uint8_t {
    Closed = 0,   // Normal operation
    Open = 1,     // Circuit is open, requests fail fast
    HalfOpen = 2  // Testing if service has recovered
};

// Retry manager with circuit breaker integration
class RetryManager {
public:
    RetryManager();
    ~RetryManager();
    
    // Configuration
    void set_default_config(const RetryConfig& config);
    const RetryConfig& get_default_config() const { return default_config_; }
    
    // Retry operations
    template<typename MessageType>
    void retry_message_send(Channel& channel, MessageType& message, 
                           const RetryConfig& config = {});
    
    // Generic retry with custom function
    MessageID schedule_retry(std::function<bool()> retry_func,
                           const RetryConfig& config = {},
                           std::function<void(bool, size_t)> completion_callback = nullptr);
    
    // Retry management
    void process_retries();
    void cancel_retry(MessageID msg_id);
    void cancel_all_retries();
    
    // Status and statistics
    size_t pending_retry_count() const;
    bool is_retry_pending(MessageID msg_id) const;
    
    struct RetryStats {
        std::atomic<uint64_t> total_retries{0};
        std::atomic<uint64_t> successful_retries{0};
        std::atomic<uint64_t> failed_retries{0};
        std::atomic<uint64_t> max_retries_exceeded{0};
        std::atomic<uint64_t> circuit_breaker_open{0};
    };
    
    const RetryStats& get_stats() const { return stats_; }
    void reset_stats();
    
    // Circuit breaker integration
    void set_circuit_breaker_enabled(bool enabled);
    void set_circuit_breaker_config(size_t failure_threshold, 
                                   std::chrono::milliseconds timeout,
                                   size_t half_open_max_calls = 1);
    CircuitState get_circuit_state() const;
    
    // Control
    void start();
    void stop();

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    
    RetryConfig default_config_;
    std::unordered_map<MessageID, RetryContext> retry_contexts_;
    std::atomic<MessageID> next_retry_id_;
    
    // Background processing
    std::thread retry_thread_;
    std::atomic<bool> running_;
    
    // Statistics
    RetryStats stats_;
    
    // Circuit breaker state
    CircuitState circuit_state_;
    size_t circuit_failure_count_;
    size_t circuit_failure_threshold_;
    std::chrono::milliseconds circuit_timeout_;
    size_t circuit_half_open_max_calls_;
    size_t circuit_half_open_calls_;
    std::chrono::steady_clock::time_point circuit_opened_time_;
    bool circuit_breaker_enabled_;
    
    // Random number generation for jitter
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> jitter_dist_;
    
    // Helper methods
    std::chrono::milliseconds calculate_delay(const RetryContext& context) const;
    void retry_worker();
    bool should_circuit_break() const;
    void update_circuit_breaker(bool success);
    void transition_circuit_state(CircuitState new_state);
};

// RAII wrapper for automatic retry of message operations
template<typename MessageType>
class RetriableMessage {
public:
    RetriableMessage(Channel& channel, RetryManager& retry_mgr,
                    const RetryConfig& config = {})
        : message_(channel), retry_mgr_(retry_mgr), config_(config), sent_(false) {}
    
    MessageType& message() { return message_; }
    const MessageType& message() const { return message_; }
    
    // Send with automatic retry on failure
    void send_with_retry(std::function<void(bool success, size_t attempts)> callback = nullptr) {
        if (sent_) {
            throw std::runtime_error("Message already sent");
        }
        
        auto retry_func = [this]() -> bool {
            try {
                message_.send();
                return true;
            } catch (const std::exception&) {
                return false;
            }
        };
        
        retry_mgr_.schedule_retry(retry_func, config_, callback);
        sent_ = true;
    }
    
    // Send without retry (normal send)
    void send() {
        message_.send();
        sent_ = true;
    }

private:
    MessageType message_;
    RetryManager& retry_mgr_;
    RetryConfig config_;
    bool sent_;
};

// Factory functions
template<typename MessageType>
RetriableMessage<MessageType> create_retriable_message(
    Channel& channel, RetryManager& retry_mgr, const RetryConfig& config = {}) {
    return RetriableMessage<MessageType>(channel, retry_mgr, config);
}

// Utility functions for common retry configurations
RetryConfig create_exponential_backoff_config(
    size_t max_attempts = 3,
    std::chrono::milliseconds initial_delay = std::chrono::milliseconds(100),
    double multiplier = 2.0,
    std::chrono::milliseconds max_delay = std::chrono::seconds(30));

RetryConfig create_fixed_delay_config(
    size_t max_attempts = 3,
    std::chrono::milliseconds delay = std::chrono::milliseconds(500));

RetryConfig create_linear_backoff_config(
    size_t max_attempts = 3,
    std::chrono::milliseconds initial_delay = std::chrono::milliseconds(100),
    std::chrono::milliseconds increment = std::chrono::milliseconds(100));

} // namespace reliability
} // namespace psyne