#include "../../include/psyne/reliability/retry.hpp"
#include <algorithm>
#include <cmath>

namespace psyne {
namespace reliability {

RetryManager::RetryManager() 
    : next_retry_id_(1)
    , running_(false)
    , circuit_state_(CircuitState::Closed)
    , circuit_failure_count_(0)
    , circuit_failure_threshold_(5)
    , circuit_timeout_(std::chrono::seconds(60))
    , circuit_half_open_max_calls_(1)
    , circuit_half_open_calls_(0)
    , circuit_breaker_enabled_(true)
    , rng_(std::random_device{}())
    , jitter_dist_(-1.0, 1.0) {
    
    // Set default retry configuration
    default_config_.strategy = RetryStrategy::ExponentialBackoff;
    default_config_.max_attempts = 3;
    default_config_.initial_delay = std::chrono::milliseconds(100);
    default_config_.max_delay = std::chrono::seconds(30);
    default_config_.backoff_multiplier = 2.0;
    default_config_.jitter_factor = 0.1;
    default_config_.jitter_enabled = true;
}

RetryManager::~RetryManager() {
    stop();
}

void RetryManager::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        running_ = true;
        retry_thread_ = std::thread(&RetryManager::retry_worker, this);
    }
}

void RetryManager::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
    }
    cv_.notify_all();
    
    if (retry_thread_.joinable()) {
        retry_thread_.join();
    }
}

void RetryManager::set_default_config(const RetryConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    default_config_ = config;
}

MessageID RetryManager::schedule_retry(std::function<bool()> retry_func,
                                     const RetryConfig& config,
                                     std::function<void(bool, size_t)> completion_callback) {
    MessageID retry_id = next_retry_id_.fetch_add(1, std::memory_order_acq_rel);
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    RetryContext context;
    context.message_id = retry_id;
    context.attempt_count = 0;
    context.last_attempt = std::chrono::steady_clock::time_point{};
    context.next_retry_time = std::chrono::steady_clock::now();
    context.config = config.max_attempts > 0 ? config : default_config_;
    context.retry_func = std::move(retry_func);
    context.completion_callback = std::move(completion_callback);
    context.completed = false;
    
    retry_contexts_[retry_id] = std::move(context);
    cv_.notify_one();
    
    return retry_id;
}

void RetryManager::process_retries() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    
    std::vector<MessageID> to_process;
    std::vector<MessageID> to_complete;
    
    for (auto& [msg_id, context] : retry_contexts_) {
        if (context.completed) {
            to_complete.push_back(msg_id);
            continue;
        }
        
        if (now >= context.next_retry_time) {
            to_process.push_back(msg_id);
        }
    }
    
    // Clean up completed retries
    for (MessageID msg_id : to_complete) {
        retry_contexts_.erase(msg_id);
    }
    
    // Process pending retries
    for (MessageID msg_id : to_process) {
        auto it = retry_contexts_.find(msg_id);
        if (it == retry_contexts_.end()) continue;
        
        RetryContext& context = it->second;
        
        // Check circuit breaker
        if (circuit_breaker_enabled_ && should_circuit_break()) {
            stats_.circuit_breaker_open.fetch_add(1, std::memory_order_relaxed);
            context.completed = true;
            if (context.completion_callback) {
                lock.unlock();
                context.completion_callback(false, context.attempt_count);
                lock.lock();
            }
            continue;
        }
        
        context.attempt_count++;
        context.last_attempt = now;
        stats_.total_retries.fetch_add(1, std::memory_order_relaxed);
        
        bool success = false;
        if (context.retry_func) {
            lock.unlock();
            try {
                success = context.retry_func();
            } catch (...) {
                success = false;
            }
            lock.lock();
        }
        
        // Update circuit breaker
        if (circuit_breaker_enabled_) {
            update_circuit_breaker(success);
        }
        
        if (success) {
            stats_.successful_retries.fetch_add(1, std::memory_order_relaxed);
            context.completed = true;
            if (context.completion_callback) {
                lock.unlock();
                context.completion_callback(true, context.attempt_count);
                lock.lock();
            }
        } else if (context.attempt_count >= context.config.max_attempts) {
            stats_.max_retries_exceeded.fetch_add(1, std::memory_order_relaxed);
            stats_.failed_retries.fetch_add(1, std::memory_order_relaxed);
            context.completed = true;
            if (context.completion_callback) {
                lock.unlock();
                context.completion_callback(false, context.attempt_count);
                lock.lock();
            }
        } else {
            // Schedule next retry
            auto delay = calculate_delay(context);
            context.next_retry_time = now + delay;
        }
    }
}

void RetryManager::cancel_retry(MessageID msg_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = retry_contexts_.find(msg_id);
    if (it != retry_contexts_.end()) {
        it->second.completed = true;
    }
}

void RetryManager::cancel_all_retries() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [msg_id, context] : retry_contexts_) {
        context.completed = true;
    }
}

size_t RetryManager::pending_retry_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& [msg_id, context] : retry_contexts_) {
        if (!context.completed) {
            count++;
        }
    }
    return count;
}

bool RetryManager::is_retry_pending(MessageID msg_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = retry_contexts_.find(msg_id);
    return it != retry_contexts_.end() && !it->second.completed;
}

void RetryManager::reset_stats() {
    stats_.total_retries.store(0, std::memory_order_relaxed);
    stats_.successful_retries.store(0, std::memory_order_relaxed);
    stats_.failed_retries.store(0, std::memory_order_relaxed);
    stats_.max_retries_exceeded.store(0, std::memory_order_relaxed);
    stats_.circuit_breaker_open.store(0, std::memory_order_relaxed);
}

void RetryManager::set_circuit_breaker_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    circuit_breaker_enabled_ = enabled;
    if (!enabled) {
        transition_circuit_state(CircuitState::Closed);
    }
}

void RetryManager::set_circuit_breaker_config(size_t failure_threshold,
                                             std::chrono::milliseconds timeout,
                                             size_t half_open_max_calls) {
    std::lock_guard<std::mutex> lock(mutex_);
    circuit_failure_threshold_ = failure_threshold;
    circuit_timeout_ = timeout;
    circuit_half_open_max_calls_ = half_open_max_calls;
}

CircuitState RetryManager::get_circuit_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return circuit_state_;
}

std::chrono::milliseconds RetryManager::calculate_delay(const RetryContext& context) const {
    auto delay = context.config.initial_delay;
    
    switch (context.config.strategy) {
        case RetryStrategy::FixedDelay:
            delay = context.config.initial_delay;
            break;
            
        case RetryStrategy::ExponentialBackoff: {
            double multiplier = std::pow(context.config.backoff_multiplier, context.attempt_count - 1);
            delay = std::chrono::milliseconds(static_cast<long long>(
                context.config.initial_delay.count() * multiplier));
            break;
        }
        
        case RetryStrategy::LinearBackoff:
            delay = std::chrono::milliseconds(
                context.config.initial_delay.count() * context.attempt_count);
            break;
            
        case RetryStrategy::Custom:
            if (context.config.custom_delay_func) {
                delay = context.config.custom_delay_func(context.attempt_count, context.config.initial_delay);
            }
            break;
    }
    
    // Apply maximum delay limit
    delay = std::min(delay, context.config.max_delay);
    
    // Apply jitter if enabled
    if (context.config.jitter_enabled && context.config.jitter_factor > 0.0) {
        double jitter = jitter_dist_(rng_) * context.config.jitter_factor;
        long long jitter_ms = static_cast<long long>(delay.count() * jitter);
        delay += std::chrono::milliseconds(jitter_ms);
        
        // Ensure delay doesn't go negative
        delay = std::max(delay, std::chrono::milliseconds(0));
    }
    
    return delay;
}

void RetryManager::retry_worker() {
    while (running_) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Find next retry time
        auto next_retry = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        bool has_pending = false;
        
        for (const auto& [msg_id, context] : retry_contexts_) {
            if (!context.completed && context.next_retry_time < next_retry) {
                next_retry = context.next_retry_time;
                has_pending = true;
            }
        }
        
        if (has_pending) {
            cv_.wait_until(lock, next_retry, [this] { return !running_; });
        } else {
            cv_.wait_for(lock, std::chrono::seconds(1), [this] { return !running_; });
        }
        
        if (!running_) break;
        
        lock.unlock();
        process_retries();
    }
}

bool RetryManager::should_circuit_break() const {
    switch (circuit_state_) {
        case CircuitState::Open: {
            auto now = std::chrono::steady_clock::now();
            if (now - circuit_opened_time_ >= circuit_timeout_) {
                const_cast<RetryManager*>(this)->transition_circuit_state(CircuitState::HalfOpen);
                return false;
            }
            return true;
        }
        case CircuitState::HalfOpen:
            return circuit_half_open_calls_ >= circuit_half_open_max_calls_;
        case CircuitState::Closed:
        default:
            return false;
    }
}

void RetryManager::update_circuit_breaker(bool success) {
    switch (circuit_state_) {
        case CircuitState::Closed:
            if (!success) {
                circuit_failure_count_++;
                if (circuit_failure_count_ >= circuit_failure_threshold_) {
                    transition_circuit_state(CircuitState::Open);
                }
            } else {
                circuit_failure_count_ = 0; // Reset on success
            }
            break;
            
        case CircuitState::HalfOpen:
            circuit_half_open_calls_++;
            if (success) {
                transition_circuit_state(CircuitState::Closed);
            } else {
                transition_circuit_state(CircuitState::Open);
            }
            break;
            
        case CircuitState::Open:
            // No updates in open state
            break;
    }
}

void RetryManager::transition_circuit_state(CircuitState new_state) {
    circuit_state_ = new_state;
    
    switch (new_state) {
        case CircuitState::Open:
            circuit_opened_time_ = std::chrono::steady_clock::now();
            break;
        case CircuitState::HalfOpen:
            circuit_half_open_calls_ = 0;
            break;
        case CircuitState::Closed:
            circuit_failure_count_ = 0;
            circuit_half_open_calls_ = 0;
            break;
    }
}

// Utility function implementations
RetryConfig create_exponential_backoff_config(size_t max_attempts,
                                             std::chrono::milliseconds initial_delay,
                                             double multiplier,
                                             std::chrono::milliseconds max_delay) {
    RetryConfig config;
    config.strategy = RetryStrategy::ExponentialBackoff;
    config.max_attempts = max_attempts;
    config.initial_delay = initial_delay;
    config.backoff_multiplier = multiplier;
    config.max_delay = max_delay;
    config.jitter_enabled = true;
    config.jitter_factor = 0.1;
    return config;
}

RetryConfig create_fixed_delay_config(size_t max_attempts, std::chrono::milliseconds delay) {
    RetryConfig config;
    config.strategy = RetryStrategy::FixedDelay;
    config.max_attempts = max_attempts;
    config.initial_delay = delay;
    config.max_delay = delay;
    config.jitter_enabled = false;
    return config;
}

RetryConfig create_linear_backoff_config(size_t max_attempts,
                                       std::chrono::milliseconds initial_delay,
                                       std::chrono::milliseconds increment) {
    RetryConfig config;
    config.strategy = RetryStrategy::LinearBackoff;
    config.max_attempts = max_attempts;
    config.initial_delay = initial_delay;
    config.max_delay = initial_delay * max_attempts;
    config.jitter_enabled = true;
    config.jitter_factor = 0.05; // Less jitter for linear backoff
    return config;
}

} // namespace reliability
} // namespace psyne