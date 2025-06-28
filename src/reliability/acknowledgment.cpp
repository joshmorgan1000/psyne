#include "../../include/psyne/reliability/acknowledgment.hpp"
#include <chrono>
#include <thread>
#include <algorithm>

namespace psyne {
namespace reliability {

AcknowledgmentManager::AcknowledgmentManager() 
    : next_msg_id_(1)
    , default_timeout_(std::chrono::seconds(30))
    , default_max_retries_(3)
    , cleanup_interval_(std::chrono::minutes(5))
    , running_(true) {
    
    // Start background timeout checking thread
    timeout_thread_ = std::thread(&AcknowledgmentManager::timeout_worker, this);
}

AcknowledgmentManager::~AcknowledgmentManager() {
    stop();
}

void AcknowledgmentManager::stop() {
    running_ = false;
    cv_.notify_all();
    
    if (timeout_thread_.joinable()) {
        timeout_thread_.join();
    }
}

void AcknowledgmentManager::set_default_timeout(std::chrono::milliseconds timeout) {
    std::lock_guard<std::mutex> lock(mutex_);
    default_timeout_ = timeout;
}

void AcknowledgmentManager::set_default_max_retries(size_t max_retries) {
    std::lock_guard<std::mutex> lock(mutex_);
    default_max_retries_ = max_retries;
}

void AcknowledgmentManager::set_cleanup_interval(std::chrono::milliseconds interval) {
    std::lock_guard<std::mutex> lock(mutex_);
    cleanup_interval_ = interval;
}

MessageID AcknowledgmentManager::generate_message_id() {
    return next_msg_id_.fetch_add(1, std::memory_order_acq_rel);
}

void AcknowledgmentManager::track_message(MessageID msg_id, AckType type, 
                                        std::chrono::milliseconds timeout,
                                        std::function<void(MessageID, AckStatus)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (timeout == std::chrono::milliseconds::zero()) {
        timeout = default_timeout_;
    }
    
    AckRequest request;
    request.msg_id = msg_id;
    request.type = type;
    request.sent_time = std::chrono::steady_clock::now();
    request.timeout = timeout;
    request.callback = std::move(callback);
    request.retry_count = 0;
    request.max_retries = default_max_retries_;
    
    pending_acks_[msg_id] = std::move(request);
    stats_.messages_sent.fetch_add(1, std::memory_order_relaxed);
    
    cv_.notify_one(); // Wake up timeout thread
}

void AcknowledgmentManager::process_acknowledgment(const AckMessage& ack) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = pending_acks_.find(ack.original_msg_id);
    if (it == pending_acks_.end()) {
        // Acknowledgment for unknown message (might be duplicate or late)
        return;
    }
    
    AckRequest& request = it->second;
    
    // Update statistics
    switch (ack.status) {
        case AckStatus::Acknowledged:
            stats_.messages_acknowledged.fetch_add(1, std::memory_order_relaxed);
            break;
        case AckStatus::Failed:
            stats_.messages_failed.fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            break;
    }
    
    // Call callback if provided
    if (request.callback) {
        try {
            request.callback(ack.original_msg_id, ack.status);
        } catch (const std::exception& e) {
            // Log error but don't let callback exceptions break the manager
            // In a real implementation, this would use proper logging
        }
    }
    
    // Move to completed and remove from pending
    completed_acks_[ack.original_msg_id] = ack.status;
    pending_acks_.erase(it);
}

void AcknowledgmentManager::send_acknowledgment(Channel& channel, MessageID msg_id, 
                                              AckStatus status, uint32_t error_code) {
    // Create acknowledgment message
    // This is a simplified implementation - in practice would need proper message creation
    AckMessage ack;
    ack.original_msg_id = msg_id;
    ack.status = status;
    ack.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    ack.error_code = error_code;
    
    // TODO: Send acknowledgment through channel
    // This would require extending the Message system to support AckMessage
    // For now, this is a placeholder for the interface
}

void AcknowledgmentManager::check_timeouts() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto now = std::chrono::steady_clock::now();
    std::vector<MessageID> timed_out_messages;
    
    for (auto& [msg_id, request] : pending_acks_) {
        auto elapsed = now - request.sent_time;
        
        if (elapsed >= request.timeout) {
            timed_out_messages.push_back(msg_id);
        }
    }
    
    // Process timeouts
    for (MessageID msg_id : timed_out_messages) {
        auto it = pending_acks_.find(msg_id);
        if (it == pending_acks_.end()) continue;
        
        AckRequest& request = it->second;
        
        if (request.retry_count < request.max_retries) {
            // Retry the message
            request.retry_count++;
            request.sent_time = now; // Reset timeout
            stats_.total_retries.fetch_add(1, std::memory_order_relaxed);
            
            // Call callback for retry notification
            if (request.callback) {
                try {
                    request.callback(msg_id, AckStatus::Pending);
                } catch (...) {
                    // Ignore callback exceptions
                }
            }
        } else {
            // Max retries exceeded, mark as timeout
            stats_.messages_timeout.fetch_add(1, std::memory_order_relaxed);
            
            if (request.callback) {
                try {
                    request.callback(msg_id, AckStatus::Timeout);
                } catch (...) {
                    // Ignore callback exceptions
                }
            }
            
            completed_acks_[msg_id] = AckStatus::Timeout;
            pending_acks_.erase(it);
        }
    }
}

void AcknowledgmentManager::retry_message(MessageID msg_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = pending_acks_.find(msg_id);
    if (it == pending_acks_.end()) {
        return; // Message not found or already completed
    }
    
    AckRequest& request = it->second;
    
    if (request.retry_count < request.max_retries) {
        request.retry_count++;
        request.sent_time = std::chrono::steady_clock::now();
        stats_.total_retries.fetch_add(1, std::memory_order_relaxed);
    }
}

AckStatus AcknowledgmentManager::get_status(MessageID msg_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check pending messages
    auto pending_it = pending_acks_.find(msg_id);
    if (pending_it != pending_acks_.end()) {
        return AckStatus::Pending;
    }
    
    // Check completed messages
    auto completed_it = completed_acks_.find(msg_id);
    if (completed_it != completed_acks_.end()) {
        return completed_it->second;
    }
    
    // Message not found
    return AckStatus::Failed;
}

bool AcknowledgmentManager::is_acknowledged(MessageID msg_id) const {
    return get_status(msg_id) == AckStatus::Acknowledged;
}

size_t AcknowledgmentManager::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_acks_.size();
}

void AcknowledgmentManager::reset_stats() {
    stats_.messages_sent.store(0, std::memory_order_relaxed);
    stats_.messages_acknowledged.store(0, std::memory_order_relaxed);
    stats_.messages_failed.store(0, std::memory_order_relaxed);
    stats_.messages_timeout.store(0, std::memory_order_relaxed);
    stats_.total_retries.store(0, std::memory_order_relaxed);
}

void AcknowledgmentManager::cleanup_completed() {
    std::lock_guard<std::mutex> lock(mutex_);
    cleanup_old_entries();
}

void AcknowledgmentManager::cleanup_old_entries() {
    // Remove completed acknowledgments older than cleanup interval
    auto cutoff_time = std::chrono::steady_clock::now() - cleanup_interval_;
    
    auto it = completed_acks_.begin();
    while (it != completed_acks_.end()) {
        // In a full implementation, we'd track completion time
        // For now, just limit the size of completed_acks_
        if (completed_acks_.size() > 10000) { // Arbitrary limit
            it = completed_acks_.erase(it);
        } else {
            ++it;
        }
    }
}

void AcknowledgmentManager::timeout_worker() {
    auto last_cleanup = std::chrono::steady_clock::now();
    
    while (running_) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait for a short interval or until notified
        cv_.wait_for(lock, std::chrono::seconds(1), [this] { return !running_; });
        
        if (!running_) break;
        
        lock.unlock();
        
        // Check for timeouts
        check_timeouts();
        
        // Periodic cleanup
        auto now = std::chrono::steady_clock::now();
        if (now - last_cleanup >= cleanup_interval_) {
            lock.lock();
            cleanup_old_entries();
            lock.unlock();
            last_cleanup = now;
        }
    }
}

} // namespace reliability
} // namespace psyne