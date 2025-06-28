#include "../../include/psyne/reliability/replay_buffer.hpp"
#include "../../include/psyne/psyne.hpp"
#include <algorithm>
#include <stdexcept>

namespace psyne {
namespace reliability {

ReplayBuffer::ReplayBuffer(const ReplayBufferConfig& config)
    : config_(config), running_(false), paused_(false), total_stored_bytes_(0) {}

ReplayBuffer::~ReplayBuffer() {
    stop();
}

void ReplayBuffer::set_config(const ReplayBufferConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    
    // If max_messages is smaller than current count, evict oldest
    if (stored_messages_.size() > config_.max_messages) {
        evict_oldest_messages(stored_messages_.size() - config_.max_messages);
    }
}

void ReplayBuffer::store_message(MessageID msg_id, const void* data, size_t size, uint32_t message_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if we need to evict messages to make room
    if (stored_messages_.size() >= config_.max_messages) {
        evict_oldest_messages(1);
    }
    
    // Create stored message
    StoredMessage stored_msg;
    stored_msg.message_id = msg_id;
    stored_msg.data.resize(size);
    std::memcpy(stored_msg.data.data(), data, size);
    stored_msg.message_type = message_type;
    stored_msg.timestamp = std::chrono::steady_clock::now();
    stored_msg.replay_count = 0;
    stored_msg.last_replay = std::chrono::steady_clock::time_point::min();
    
    // Store the message
    stored_messages_[msg_id] = std::move(stored_msg);
    update_message_order(msg_id);
    
    total_stored_bytes_ += size;
    stats_.messages_stored++;
}

void ReplayBuffer::store_message(MessageID msg_id, const std::vector<uint8_t>& data, uint32_t message_type) {
    store_message(msg_id, data.data(), data.size(), message_type);
}

bool ReplayBuffer::replay_message(MessageID msg_id, Channel& target_channel) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = stored_messages_.find(msg_id);
    if (it == stored_messages_.end()) {
        return false;
    }
    
    StoredMessage& stored_msg = it->second;
    
    // Check replay limits
    if (stored_msg.replay_count >= config_.max_replay_count) {
        stats_.replay_failures++;
        return false;
    }
    
    // Check replay delay
    auto now = std::chrono::steady_clock::now();
    if (now - stored_msg.last_replay < config_.replay_delay) {
        stats_.replay_failures++;
        return false;
    }
    
    bool success = send_stored_message(stored_msg, target_channel);
    
    if (success) {
        stored_msg.replay_count++;
        stored_msg.last_replay = now;
        update_message_order(msg_id);  // Move to end of LRU
        stats_.replay_successes++;
    } else {
        stats_.replay_failures++;
    }
    
    stats_.messages_replayed++;
    return success;
}

void ReplayBuffer::replay_message_async(MessageID msg_id, Channel& target_channel,
                                       std::function<void(bool success, const std::string& error)> callback) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ReplayRequest request;
        request.message_id = msg_id;
        request.target_channel = &target_channel;
        request.callback = callback;
        request.requested_at = std::chrono::steady_clock::now();
        
        replay_queue_.push_back(std::move(request));
    }
    replay_cv_.notify_one();
}

size_t ReplayBuffer::replay_range(std::chrono::steady_clock::time_point start_time,
                                 std::chrono::steady_clock::time_point end_time,
                                 Channel& target_channel) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<MessageID> messages_to_replay;
    
    for (const auto& [msg_id, stored_msg] : stored_messages_) {
        if (stored_msg.timestamp >= start_time && stored_msg.timestamp <= end_time) {
            messages_to_replay.push_back(msg_id);
        }
    }
    
    // Sort by timestamp
    std::sort(messages_to_replay.begin(), messages_to_replay.end(),
              [this](MessageID a, MessageID b) {
                  return stored_messages_[a].timestamp < stored_messages_[b].timestamp;
              });
    
    size_t replayed = 0;
    for (MessageID msg_id : messages_to_replay) {
        auto it = stored_messages_.find(msg_id);
        if (it != stored_messages_.end()) {
            StoredMessage& stored_msg = it->second;
            
            if (stored_msg.replay_count < config_.max_replay_count) {
                auto now = std::chrono::steady_clock::now();
                if (now - stored_msg.last_replay >= config_.replay_delay) {
                    if (send_stored_message(stored_msg, target_channel)) {
                        stored_msg.replay_count++;
                        stored_msg.last_replay = now;
                        replayed++;
                        stats_.replay_successes++;
                    } else {
                        stats_.replay_failures++;
                    }
                    stats_.messages_replayed++;
                }
            }
        }
    }
    
    return replayed;
}

size_t ReplayBuffer::replay_from_message(MessageID start_msg_id, Channel& target_channel) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find the starting message
    auto start_it = stored_messages_.find(start_msg_id);
    if (start_it == stored_messages_.end()) {
        return 0;
    }
    
    auto start_time = start_it->second.timestamp;
    
    std::vector<std::pair<MessageID, std::chrono::steady_clock::time_point>> messages_to_replay;
    
    for (const auto& [msg_id, stored_msg] : stored_messages_) {
        if (stored_msg.timestamp >= start_time && msg_id >= start_msg_id) {
            messages_to_replay.emplace_back(msg_id, stored_msg.timestamp);
        }
    }
    
    // Sort by timestamp, then by message ID
    std::sort(messages_to_replay.begin(), messages_to_replay.end(),
              [](const auto& a, const auto& b) {
                  if (a.second == b.second) {
                      return a.first < b.first;
                  }
                  return a.second < b.second;
              });
    
    size_t replayed = 0;
    for (const auto& [msg_id, timestamp] : messages_to_replay) {
        auto it = stored_messages_.find(msg_id);
        if (it != stored_messages_.end()) {
            StoredMessage& stored_msg = it->second;
            
            if (stored_msg.replay_count < config_.max_replay_count) {
                auto now = std::chrono::steady_clock::now();
                if (now - stored_msg.last_replay >= config_.replay_delay) {
                    if (send_stored_message(stored_msg, target_channel)) {
                        stored_msg.replay_count++;
                        stored_msg.last_replay = now;
                        replayed++;
                        stats_.replay_successes++;
                    } else {
                        stats_.replay_failures++;
                    }
                    stats_.messages_replayed++;
                }
            }
        }
    }
    
    return replayed;
}

bool ReplayBuffer::has_message(MessageID msg_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stored_messages_.find(msg_id) != stored_messages_.end();
}

bool ReplayBuffer::remove_message(MessageID msg_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = stored_messages_.find(msg_id);
    if (it == stored_messages_.end()) {
        return false;
    }
    
    total_stored_bytes_ -= it->second.data.size();
    stored_messages_.erase(it);
    
    // Remove from order tracking
    message_order_.erase(
        std::remove(message_order_.begin(), message_order_.end(), msg_id),
        message_order_.end());
    
    return true;
}

void ReplayBuffer::clear_all_messages() {
    std::lock_guard<std::mutex> lock(mutex_);
    stored_messages_.clear();
    message_order_.clear();
    total_stored_bytes_ = 0;
}

std::vector<MessageID> ReplayBuffer::get_stored_message_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<MessageID> ids;
    ids.reserve(stored_messages_.size());
    
    for (const auto& [msg_id, stored_msg] : stored_messages_) {
        ids.push_back(msg_id);
    }
    
    return ids;
}

std::vector<MessageID> ReplayBuffer::get_messages_in_range(
    std::chrono::steady_clock::time_point start_time,
    std::chrono::steady_clock::time_point end_time) const {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<MessageID> ids;
    
    for (const auto& [msg_id, stored_msg] : stored_messages_) {
        if (stored_msg.timestamp >= start_time && stored_msg.timestamp <= end_time) {
            ids.push_back(msg_id);
        }
    }
    
    return ids;
}

size_t ReplayBuffer::get_message_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stored_messages_.size();
}

size_t ReplayBuffer::get_total_size() const {
    return total_stored_bytes_;
}

void ReplayBuffer::cleanup_old_messages() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto now = std::chrono::steady_clock::now();
    std::vector<MessageID> to_remove;
    
    for (const auto& [msg_id, stored_msg] : stored_messages_) {
        if (should_cleanup_message(stored_msg)) {
            to_remove.push_back(msg_id);
        }
    }
    
    for (MessageID msg_id : to_remove) {
        auto it = stored_messages_.find(msg_id);
        if (it != stored_messages_.end()) {
            total_stored_bytes_ -= it->second.data.size();
            stored_messages_.erase(it);
            stats_.messages_expired++;
        }
    }
    
    // Remove from order tracking
    for (MessageID msg_id : to_remove) {
        message_order_.erase(
            std::remove(message_order_.begin(), message_order_.end(), msg_id),
            message_order_.end());
    }
    
    stats_.cleanup_runs++;
}

void ReplayBuffer::cleanup_by_age(std::chrono::minutes max_age) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto cutoff_time = std::chrono::steady_clock::now() - max_age;
    std::vector<MessageID> to_remove;
    
    for (const auto& [msg_id, stored_msg] : stored_messages_) {
        if (stored_msg.timestamp < cutoff_time) {
            to_remove.push_back(msg_id);
        }
    }
    
    for (MessageID msg_id : to_remove) {
        auto it = stored_messages_.find(msg_id);
        if (it != stored_messages_.end()) {
            total_stored_bytes_ -= it->second.data.size();
            stored_messages_.erase(it);
            stats_.messages_expired++;
        }
    }
    
    // Remove from order tracking
    for (MessageID msg_id : to_remove) {
        message_order_.erase(
            std::remove(message_order_.begin(), message_order_.end(), msg_id),
            message_order_.end());
    }
}

void ReplayBuffer::cleanup_by_count(size_t max_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (stored_messages_.size() > max_count) {
        evict_oldest_messages(stored_messages_.size() - max_count);
    }
}

void ReplayBuffer::reset_stats() {
    stats_.messages_stored = 0;
    stats_.messages_replayed = 0;
    stats_.replay_successes = 0;
    stats_.replay_failures = 0;
    stats_.messages_expired = 0;
    stats_.messages_evicted = 0;
    stats_.cleanup_runs = 0;
}

void ReplayBuffer::start() {
    if (!running_) {
        running_ = true;
        paused_ = false;
        
        replay_thread_ = std::thread(&ReplayBuffer::replay_worker, this);
        
        if (config_.auto_cleanup) {
            cleanup_thread_ = std::thread(&ReplayBuffer::cleanup_worker, this);
        }
    }
}

void ReplayBuffer::stop() {
    if (running_) {
        running_ = false;
        replay_cv_.notify_all();
        cleanup_cv_.notify_all();
        
        if (replay_thread_.joinable()) {
            replay_thread_.join();
        }
        
        if (cleanup_thread_.joinable()) {
            cleanup_thread_.join();
        }
    }
}

void ReplayBuffer::pause() {
    paused_ = true;
}

void ReplayBuffer::resume() {
    paused_ = false;
    replay_cv_.notify_all();
    cleanup_cv_.notify_all();
}

void ReplayBuffer::replay_worker() {
    while (running_) {
        std::unique_lock<std::mutex> lock(mutex_);
        replay_cv_.wait(lock, [this] { return !running_ || (!replay_queue_.empty() && !paused_); });
        
        if (!running_) break;
        
        if (!replay_queue_.empty()) {
            ReplayRequest request = std::move(replay_queue_.front());
            replay_queue_.pop_front();
            lock.unlock();
            
            // Process the replay request
            bool success = false;
            std::string error;
            
            try {
                success = replay_message(request.message_id, *request.target_channel);
                if (!success) {
                    error = "Message replay failed (limits exceeded or not found)";
                }
            } catch (const std::exception& e) {
                error = e.what();
            }
            
            if (request.callback) {
                request.callback(success, error);
            }
        }
    }
}

void ReplayBuffer::cleanup_worker() {
    while (running_) {
        std::unique_lock<std::mutex> lock(mutex_);
        cleanup_cv_.wait_for(lock, config_.cleanup_interval, [this] { return !running_ || !paused_; });
        
        if (!running_) break;
        
        if (!paused_) {
            lock.unlock();
            cleanup_old_messages();
        }
    }
}

bool ReplayBuffer::send_stored_message(const StoredMessage& stored_msg, Channel& channel) {
    try {
        // This is a simplified approach - in a real implementation, we would need
        // to reconstruct the proper message type and send it through the channel
        // For now, we'll assume the channel can handle raw data
        
        // NOTE: This would need integration with the actual message system
        // to properly reconstruct and send typed messages
        
        return true; // Placeholder - actual implementation would depend on message system integration
    } catch (const std::exception&) {
        return false;
    }
}

void ReplayBuffer::evict_oldest_messages(size_t count) {
    // Evict from the front of the order queue (oldest)
    for (size_t i = 0; i < count && !message_order_.empty(); ++i) {
        MessageID oldest_id = message_order_.front();
        message_order_.pop_front();
        
        auto it = stored_messages_.find(oldest_id);
        if (it != stored_messages_.end()) {
            total_stored_bytes_ -= it->second.data.size();
            stored_messages_.erase(it);
            stats_.messages_evicted++;
        }
    }
}

void ReplayBuffer::update_message_order(MessageID msg_id) {
    // Remove from current position
    message_order_.erase(
        std::remove(message_order_.begin(), message_order_.end(), msg_id),
        message_order_.end());
    
    // Add to back (most recently used)
    message_order_.push_back(msg_id);
}

bool ReplayBuffer::should_cleanup_message(const StoredMessage& msg) const {
    auto now = std::chrono::steady_clock::now();
    
    // Check age
    if (now - msg.timestamp > config_.max_age) {
        return true;
    }
    
    // Check if replay count exceeded
    if (msg.replay_count >= config_.max_replay_count) {
        return true;
    }
    
    return false;
}

// Utility functions
ReplayBufferConfig create_short_term_replay_config(size_t max_messages, std::chrono::minutes max_age) {
    ReplayBufferConfig config;
    config.max_messages = max_messages;
    config.max_age = max_age;
    config.max_replay_count = 3;
    config.replay_delay = std::chrono::milliseconds(500);
    config.auto_cleanup = true;
    config.cleanup_interval = std::chrono::minutes(1);
    return config;
}

ReplayBufferConfig create_long_term_replay_config(size_t max_messages, std::chrono::minutes max_age) {
    ReplayBufferConfig config;
    config.max_messages = max_messages;
    config.max_age = max_age;
    config.max_replay_count = 10;
    config.replay_delay = std::chrono::seconds(5);
    config.auto_cleanup = true;
    config.cleanup_interval = std::chrono::minutes(10);
    return config;
}

ReplayBufferConfig create_minimal_replay_config(size_t max_messages, std::chrono::minutes max_age) {
    ReplayBufferConfig config;
    config.max_messages = max_messages;
    config.max_age = max_age;
    config.max_replay_count = 1;
    config.replay_delay = std::chrono::milliseconds(100);
    config.auto_cleanup = true;
    config.cleanup_interval = std::chrono::minutes(1);
    return config;
}

} // namespace reliability
} // namespace psyne