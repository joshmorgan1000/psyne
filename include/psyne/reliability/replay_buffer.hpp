#pragma once

#include <chrono>
#include <vector>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <condition_variable>

namespace psyne {

// Forward declarations
class Channel;

// MessageID type definition
using MessageID = uint64_t;

namespace reliability {

// Stored message information for replay
struct StoredMessage {
    MessageID message_id;
    std::vector<uint8_t> data;
    uint32_t message_type;
    std::chrono::steady_clock::time_point timestamp;
    size_t replay_count;
    std::chrono::steady_clock::time_point last_replay;
};

// Replay buffer configuration
struct ReplayBufferConfig {
    size_t max_messages = 1000;           // Maximum messages to store
    std::chrono::minutes max_age = std::chrono::minutes(60);  // Maximum age of stored messages
    size_t max_replay_count = 5;          // Maximum times a message can be replayed
    std::chrono::milliseconds replay_delay = std::chrono::seconds(1);  // Delay between replays
    bool auto_cleanup = true;             // Automatically clean up old messages
    std::chrono::minutes cleanup_interval = std::chrono::minutes(5);   // Cleanup frequency
};

// Replay request information
struct ReplayRequest {
    MessageID message_id;
    Channel* target_channel;
    std::function<void(bool success, const std::string& error)> callback;
    std::chrono::steady_clock::time_point requested_at;
};

// Message replay buffer for fault tolerance
class ReplayBuffer {
public:
    ReplayBuffer(const ReplayBufferConfig& config = {});
    ~ReplayBuffer();
    
    // Configuration
    void set_config(const ReplayBufferConfig& config);
    const ReplayBufferConfig& get_config() const { return config_; }
    
    // Message storage
    void store_message(MessageID msg_id, const void* data, size_t size, uint32_t message_type);
    void store_message(MessageID msg_id, const std::vector<uint8_t>& data, uint32_t message_type);
    
    // Message replay
    bool replay_message(MessageID msg_id, Channel& target_channel);
    void replay_message_async(MessageID msg_id, Channel& target_channel,
                             std::function<void(bool success, const std::string& error)> callback = nullptr);
    
    // Range replay (replay all messages in a time range)
    size_t replay_range(std::chrono::steady_clock::time_point start_time,
                       std::chrono::steady_clock::time_point end_time,
                       Channel& target_channel);
    
    // Replay all messages after a specific message ID
    size_t replay_from_message(MessageID start_msg_id, Channel& target_channel);
    
    // Message management
    bool has_message(MessageID msg_id) const;
    bool remove_message(MessageID msg_id);
    void clear_all_messages();
    
    // Message queries
    std::vector<MessageID> get_stored_message_ids() const;
    std::vector<MessageID> get_messages_in_range(
        std::chrono::steady_clock::time_point start_time,
        std::chrono::steady_clock::time_point end_time) const;
    
    size_t get_message_count() const;
    size_t get_total_size() const;  // Total bytes stored
    
    // Cleanup operations
    void cleanup_old_messages();
    void cleanup_by_age(std::chrono::minutes max_age);
    void cleanup_by_count(size_t max_count);
    
    // Statistics
    struct ReplayStats {
        std::atomic<uint64_t> messages_stored{0};
        std::atomic<uint64_t> messages_replayed{0};
        std::atomic<uint64_t> replay_successes{0};
        std::atomic<uint64_t> replay_failures{0};
        std::atomic<uint64_t> messages_expired{0};
        std::atomic<uint64_t> messages_evicted{0};
        std::atomic<uint64_t> cleanup_runs{0};
    };
    
    const ReplayStats& get_stats() const { return stats_; }
    void reset_stats();
    
    // Control
    void start();
    void stop();
    void pause();
    void resume();

private:
    mutable std::mutex mutex_;
    
    ReplayBufferConfig config_;
    std::unordered_map<MessageID, StoredMessage> stored_messages_;
    std::deque<MessageID> message_order_;  // For LRU eviction
    
    // Async replay processing
    std::deque<ReplayRequest> replay_queue_;
    std::thread replay_thread_;
    std::condition_variable replay_cv_;
    
    // Background cleanup
    std::thread cleanup_thread_;
    std::condition_variable cleanup_cv_;
    
    std::atomic<bool> running_;
    std::atomic<bool> paused_;
    
    // Statistics
    ReplayStats stats_;
    
    // Size tracking
    std::atomic<size_t> total_stored_bytes_;
    
    // Helper methods
    void replay_worker();
    void cleanup_worker();
    bool send_stored_message(const StoredMessage& stored_msg, Channel& channel);
    void evict_oldest_messages(size_t count);
    void update_message_order(MessageID msg_id);
    bool should_cleanup_message(const StoredMessage& msg) const;
};

// RAII wrapper for automatic message storage
template<typename MessageType>
class ReplayableMessage {
public:
    ReplayableMessage(Channel& channel, ReplayBuffer& replay_buffer)
        : message_(channel), replay_buffer_(replay_buffer), stored_(false) {}
    
    ~ReplayableMessage() {
        if (stored_ && message_.is_valid()) {
            // Store the message in replay buffer before destruction
            replay_buffer_.store_message(message_id_, message_.data(), message_.size(), 
                                       MessageType::message_type);
        }
    }
    
    // Access underlying message
    MessageType& message() { return message_; }
    const MessageType& message() const { return message_; }
    
    // Send with automatic storage
    void send_with_storage(MessageID msg_id) {
        message_id_ = msg_id;
        stored_ = true;
        message_.send();
    }
    
    // Send without storage (normal send)
    void send() {
        message_.send();
    }
    
    // Manual storage control
    void enable_storage(MessageID msg_id) {
        message_id_ = msg_id;
        stored_ = true;
    }
    
    void disable_storage() {
        stored_ = false;
    }
    
    MessageID get_message_id() const { return message_id_; }
    bool is_storage_enabled() const { return stored_; }

private:
    MessageType message_;
    ReplayBuffer& replay_buffer_;
    MessageID message_id_;
    bool stored_;
};

// Factory function for creating replayable messages
template<typename MessageType>
ReplayableMessage<MessageType> create_replayable_message(
    Channel& channel, ReplayBuffer& replay_buffer) {
    return ReplayableMessage<MessageType>(channel, replay_buffer);
}

// Utility functions for common replay configurations
ReplayBufferConfig create_short_term_replay_config(
    size_t max_messages = 100,
    std::chrono::minutes max_age = std::chrono::minutes(5));

ReplayBufferConfig create_long_term_replay_config(
    size_t max_messages = 10000,
    std::chrono::minutes max_age = std::chrono::hours(24));

ReplayBufferConfig create_minimal_replay_config(
    size_t max_messages = 50,
    std::chrono::minutes max_age = std::chrono::minutes(1));

} // namespace reliability
} // namespace psyne