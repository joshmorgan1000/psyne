#pragma once

#include <chrono>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <cstdint>
#include <thread>

namespace psyne {

// Forward declarations
class Channel;

// MessageID type definition 
using MessageID = uint64_t;

namespace reliability {

// Message acknowledgment types
enum class AckType : uint8_t {
    None = 0,        // No acknowledgment required
    Simple = 1,      // Simple ACK/NACK
    Delivery = 2,    // Delivery confirmation
    Processing = 3   // Processing completion confirmation
};

// Acknowledgment status
enum class AckStatus : uint8_t {
    Pending = 0,     // Waiting for acknowledgment
    Acknowledged = 1, // Successfully acknowledged
    Failed = 2,      // Failed to acknowledge (NACK or timeout)
    Timeout = 3      // Acknowledgment timeout
};

// Acknowledgment message structure
struct AckMessage {
    static constexpr uint32_t message_type = 100;
    MessageID original_msg_id;
    AckStatus status;
    uint64_t timestamp;
    uint32_t error_code; // Optional error information
};

// Acknowledgment request information
struct AckRequest {
    MessageID msg_id;
    AckType type;
    std::chrono::steady_clock::time_point sent_time;
    std::chrono::milliseconds timeout;
    std::function<void(MessageID, AckStatus)> callback;
    size_t retry_count;
    size_t max_retries;
};

// Message acknowledgment manager
class AcknowledgmentManager {
public:
    AcknowledgmentManager();
    ~AcknowledgmentManager();
    
    // Configuration
    void set_default_timeout(std::chrono::milliseconds timeout);
    void set_default_max_retries(size_t max_retries);
    void set_cleanup_interval(std::chrono::milliseconds interval);
    
    // Message tracking
    MessageID generate_message_id();
    void track_message(MessageID msg_id, AckType type, 
                      std::chrono::milliseconds timeout = std::chrono::milliseconds::zero(),
                      std::function<void(MessageID, AckStatus)> callback = nullptr);
    
    // Acknowledgment handling
    void process_acknowledgment(const AckMessage& ack);
    void send_acknowledgment(Channel& channel, MessageID msg_id, AckStatus status, uint32_t error_code = 0);
    
    // Timeout and retry management
    void check_timeouts();
    void retry_message(MessageID msg_id);
    
    // Status queries
    AckStatus get_status(MessageID msg_id) const;
    bool is_acknowledged(MessageID msg_id) const;
    size_t pending_count() const;
    
    // Statistics
    struct Stats {
        std::atomic<uint64_t> messages_sent{0};
        std::atomic<uint64_t> messages_acknowledged{0};
        std::atomic<uint64_t> messages_failed{0};
        std::atomic<uint64_t> messages_timeout{0};
        std::atomic<uint64_t> total_retries{0};
    };
    
    const Stats& get_stats() const { return stats_; }
    void reset_stats();
    
    // Cleanup
    void cleanup_completed();
    void stop();

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    
    std::atomic<MessageID> next_msg_id_;
    std::unordered_map<MessageID, AckRequest> pending_acks_;
    std::unordered_map<MessageID, AckStatus> completed_acks_;
    
    // Configuration
    std::chrono::milliseconds default_timeout_;
    size_t default_max_retries_;
    std::chrono::milliseconds cleanup_interval_;
    
    // Background thread for timeout checking
    std::thread timeout_thread_;
    std::atomic<bool> running_;
    
    Stats stats_;
    
    void timeout_worker();
    void cleanup_old_entries();
};

// RAII wrapper for automatic message acknowledgment tracking
template<typename MessageType>
class AcknowledgedMessage {
public:
    AcknowledgedMessage(Channel& channel, AcknowledgmentManager& ack_mgr, 
                       AckType ack_type = AckType::Simple,
                       std::chrono::milliseconds timeout = std::chrono::milliseconds::zero())
        : message_(channel), ack_mgr_(ack_mgr), msg_id_(0), sent_(false) {
        msg_id_ = ack_mgr_.generate_message_id();
        
        // Track this message for acknowledgment
        ack_mgr_.track_message(msg_id_, ack_type, timeout);
    }
    
    ~AcknowledgedMessage() {
        if (!sent_) {
            // If message was never sent, clean up tracking
            // This would need implementation in AcknowledgmentManager
        }
    }
    
    // Access underlying message
    MessageType& message() { return message_; }
    const MessageType& message() const { return message_; }
    
    // Send with acknowledgment tracking
    void send() {
        if (sent_) {
            throw std::runtime_error("Message already sent");
        }
        
        // TODO: Add message ID to message header
        message_.send();
        sent_ = true;
    }
    
    // Query acknowledgment status
    AckStatus get_ack_status() const {
        return ack_mgr_.get_status(msg_id_);
    }
    
    bool is_acknowledged() const {
        return ack_mgr_.is_acknowledged(msg_id_);
    }
    
    MessageID get_message_id() const { return msg_id_; }

private:
    MessageType message_;
    AcknowledgmentManager& ack_mgr_;
    MessageID msg_id_;
    bool sent_;
};

// Factory function for creating acknowledged messages
template<typename MessageType>
AcknowledgedMessage<MessageType> create_acknowledged_message(
    Channel& channel, AcknowledgmentManager& ack_mgr,
    AckType ack_type = AckType::Simple,
    std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
    return AcknowledgedMessage<MessageType>(channel, ack_mgr, ack_type, timeout);
}

} // namespace reliability
} // namespace psyne