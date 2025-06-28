#pragma once

// Psyne Reliability Features
// Comprehensive reliability and fault tolerance for zero-copy messaging

#include "acknowledgment.hpp"
#include "retry.hpp"
#include "heartbeat.hpp"
#include <optional>
#include <memory>

namespace psyne {

// Forward declarations
enum class ChannelMode : int;
enum class ChannelType : int;
class Channel;
class FloatVector;
class DoubleMatrix;
std::unique_ptr<Channel> create_channel(const std::string& uri, size_t buffer_size, 
                                       ChannelMode mode, ChannelType type);

namespace reliability {

// Integrated reliability manager that combines all reliability features
class ReliabilityManager {
public:
    ReliabilityManager();
    ~ReliabilityManager();
    
    // Component access
    AcknowledgmentManager& acknowledgments() { return ack_manager_; }
    RetryManager& retries() { return retry_manager_; }
    HeartbeatManager& heartbeats() { return heartbeat_manager_; }
    
    const AcknowledgmentManager& acknowledgments() const { return ack_manager_; }
    const RetryManager& retries() const { return retry_manager_; }
    const HeartbeatManager& heartbeats() const { return heartbeat_manager_; }
    
    // Integrated operations
    template<typename MessageType>
    void send_reliable_message(Channel& channel, MessageType& message,
                              AckType ack_type = AckType::Simple,
                              const RetryConfig& retry_config = {},
                              std::chrono::milliseconds ack_timeout = std::chrono::milliseconds::zero());
    
    // Connection management with full reliability
    void add_reliable_connection(const std::string& connection_id, Channel& channel,
                               const HeartbeatConfig& heartbeat_config = {},
                               const RetryConfig& retry_config = {});
    
    void remove_reliable_connection(const std::string& connection_id);
    
    // Combined statistics (with plain uint64_t instead of atomics for copying)
    struct CombinedStats {
        struct AckStats {
            uint64_t messages_sent;
            uint64_t messages_acknowledged;
            uint64_t messages_failed;
            uint64_t messages_timeout;
            uint64_t total_retries;
        } acknowledgments;
        
        struct RetryStatsData {
            uint64_t total_retries;
            uint64_t successful_retries;
            uint64_t failed_retries;
            uint64_t max_retries_exceeded;
            uint64_t circuit_breaker_open;
        } retries;
        
        struct HeartbeatStatsData {
            uint64_t heartbeats_sent;
            uint64_t heartbeats_received;
            uint64_t heartbeats_missed;
            uint64_t connections_lost;
            uint64_t reconnections_attempted;
            uint64_t reconnections_successful;
            uint64_t reconnections_failed;
        } heartbeats;
        
        // Derived metrics
        double message_success_rate() const;
        double connection_uptime() const;
        std::chrono::milliseconds average_retry_delay() const;
    };
    
    CombinedStats get_combined_stats() const;
    void reset_all_stats();
    
    // Control all components
    void start_all();
    void stop_all();
    void pause_all();
    void resume_all();

private:
    AcknowledgmentManager ack_manager_;
    RetryManager retry_manager_;
    HeartbeatManager heartbeat_manager_;
    
    bool started_;
};

// Factory function for creating reliable channels with all features enabled
std::unique_ptr<Channel> create_reliable_channel(
    const std::string& uri,
    ReliabilityManager& reliability_mgr,
    size_t buffer_size,
    ChannelMode mode,
    ChannelType type,
    const HeartbeatConfig& heartbeat_config,
    const RetryConfig& retry_config);

// Utility class for easy reliable messaging setup
class ReliableChannelGuard {
public:
    ReliableChannelGuard(const std::string& uri, 
                        size_t buffer_size,
                        const HeartbeatConfig& heartbeat_config,
                        const RetryConfig& retry_config);
    
    ~ReliableChannelGuard() = default;
    
    // Non-copyable, non-movable (due to mutex members in reliability manager)
    ReliableChannelGuard(const ReliableChannelGuard&) = delete;
    ReliableChannelGuard& operator=(const ReliableChannelGuard&) = delete;
    ReliableChannelGuard(ReliableChannelGuard&&) = delete;
    ReliableChannelGuard& operator=(ReliableChannelGuard&&) = delete;
    
    Channel& channel() { return *channel_; }
    const Channel& channel() const { return *channel_; }
    
    ReliabilityManager& reliability() { return reliability_mgr_; }
    const ReliabilityManager& reliability() const { return reliability_mgr_; }
    
    template<typename MessageType>
    void send_reliable(MessageType& message, 
                      AckType ack_type = AckType::Simple,
                      const RetryConfig& retry_config = {}) {
        reliability_mgr_.send_reliable_message(channel(), message, ack_type, retry_config);
    }

private:
    ReliabilityManager reliability_mgr_;
    std::unique_ptr<Channel> channel_;
    std::string connection_id_;
    std::optional<HeartbeatGuard> heartbeat_guard_;
};

} // namespace reliability
} // namespace psyne