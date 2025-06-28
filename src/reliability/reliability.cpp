#include "../../include/psyne/reliability/reliability.hpp"
#include "../../include/psyne/psyne.hpp"
#include <sstream>
#include <algorithm>

namespace psyne {
namespace reliability {

ReliabilityManager::ReliabilityManager() : started_(false) {
    // Configure components to work together
    
    // Set reasonable defaults for integrated operation
    RetryConfig default_retry_config = create_exponential_backoff_config(3, 
        std::chrono::milliseconds(500), 2.0, std::chrono::seconds(10));
    retry_manager_.set_default_config(default_retry_config);
    
    HeartbeatConfig default_heartbeat_config = create_fast_heartbeat_config(
        std::chrono::seconds(30), std::chrono::seconds(90));
    heartbeat_manager_.set_default_config(default_heartbeat_config);
    
    ack_manager_.set_default_timeout(std::chrono::seconds(30));
    ack_manager_.set_default_max_retries(3);
}

ReliabilityManager::~ReliabilityManager() {
    stop_all();
}

template<typename MessageType>
void ReliabilityManager::send_reliable_message(Channel& channel, MessageType& message,
                                              AckType ack_type,
                                              const RetryConfig& retry_config,
                                              std::chrono::milliseconds ack_timeout) {
    
    // Generate message ID for tracking
    MessageID msg_id = ack_manager_.generate_message_id();
    
    // Set up acknowledgment tracking if required
    if (ack_type != AckType::None) {
        auto timeout = ack_timeout.count() > 0 ? ack_timeout : std::chrono::seconds(30);
        
        auto ack_callback = [this, msg_id](MessageID id, AckStatus status) {
            // Handle acknowledgment result
            if (status == AckStatus::Failed || status == AckStatus::Timeout) {
                // Could trigger additional retry logic here
            }
        };
        
        ack_manager_.track_message(msg_id, ack_type, timeout, ack_callback);
    }
    
    // Set up retry mechanism
    auto retry_func = [&message]() -> bool {
        try {
            message.send();
            return true;
        } catch (const std::exception&) {
            return false;
        }
    };
    
    auto completion_callback = [this, msg_id, ack_type](bool success, size_t attempts) {
        if (!success && ack_type != AckType::None) {
            // Mark acknowledgment as failed if retry exhausted
            // This would need coordination with the acknowledgment manager
        }
    };
    
    // Schedule the retry operation
    retry_manager_.schedule_retry(retry_func, retry_config, completion_callback);
}

void ReliabilityManager::add_reliable_connection(const std::string& connection_id, 
                                                Channel& channel,
                                                const HeartbeatConfig& heartbeat_config,
                                                const RetryConfig& retry_config) {
    
    // Set up heartbeat monitoring with connection state callback
    auto state_callback = [this, connection_id](const std::string& conn_id, ConnectionState state) {
        // Handle connection state changes
        if (state == ConnectionState::Disconnected || state == ConnectionState::Failed) {
            // Could pause retries for this connection or trigger reconnection
        } else if (state == ConnectionState::Connected) {
            // Resume normal operations
        }
    };
    
    auto reconnect_callback = [&channel](const std::string& conn_id) -> bool {
        // Attempt to reconnect the channel
        // This would need channel-specific reconnection logic
        // For now, this is a placeholder
        return false;
    };
    
    heartbeat_manager_.add_connection(connection_id, channel, heartbeat_config,
                                    state_callback, reconnect_callback);
}

void ReliabilityManager::remove_reliable_connection(const std::string& connection_id) {
    heartbeat_manager_.remove_connection(connection_id);
    // Could also cancel pending retries and acknowledgments for this connection
}

ReliabilityManager::CombinedStats ReliabilityManager::get_combined_stats() const {
    CombinedStats stats;
    
    // Copy acknowledgment stats
    const auto& ack_stats = ack_manager_.get_stats();
    stats.acknowledgments.messages_sent = ack_stats.messages_sent.load();
    stats.acknowledgments.messages_acknowledged = ack_stats.messages_acknowledged.load();
    stats.acknowledgments.messages_failed = ack_stats.messages_failed.load();
    stats.acknowledgments.messages_timeout = ack_stats.messages_timeout.load();
    stats.acknowledgments.total_retries = ack_stats.total_retries.load();
    
    // Copy retry stats
    const auto& retry_stats = retry_manager_.get_stats();
    stats.retries.total_retries = retry_stats.total_retries.load();
    stats.retries.successful_retries = retry_stats.successful_retries.load();
    stats.retries.failed_retries = retry_stats.failed_retries.load();
    stats.retries.max_retries_exceeded = retry_stats.max_retries_exceeded.load();
    stats.retries.circuit_breaker_open = retry_stats.circuit_breaker_open.load();
    
    // Copy heartbeat stats
    const auto& heartbeat_stats = heartbeat_manager_.get_stats();
    stats.heartbeats.heartbeats_sent = heartbeat_stats.heartbeats_sent.load();
    stats.heartbeats.heartbeats_received = heartbeat_stats.heartbeats_received.load();
    stats.heartbeats.heartbeats_missed = heartbeat_stats.heartbeats_missed.load();
    stats.heartbeats.connections_lost = heartbeat_stats.connections_lost.load();
    stats.heartbeats.reconnections_attempted = heartbeat_stats.reconnections_attempted.load();
    stats.heartbeats.reconnections_successful = heartbeat_stats.reconnections_successful.load();
    stats.heartbeats.reconnections_failed = heartbeat_stats.reconnections_failed.load();
    
    return stats;
}

double ReliabilityManager::CombinedStats::message_success_rate() const {
    uint64_t total_sent = acknowledgments.messages_sent;
    uint64_t total_acked = acknowledgments.messages_acknowledged;
    
    if (total_sent == 0) return 1.0;
    return static_cast<double>(total_acked) / static_cast<double>(total_sent);
}

double ReliabilityManager::CombinedStats::connection_uptime() const {
    uint64_t total_connections = heartbeats.connections_lost + 
                               heartbeats.reconnections_successful;
    uint64_t lost_connections = heartbeats.connections_lost;
    
    if (total_connections == 0) return 1.0;
    return 1.0 - (static_cast<double>(lost_connections) / static_cast<double>(total_connections));
}

std::chrono::milliseconds ReliabilityManager::CombinedStats::average_retry_delay() const {
    // This is a simplified calculation - in practice would need more detailed tracking
    uint64_t total_retries = retries.total_retries;
    if (total_retries == 0) return std::chrono::milliseconds(0);
    
    // Estimate based on exponential backoff pattern
    return std::chrono::milliseconds(500 * total_retries / std::max(1ULL, 
        retries.successful_retries + retries.failed_retries));
}

void ReliabilityManager::reset_all_stats() {
    ack_manager_.reset_stats();
    retry_manager_.reset_stats();
    heartbeat_manager_.reset_stats();
}

void ReliabilityManager::start_all() {
    if (!started_) {
        retry_manager_.start();
        heartbeat_manager_.start();
        started_ = true;
    }
}

void ReliabilityManager::stop_all() {
    if (started_) {
        ack_manager_.stop();
        retry_manager_.stop();
        heartbeat_manager_.stop();
        started_ = false;
    }
}

void ReliabilityManager::pause_all() {
    retry_manager_.stop(); // RetryManager doesn't have pause, so stop/start
    heartbeat_manager_.pause();
}

void ReliabilityManager::resume_all() {
    retry_manager_.start();
    heartbeat_manager_.resume();
}

// Factory function implementation
std::unique_ptr<Channel> create_reliable_channel(const std::string& uri,
                                                ReliabilityManager& reliability_mgr,
                                                size_t buffer_size,
                                                ChannelMode mode,
                                                ChannelType type,
                                                const HeartbeatConfig& heartbeat_config,
                                                const RetryConfig& retry_config) {
    auto channel = create_channel(uri, buffer_size, mode, type);
    
    if (channel) {
        // Extract connection ID from URI for tracking
        std::string connection_id = uri; // Simplified - could parse URI for better ID
        reliability_mgr.add_reliable_connection(connection_id, *channel, 
                                              heartbeat_config, retry_config);
    }
    
    return channel;
}

// ReliableChannelGuard implementation
ReliableChannelGuard::ReliableChannelGuard(const std::string& uri,
                                          size_t buffer_size,
                                          const HeartbeatConfig& heartbeat_config,
                                          const RetryConfig& retry_config)
    : connection_id_(uri) {
    
    reliability_mgr_.start_all();
    
    channel_ = create_channel(uri, buffer_size, ChannelMode::SPSC, ChannelType::MultiType);
    if (!channel_) {
        throw std::runtime_error("Failed to create channel: " + uri);
    }
    
    reliability_mgr_.add_reliable_connection(connection_id_, *channel_, 
                                           heartbeat_config, retry_config);
    
    // Initialize heartbeat guard after everything is set up
    heartbeat_guard_.emplace(reliability_mgr_.heartbeats(), connection_id_, *channel_, heartbeat_config);
}

// Note: Template instantiations are handled by the inclusion of actual message type headers
// when this library is used by clients. The template methods will be instantiated on demand.

} // namespace reliability
} // namespace psyne