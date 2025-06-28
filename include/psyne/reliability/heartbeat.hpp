#pragma once

#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <functional>
#include <cstdint>
#include <string>
#include <vector>

namespace psyne {

// Forward declarations
class Channel;

namespace reliability {

// Heartbeat message structure
struct HeartbeatMessage {
    static constexpr uint32_t message_type = 101;
    uint64_t sequence_number;
    uint64_t timestamp;
    uint32_t sender_id;
    uint8_t heartbeat_type; // 0 = ping, 1 = pong
};

// Connection states for heartbeat tracking
enum class ConnectionState : uint8_t {
    Unknown = 0,
    Connecting = 1,
    Connected = 2,
    Heartbeat_Missing = 3,
    Disconnected = 4,
    Failed = 5
};

// Heartbeat configuration
struct HeartbeatConfig {
    std::chrono::milliseconds interval = std::chrono::seconds(30);
    std::chrono::milliseconds timeout = std::chrono::seconds(90);
    size_t max_missed_heartbeats = 3;
    bool send_heartbeats = true;
    bool monitor_heartbeats = true;
    bool auto_reconnect = true;
    std::chrono::milliseconds reconnect_delay = std::chrono::seconds(5);
    size_t max_reconnect_attempts = 10;
};

// Connection information for heartbeat monitoring
struct ConnectionInfo {
    std::string connection_id;
    Channel* channel;
    ConnectionState state;
    std::chrono::steady_clock::time_point last_heartbeat_sent;
    std::chrono::steady_clock::time_point last_heartbeat_received;
    uint64_t last_received_sequence;
    uint64_t next_send_sequence;
    size_t missed_heartbeats;
    size_t reconnect_attempts;
    HeartbeatConfig config;
    std::function<void(const std::string&, ConnectionState)> state_change_callback;
    std::function<bool(const std::string&)> reconnect_callback; // Return true if reconnection succeeded
};

// Heartbeat and keepalive manager
class HeartbeatManager {
public:
    HeartbeatManager();
    ~HeartbeatManager();
    
    // Connection management
    void add_connection(const std::string& connection_id, Channel& channel,
                       const HeartbeatConfig& config = {},
                       std::function<void(const std::string&, ConnectionState)> state_callback = nullptr,
                       std::function<bool(const std::string&)> reconnect_callback = nullptr);
    
    void remove_connection(const std::string& connection_id);
    void update_connection_config(const std::string& connection_id, const HeartbeatConfig& config);
    
    // Heartbeat operations
    void send_heartbeat(const std::string& connection_id);
    void process_heartbeat(const std::string& connection_id, const HeartbeatMessage& heartbeat);
    void send_heartbeat_response(const std::string& connection_id, const HeartbeatMessage& received_heartbeat);
    
    // Status and monitoring
    ConnectionState get_connection_state(const std::string& connection_id) const;
    std::vector<std::string> get_all_connections() const;
    std::vector<std::string> get_disconnected_connections() const;
    
    // Statistics
    struct HeartbeatStats {
        std::atomic<uint64_t> heartbeats_sent{0};
        std::atomic<uint64_t> heartbeats_received{0};
        std::atomic<uint64_t> heartbeats_missed{0};
        std::atomic<uint64_t> connections_lost{0};
        std::atomic<uint64_t> reconnections_attempted{0};
        std::atomic<uint64_t> reconnections_successful{0};
        std::atomic<uint64_t> reconnections_failed{0};
    };
    
    const HeartbeatStats& get_stats() const { return stats_; }
    void reset_stats();
    
    // Global configuration
    void set_default_config(const HeartbeatConfig& config);
    const HeartbeatConfig& get_default_config() const { return default_config_; }
    
    // Control
    void start();
    void stop();
    void pause();
    void resume();
    
    // Manual heartbeat processing (for single-threaded mode)
    void process_all_heartbeats();

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    
    std::unordered_map<std::string, ConnectionInfo> connections_;
    HeartbeatConfig default_config_;
    uint32_t sender_id_;
    
    // Background processing
    std::thread heartbeat_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> paused_;
    
    // Statistics
    HeartbeatStats stats_;
    
    // Helper methods
    void heartbeat_worker();
    void process_connection_heartbeat(ConnectionInfo& conn_info);
    void check_connection_timeouts(ConnectionInfo& conn_info);
    void update_connection_state(ConnectionInfo& conn_info, ConnectionState new_state);
    void attempt_reconnection(ConnectionInfo& conn_info);
    uint32_t generate_sender_id();
};

// RAII wrapper for automatic heartbeat management
class HeartbeatGuard {
public:
    HeartbeatGuard(HeartbeatManager& manager, const std::string& connection_id,
                  Channel& channel, const HeartbeatConfig& config = {})
        : manager_(manager), connection_id_(connection_id), active_(true) {
        manager_.add_connection(connection_id_, channel, config);
    }
    
    ~HeartbeatGuard() {
        if (active_) {
            manager_.remove_connection(connection_id_);
        }
    }
    
    // Non-copyable, movable
    HeartbeatGuard(const HeartbeatGuard&) = delete;
    HeartbeatGuard& operator=(const HeartbeatGuard&) = delete;
    
    HeartbeatGuard(HeartbeatGuard&& other) noexcept
        : manager_(other.manager_), connection_id_(std::move(other.connection_id_)), active_(other.active_) {
        other.active_ = false;
    }
    
    HeartbeatGuard& operator=(HeartbeatGuard&& other) noexcept {
        if (this != &other) {
            if (active_) {
                manager_.remove_connection(connection_id_);
            }
            connection_id_ = std::move(other.connection_id_);
            active_ = other.active_;
            other.active_ = false;
        }
        return *this;
    }
    
    void release() { active_ = false; }
    
    ConnectionState get_state() const {
        return manager_.get_connection_state(connection_id_);
    }

private:
    HeartbeatManager& manager_;
    std::string connection_id_;
    bool active_;
};

// Utility functions
HeartbeatConfig create_fast_heartbeat_config(
    std::chrono::milliseconds interval = std::chrono::seconds(10),
    std::chrono::milliseconds timeout = std::chrono::seconds(30));

HeartbeatConfig create_slow_heartbeat_config(
    std::chrono::milliseconds interval = std::chrono::minutes(1),
    std::chrono::milliseconds timeout = std::chrono::minutes(3));

HeartbeatConfig create_minimal_heartbeat_config(
    std::chrono::milliseconds interval = std::chrono::minutes(5),
    std::chrono::milliseconds timeout = std::chrono::minutes(15));

} // namespace reliability
} // namespace psyne