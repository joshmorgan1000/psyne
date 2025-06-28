#include "../../include/psyne/reliability/heartbeat.hpp"
#include <algorithm>
#include <random>

namespace psyne {
namespace reliability {

HeartbeatManager::HeartbeatManager() 
    : sender_id_(generate_sender_id())
    , running_(false)
    , paused_(false) {
    
    // Set default heartbeat configuration
    default_config_.interval = std::chrono::seconds(30);
    default_config_.timeout = std::chrono::seconds(90);
    default_config_.max_missed_heartbeats = 3;
    default_config_.send_heartbeats = true;
    default_config_.monitor_heartbeats = true;
    default_config_.auto_reconnect = true;
    default_config_.reconnect_delay = std::chrono::seconds(5);
    default_config_.max_reconnect_attempts = 10;
}

HeartbeatManager::~HeartbeatManager() {
    stop();
}

void HeartbeatManager::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        running_ = true;
        paused_ = false;
        heartbeat_thread_ = std::thread(&HeartbeatManager::heartbeat_worker, this);
    }
}

void HeartbeatManager::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        paused_ = false;
    }
    cv_.notify_all();
    
    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }
}

void HeartbeatManager::pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = true;
}

void HeartbeatManager::resume() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        paused_ = false;
    }
    cv_.notify_all();
}

uint32_t HeartbeatManager::generate_sender_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint32_t> dis(1, UINT32_MAX);
    return dis(gen);
}

void HeartbeatManager::set_default_config(const HeartbeatConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    default_config_ = config;
}

void HeartbeatManager::add_connection(const std::string& connection_id, Channel& channel,
                                    const HeartbeatConfig& config,
                                    std::function<void(const std::string&, ConnectionState)> state_callback,
                                    std::function<bool(const std::string&)> reconnect_callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    ConnectionInfo conn_info;
    conn_info.connection_id = connection_id;
    conn_info.channel = &channel;
    conn_info.state = ConnectionState::Connected;
    conn_info.last_heartbeat_sent = std::chrono::steady_clock::time_point{};
    conn_info.last_heartbeat_received = std::chrono::steady_clock::now();
    conn_info.last_received_sequence = 0;
    conn_info.next_send_sequence = 1;
    conn_info.missed_heartbeats = 0;
    conn_info.reconnect_attempts = 0;
    conn_info.config = config.interval.count() > 0 ? config : default_config_;
    conn_info.state_change_callback = std::move(state_callback);
    conn_info.reconnect_callback = std::move(reconnect_callback);
    
    connections_[connection_id] = std::move(conn_info);
    cv_.notify_one();
}

void HeartbeatManager::remove_connection(const std::string& connection_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    connections_.erase(connection_id);
}

void HeartbeatManager::update_connection_config(const std::string& connection_id, 
                                               const HeartbeatConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = connections_.find(connection_id);
    if (it != connections_.end()) {
        it->second.config = config;
    }
}

void HeartbeatManager::send_heartbeat(const std::string& connection_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = connections_.find(connection_id);
    if (it == connections_.end()) return;
    
    ConnectionInfo& conn_info = it->second;
    if (!conn_info.config.send_heartbeats || !conn_info.channel) return;
    
    HeartbeatMessage heartbeat;
    heartbeat.sequence_number = conn_info.next_send_sequence++;
    heartbeat.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    heartbeat.sender_id = sender_id_;
    heartbeat.heartbeat_type = 0; // ping
    
    try {
        // TODO: Create actual heartbeat message and send
        // This would require extending the message system to support HeartbeatMessage
        // For now, this is a placeholder for the interface
        
        conn_info.last_heartbeat_sent = std::chrono::steady_clock::now();
        stats_.heartbeats_sent.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception& e) {
        // Failed to send heartbeat - consider connection issues
        update_connection_state(conn_info, ConnectionState::Failed);
    }
}

void HeartbeatManager::process_heartbeat(const std::string& connection_id, 
                                       const HeartbeatMessage& heartbeat) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = connections_.find(connection_id);
    if (it == connections_.end()) return;
    
    ConnectionInfo& conn_info = it->second;
    
    if (heartbeat.heartbeat_type == 0) { // ping
        // Send pong response
        if (conn_info.channel) {
            send_heartbeat_response(connection_id, heartbeat);
        }
    } else if (heartbeat.heartbeat_type == 1) { // pong
        // Update connection state
        conn_info.last_heartbeat_received = std::chrono::steady_clock::now();
        conn_info.last_received_sequence = heartbeat.sequence_number;
        conn_info.missed_heartbeats = 0;
        
        if (conn_info.state != ConnectionState::Connected) {
            update_connection_state(conn_info, ConnectionState::Connected);
        }
        
        stats_.heartbeats_received.fetch_add(1, std::memory_order_relaxed);
    }
}

void HeartbeatManager::send_heartbeat_response(const std::string& connection_id,
                                             const HeartbeatMessage& received_heartbeat) {
    auto it = connections_.find(connection_id);
    if (it == connections_.end()) return;
    
    ConnectionInfo& conn_info = it->second;
    if (!conn_info.channel) return;
    
    HeartbeatMessage response;
    response.sequence_number = received_heartbeat.sequence_number; // Echo sequence
    response.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    response.sender_id = sender_id_;
    response.heartbeat_type = 1; // pong
    
    try {
        // TODO: Create and send heartbeat response message
        // Placeholder for actual implementation
    } catch (const std::exception& e) {
        // Failed to send response
    }
}

ConnectionState HeartbeatManager::get_connection_state(const std::string& connection_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = connections_.find(connection_id);
    return it != connections_.end() ? it->second.state : ConnectionState::Unknown;
}

std::vector<std::string> HeartbeatManager::get_all_connections() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    result.reserve(connections_.size());
    
    for (const auto& [conn_id, conn_info] : connections_) {
        result.push_back(conn_id);
    }
    
    return result;
}

std::vector<std::string> HeartbeatManager::get_disconnected_connections() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    
    for (const auto& [conn_id, conn_info] : connections_) {
        if (conn_info.state == ConnectionState::Disconnected || 
            conn_info.state == ConnectionState::Failed) {
            result.push_back(conn_id);
        }
    }
    
    return result;
}

void HeartbeatManager::reset_stats() {
    stats_.heartbeats_sent.store(0, std::memory_order_relaxed);
    stats_.heartbeats_received.store(0, std::memory_order_relaxed);
    stats_.heartbeats_missed.store(0, std::memory_order_relaxed);
    stats_.connections_lost.store(0, std::memory_order_relaxed);
    stats_.reconnections_attempted.store(0, std::memory_order_relaxed);
    stats_.reconnections_successful.store(0, std::memory_order_relaxed);
    stats_.reconnections_failed.store(0, std::memory_order_relaxed);
}

void HeartbeatManager::process_all_heartbeats() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [conn_id, conn_info] : connections_) {
        process_connection_heartbeat(conn_info);
        check_connection_timeouts(conn_info);
    }
}

void HeartbeatManager::heartbeat_worker() {
    while (running_) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait for interval or until notified
        cv_.wait_for(lock, std::chrono::seconds(1), [this] { 
            return !running_ || !paused_; 
        });
        
        if (!running_) break;
        if (paused_) continue;
        
        // Process all connections
        for (auto& [conn_id, conn_info] : connections_) {
            process_connection_heartbeat(conn_info);
            check_connection_timeouts(conn_info);
        }
        
        lock.unlock();
        
        // Small delay to prevent busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void HeartbeatManager::process_connection_heartbeat(ConnectionInfo& conn_info) {
    if (!conn_info.config.send_heartbeats) return;
    
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_sent = now - conn_info.last_heartbeat_sent;
    
    if (time_since_last_sent >= conn_info.config.interval) {
        send_heartbeat(conn_info.connection_id);
    }
}

void HeartbeatManager::check_connection_timeouts(ConnectionInfo& conn_info) {
    if (!conn_info.config.monitor_heartbeats) return;
    
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_received = now - conn_info.last_heartbeat_received;
    
    if (time_since_last_received >= conn_info.config.timeout) {
        conn_info.missed_heartbeats++;
        stats_.heartbeats_missed.fetch_add(1, std::memory_order_relaxed);
        
        if (conn_info.missed_heartbeats >= conn_info.config.max_missed_heartbeats) {
            if (conn_info.state == ConnectionState::Connected) {
                update_connection_state(conn_info, ConnectionState::Disconnected);
                stats_.connections_lost.fetch_add(1, std::memory_order_relaxed);
                
                // Attempt reconnection if enabled
                if (conn_info.config.auto_reconnect) {
                    attempt_reconnection(conn_info);
                }
            }
        } else if (conn_info.state == ConnectionState::Connected) {
            update_connection_state(conn_info, ConnectionState::Heartbeat_Missing);
        }
    }
}

void HeartbeatManager::update_connection_state(ConnectionInfo& conn_info, 
                                             ConnectionState new_state) {
    if (conn_info.state != new_state) {
        conn_info.state = new_state;
        
        if (conn_info.state_change_callback) {
            try {
                conn_info.state_change_callback(conn_info.connection_id, new_state);
            } catch (...) {
                // Ignore callback exceptions
            }
        }
    }
}

void HeartbeatManager::attempt_reconnection(ConnectionInfo& conn_info) {
    if (conn_info.reconnect_attempts >= conn_info.config.max_reconnect_attempts) {
        update_connection_state(conn_info, ConnectionState::Failed);
        return;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_attempt = now - conn_info.last_heartbeat_sent;
    
    if (time_since_last_attempt >= conn_info.config.reconnect_delay) {
        conn_info.reconnect_attempts++;
        stats_.reconnections_attempted.fetch_add(1, std::memory_order_relaxed);
        
        update_connection_state(conn_info, ConnectionState::Connecting);
        
        bool success = false;
        if (conn_info.reconnect_callback) {
            try {
                success = conn_info.reconnect_callback(conn_info.connection_id);
            } catch (...) {
                success = false;
            }
        }
        
        if (success) {
            stats_.reconnections_successful.fetch_add(1, std::memory_order_relaxed);
            conn_info.missed_heartbeats = 0;
            conn_info.last_heartbeat_received = now;
            update_connection_state(conn_info, ConnectionState::Connected);
        } else {
            stats_.reconnections_failed.fetch_add(1, std::memory_order_relaxed);
            conn_info.last_heartbeat_sent = now; // Reset for next attempt delay
        }
    }
}

// Utility function implementations
HeartbeatConfig create_fast_heartbeat_config(std::chrono::milliseconds interval,
                                           std::chrono::milliseconds timeout) {
    HeartbeatConfig config;
    config.interval = interval;
    config.timeout = timeout;
    config.max_missed_heartbeats = 2;
    config.send_heartbeats = true;
    config.monitor_heartbeats = true;
    config.auto_reconnect = true;
    config.reconnect_delay = std::chrono::seconds(1);
    config.max_reconnect_attempts = 5;
    return config;
}

HeartbeatConfig create_slow_heartbeat_config(std::chrono::milliseconds interval,
                                           std::chrono::milliseconds timeout) {
    HeartbeatConfig config;
    config.interval = interval;
    config.timeout = timeout;
    config.max_missed_heartbeats = 3;
    config.send_heartbeats = true;
    config.monitor_heartbeats = true;
    config.auto_reconnect = true;
    config.reconnect_delay = std::chrono::seconds(10);
    config.max_reconnect_attempts = 3;
    return config;
}

HeartbeatConfig create_minimal_heartbeat_config(std::chrono::milliseconds interval,
                                              std::chrono::milliseconds timeout) {
    HeartbeatConfig config;
    config.interval = interval;
    config.timeout = timeout;
    config.max_missed_heartbeats = 2;
    config.send_heartbeats = true;
    config.monitor_heartbeats = true;
    config.auto_reconnect = false; // No auto-reconnect for minimal config
    config.reconnect_delay = std::chrono::minutes(1);
    config.max_reconnect_attempts = 1;
    return config;
}

} // namespace reliability
} // namespace psyne