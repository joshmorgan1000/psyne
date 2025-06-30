#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <psyne/psyne.hpp>
#include <string>
#include <vector>
#include "../utils/logger.hpp"

namespace psyne {
namespace debug {

/**
 * @enum ChannelHealth
 * @brief Health status of a channel
 */
enum class ChannelHealth {
    Healthy,      ///< Channel operating normally
    Warning,      ///< Minor issues detected
    Critical,     ///< Serious problems detected
    Disconnected, ///< Channel not connected/available
    Unknown       ///< Cannot determine status
};

/**
 * @struct BufferUsage
 * @brief Information about ring buffer usage
 */
struct BufferUsage {
    size_t total_size = 0;            ///< Total buffer size in bytes
    size_t used_bytes = 0;            ///< Currently used bytes
    size_t available_bytes = 0;       ///< Available space in bytes
    size_t read_position = 0;         ///< Current read position
    size_t write_position = 0;        ///< Current write position
    double utilization_percent = 0.0; ///< Usage as percentage (0-100)

    // Visual representation
    std::string visual_bar() const;
};

/**
 * @struct ConnectionInfo
 * @brief Network connection details for TCP/Unix channels
 */
struct ConnectionInfo {
    std::string local_address;
    std::string remote_address;
    std::string protocol;
    bool is_connected = false;
    std::chrono::milliseconds connection_time{0};
    uint64_t bytes_transmitted = 0;
    uint64_t bytes_received = 0;
};

/**
 * @struct PerformanceSnapshot
 * @brief Point-in-time performance metrics
 */
struct PerformanceSnapshot {
    std::chrono::steady_clock::time_point timestamp;
    ChannelMetrics metrics;
    BufferUsage buffer_usage;
    double messages_per_second = 0.0;
    double bytes_per_second = 0.0;
    double average_message_size = 0.0;
    std::chrono::microseconds average_latency{0};
};

/**
 * @struct ChannelDiagnostics
 * @brief Complete diagnostic information for a channel
 */
struct ChannelDiagnostics {
    std::string uri;
    ChannelType type;
    ChannelMode mode;
    ChannelHealth health;
    BufferUsage buffer_usage;
    ChannelMetrics current_metrics;
    ConnectionInfo connection_info;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    std::chrono::steady_clock::time_point last_activity;

    // Performance history
    std::vector<PerformanceSnapshot> performance_history;

    // Health checks
    bool has_warnings() const {
        return !warnings.empty();
    }
    bool has_errors() const {
        return !errors.empty();
    }
    std::string summary() const;
};

/**
 * @class ChannelInspector
 * @brief Main class for inspecting and debugging channels
 */
class ChannelInspector {
public:
    /**
     * @brief Inspect a channel and return diagnostic information
     */
    static ChannelDiagnostics inspect(const Channel &channel);

    /**
     * @brief Get detailed buffer usage information
     */
    static BufferUsage get_buffer_usage(const Channel &channel);

    /**
     * @brief Analyze channel health and detect issues
     */
    static ChannelHealth analyze_health(const Channel &channel);

    /**
     * @brief Get human-readable status report
     */
    static std::string get_status_report(const Channel &channel);

    /**
     * @brief Create visual representation of buffer state
     */
    static std::string visualize_buffer(const Channel &channel,
                                        size_t width = 50);

    /**
     * @brief Run comprehensive health check
     */
    static std::vector<std::string> health_check(const Channel &channel);
};

/**
 * @class PerformanceProfiler
 * @brief Profiles channel performance over time
 */
class PerformanceProfiler {
public:
    explicit PerformanceProfiler(const Channel &channel);

    /**
     * @brief Start profiling
     */
    void start();

    /**
     * @brief Stop profiling
     */
    void stop();

    /**
     * @brief Take a performance snapshot
     */
    PerformanceSnapshot snapshot();

    /**
     * @brief Get performance history
     */
    const std::vector<PerformanceSnapshot> &history() const {
        return history_;
    }

    /**
     * @brief Generate performance report
     */
    std::string generate_report() const;

    /**
     * @brief Clear history
     */
    void clear_history() {
        history_.clear();
    }

private:
    const Channel *channel_;
    std::vector<PerformanceSnapshot> history_;
    std::chrono::steady_clock::time_point start_time_;
    bool profiling_ = false;
    PerformanceSnapshot last_snapshot_;
};

/**
 * @class MessageTracer
 * @brief Traces message flow through channels for debugging
 */
class MessageTracer {
public:
    struct TraceEvent {
        std::chrono::steady_clock::time_point timestamp;
        std::string channel_uri;
        std::string event_type; // "send", "receive", "drop", "error"
        size_t message_size;
        uint32_t message_type;
        std::string details;
    };

    /**
     * @brief Enable tracing for a channel
     */
    static void enable_tracing(const std::string &channel_uri);

    /**
     * @brief Disable tracing for a channel
     */
    static void disable_tracing(const std::string &channel_uri);

    /**
     * @brief Log a trace event
     */
    static void trace_event(const std::string &channel_uri,
                            const std::string &event_type, size_t message_size,
                            uint32_t message_type = 0,
                            const std::string &details = "");

    /**
     * @brief Get trace history for a channel
     */
    static std::vector<TraceEvent>
    get_trace_history(const std::string &channel_uri);

    /**
     * @brief Clear trace history
     */
    static void clear_traces(const std::string &channel_uri = "");

    /**
     * @brief Generate trace report
     */
    static std::string generate_trace_report(const std::string &channel_uri);

private:
    static std::map<std::string, std::vector<TraceEvent>> traces_;
    static std::map<std::string, bool> tracing_enabled_;
};

/**
 * @class DebugConsole
 * @brief Interactive debugging console for channels
 */
class DebugConsole {
public:
    /**
     * @brief Start interactive debug session
     */
    static void start_session();

    /**
     * @brief Register a channel for debugging
     */
    static void register_channel(const std::string &name,
                                 std::shared_ptr<Channel> channel);

    /**
     * @brief Execute debug command
     */
    static std::string execute_command(const std::string &command);

    /**
     * @brief Show help for available commands
     */
    static std::string show_help();

private:
    static std::map<std::string, std::shared_ptr<Channel>> registered_channels_;
    static std::map<std::string, std::function<std::string(
                                     const std::vector<std::string> &)>>
        commands_;

    static void initialize_commands();
};

/**
 * @brief Print formatted diagnostic information
 */
void print_diagnostics(const ChannelDiagnostics &diag, bool verbose = false);

/**
 * @brief Create a visual dashboard showing all registered channels
 */
std::string create_dashboard();

/**
 * @brief Monitor channels continuously and alert on issues
 */
class ChannelMonitor {
public:
    /**
     * @brief Add channel to monitor
     */
    void add_channel(const std::string &name, std::shared_ptr<Channel> channel);

    /**
     * @brief Start monitoring (runs in background thread)
     */
    void start_monitoring(
        std::chrono::milliseconds interval = std::chrono::milliseconds(1000));

    /**
     * @brief Stop monitoring
     */
    void stop_monitoring();

    /**
     * @brief Set alert callback for health issues
     */
    void
    set_alert_callback(std::function<void(const std::string &, ChannelHealth,
                                          const std::string &)>
                           callback);

    /**
     * @brief Get current status of all monitored channels
     */
    std::string get_status_summary();

private:
    std::map<std::string, std::shared_ptr<Channel>> channels_;
    std::function<void(const std::string &, ChannelHealth, const std::string &)>
        alert_callback_;
    bool monitoring_ = false;
    std::thread monitor_thread_;

    void monitor_loop(std::chrono::milliseconds interval);
};

} // namespace debug
} // namespace psyne