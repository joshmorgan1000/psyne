/**
 * @file metrics_collector.hpp
 * @brief Real-time metrics collection and debugging for Psyne channels
 *
 * Provides comprehensive metrics collection during debug runs including:
 * - Real-time performance counters
 * - Memory usage tracking
 * - Latency histograms
 * - Channel state inspection
 * - Event tracing
 */

#pragma once

#include "psyne/channel/channel_base.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace psyne {
namespace debug {

/**
 * @brief Metrics sampling configuration
 */
struct MetricsConfig {
    bool enabled = true;
    bool console_output = true;
    bool file_output = false;
    std::string output_file = "psyne_metrics.log";
    uint32_t sampling_interval_ms = 1000;
    bool detailed_histograms = true;
    bool memory_tracking = true;
    bool event_tracing = false;
    size_t event_buffer_size = 10000;
    bool live_dashboard = false;
};

/**
 * @brief Event types for tracing
 */
enum class EventType {
    CHANNEL_CREATE,
    CHANNEL_DESTROY,
    MESSAGE_ALLOCATE,
    MESSAGE_COMMIT,
    MESSAGE_RECEIVE,
    MESSAGE_RELEASE,
    BUFFER_FULL,
    BUFFER_EMPTY,
    CONNECTION_ESTABLISHED,
    CONNECTION_LOST,
    ERROR_OCCURRED
};

/**
 * @brief Traced event structure
 */
struct TracedEvent {
    uint64_t timestamp_ns;
    EventType type;
    std::string channel_name;
    uint64_t sequence;
    size_t size;
    std::thread::id thread_id;
    std::string extra_info;
};

/**
 * @brief Latency histogram with percentile calculation
 */
class LatencyHistogram {
public:
    static constexpr size_t NUM_BUCKETS = 50;
    static constexpr uint64_t MAX_LATENCY_NS = 1000000000; // 1 second
    
    LatencyHistogram() : buckets_(NUM_BUCKETS, 0), total_samples_(0) {}
    
    void record(uint64_t latency_ns) {
        size_t bucket = calculate_bucket(latency_ns);
        buckets_[bucket].fetch_add(1, std::memory_order_relaxed);
        total_samples_.fetch_add(1, std::memory_order_relaxed);
        
        // Update min/max
        uint64_t current_min = min_latency_.load();
        while (latency_ns < current_min && 
               !min_latency_.compare_exchange_weak(current_min, latency_ns));
        
        uint64_t current_max = max_latency_.load();
        while (latency_ns > current_max && 
               !max_latency_.compare_exchange_weak(current_max, latency_ns));
    }
    
    struct Percentiles {
        uint64_t p50;
        uint64_t p90;
        uint64_t p95;
        uint64_t p99;
        uint64_t p999;
        uint64_t min;
        uint64_t max;
    };
    
    Percentiles get_percentiles() const {
        Percentiles result{};
        uint64_t total = total_samples_.load();
        if (total == 0) return result;
        
        result.min = min_latency_.load();
        result.max = max_latency_.load();
        
        // Calculate percentiles
        uint64_t running_sum = 0;
        uint64_t p50_target = total * 50 / 100;
        uint64_t p90_target = total * 90 / 100;
        uint64_t p95_target = total * 95 / 100;
        uint64_t p99_target = total * 99 / 100;
        uint64_t p999_target = total * 999 / 1000;
        
        for (size_t i = 0; i < NUM_BUCKETS; ++i) {
            running_sum += buckets_[i].load();
            uint64_t bucket_value = bucket_to_value(i);
            
            if (result.p50 == 0 && running_sum >= p50_target) result.p50 = bucket_value;
            if (result.p90 == 0 && running_sum >= p90_target) result.p90 = bucket_value;
            if (result.p95 == 0 && running_sum >= p95_target) result.p95 = bucket_value;
            if (result.p99 == 0 && running_sum >= p99_target) result.p99 = bucket_value;
            if (result.p999 == 0 && running_sum >= p999_target) result.p999 = bucket_value;
        }
        
        return result;
    }
    
    void reset() {
        for (auto& bucket : buckets_) {
            bucket.store(0);
        }
        total_samples_.store(0);
        min_latency_.store(UINT64_MAX);
        max_latency_.store(0);
    }
    
private:
    std::vector<std::atomic<uint64_t>> buckets_;
    std::atomic<uint64_t> total_samples_;
    std::atomic<uint64_t> min_latency_{UINT64_MAX};
    std::atomic<uint64_t> max_latency_{0};
    
    size_t calculate_bucket(uint64_t latency_ns) const {
        if (latency_ns >= MAX_LATENCY_NS) return NUM_BUCKETS - 1;
        return (latency_ns * NUM_BUCKETS) / MAX_LATENCY_NS;
    }
    
    uint64_t bucket_to_value(size_t bucket) const {
        return (bucket * MAX_LATENCY_NS) / NUM_BUCKETS;
    }
};

/**
 * @brief Per-channel metrics
 */
struct ChannelMetrics {
    std::string name;
    ChannelTransport transport;
    size_t capacity_bytes;
    
    // Counters
    std::atomic<uint64_t> messages_sent{0};
    std::atomic<uint64_t> messages_received{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> allocation_failures{0};
    std::atomic<uint64_t> receive_failures{0};
    
    // Current state
    std::atomic<size_t> bytes_used{0};
    std::atomic<size_t> bytes_available{0};
    std::atomic<bool> is_connected{false};
    
    // Latency tracking
    LatencyHistogram latency_histogram;
    
    // Rate calculation
    std::atomic<uint64_t> last_messages_sent{0};
    std::atomic<uint64_t> last_messages_received{0};
    std::atomic<uint64_t> last_bytes_sent{0};
    std::atomic<uint64_t> last_bytes_received{0};
    std::chrono::steady_clock::time_point last_sample_time;
    
    // Memory tracking
    std::atomic<size_t> current_allocations{0};
    std::atomic<size_t> peak_allocations{0};
    std::atomic<size_t> total_allocation_size{0};
};

/**
 * @brief Global metrics collector singleton
 */
class MetricsCollector {
public:
    static MetricsCollector& instance() {
        static MetricsCollector instance;
        return instance;
    }
    
    void configure(const MetricsConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = config;
        
        if (config_.file_output) {
            output_file_.open(config_.output_file, std::ios::out | std::ios::trunc);
            output_file_ << "timestamp_ms,channel,transport,msg_sent,msg_recv,"
                        << "bytes_sent,bytes_recv,msg_rate_send,msg_rate_recv,"
                        << "bandwidth_send_mbps,bandwidth_recv_mbps,"
                        << "latency_p50_us,latency_p99_us,bytes_used,bytes_available\n";
        }
    }
    
    void start() {
        if (!config_.enabled) return;
        
        running_ = true;
        collector_thread_ = std::thread(&MetricsCollector::collection_loop, this);
        
        if (config_.live_dashboard) {
            dashboard_thread_ = std::thread(&MetricsCollector::dashboard_loop, this);
        }
    }
    
    void stop() {
        running_ = false;
        cv_.notify_all();
        
        if (collector_thread_.joinable()) {
            collector_thread_.join();
        }
        
        if (dashboard_thread_.joinable()) {
            dashboard_thread_.join();
        }
        
        if (output_file_.is_open()) {
            output_file_.close();
        }
    }
    
    void register_channel(const std::string& name, ChannelTransport transport, 
                         size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& metrics = channels_[name];
        metrics.name = name;
        metrics.transport = transport;
        metrics.capacity_bytes = capacity;
        metrics.last_sample_time = std::chrono::steady_clock::now();
        
        if (config_.event_tracing) {
            add_event(EventType::CHANNEL_CREATE, name, 0, 0, "");
        }
    }
    
    void unregister_channel(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (config_.event_tracing) {
            add_event(EventType::CHANNEL_DESTROY, name, 0, 0, "");
        }
        
        channels_.erase(name);
    }
    
    void record_send(const std::string& channel, size_t size, uint64_t sequence) {
        auto it = channels_.find(channel);
        if (it != channels_.end()) {
            it->second.messages_sent++;
            it->second.bytes_sent += size;
            
            if (config_.event_tracing) {
                add_event(EventType::MESSAGE_COMMIT, channel, sequence, size, "");
            }
        }
    }
    
    void record_receive(const std::string& channel, size_t size, uint64_t sequence,
                       uint64_t latency_ns) {
        auto it = channels_.find(channel);
        if (it != channels_.end()) {
            it->second.messages_received++;
            it->second.bytes_received += size;
            it->second.latency_histogram.record(latency_ns);
            
            if (config_.event_tracing) {
                add_event(EventType::MESSAGE_RECEIVE, channel, sequence, size,
                         "latency_ns=" + std::to_string(latency_ns));
            }
        }
    }
    
    void record_allocation_failure(const std::string& channel, size_t requested_size) {
        auto it = channels_.find(channel);
        if (it != channels_.end()) {
            it->second.allocation_failures++;
            
            if (config_.event_tracing) {
                add_event(EventType::BUFFER_FULL, channel, 0, requested_size, "");
            }
        }
    }
    
    void update_buffer_state(const std::string& channel, size_t used, size_t available) {
        auto it = channels_.find(channel);
        if (it != channels_.end()) {
            it->second.bytes_used = used;
            it->second.bytes_available = available;
        }
    }
    
    void record_connection_state(const std::string& channel, bool connected) {
        auto it = channels_.find(channel);
        if (it != channels_.end()) {
            it->second.is_connected = connected;
            
            if (config_.event_tracing) {
                add_event(connected ? EventType::CONNECTION_ESTABLISHED 
                                   : EventType::CONNECTION_LOST,
                         channel, 0, 0, "");
            }
        }
    }
    
    // Memory tracking
    void record_allocation(const std::string& channel, size_t size) {
        if (!config_.memory_tracking) return;
        
        auto it = channels_.find(channel);
        if (it != channels_.end()) {
            it->second.current_allocations++;
            it->second.total_allocation_size += size;
            
            size_t current = it->second.current_allocations.load();
            size_t peak = it->second.peak_allocations.load();
            while (current > peak && 
                   !it->second.peak_allocations.compare_exchange_weak(peak, current));
        }
    }
    
    void record_deallocation(const std::string& channel, size_t size) {
        if (!config_.memory_tracking) return;
        
        auto it = channels_.find(channel);
        if (it != channels_.end()) {
            it->second.current_allocations--;
            it->second.total_allocation_size -= size;
        }
    }
    
    // Get snapshot of current metrics
    std::map<std::string, ChannelMetrics> get_snapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        return channels_;
    }
    
    // Dump events for analysis
    std::vector<TracedEvent> get_events() {
        std::lock_guard<std::mutex> lock(event_mutex_);
        return events_;
    }
    
private:
    MetricsCollector() = default;
    ~MetricsCollector() { stop(); }
    
    MetricsCollector(const MetricsCollector&) = delete;
    MetricsCollector& operator=(const MetricsCollector&) = delete;
    
    void collection_loop() {
        while (running_) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(config_.sampling_interval_ms),
                            [this] { return !running_; });
            }
            
            if (!running_) break;
            
            collect_and_report();
        }
    }
    
    void collect_and_report() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto now = std::chrono::steady_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        
        for (auto& [name, metrics] : channels_) {
            // Calculate rates
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - metrics.last_sample_time).count();
            
            if (duration > 0) {
                double duration_sec = duration / 1000.0;
                
                uint64_t msg_sent = metrics.messages_sent.load();
                uint64_t msg_recv = metrics.messages_received.load();
                uint64_t bytes_sent = metrics.bytes_sent.load();
                uint64_t bytes_recv = metrics.bytes_received.load();
                
                double msg_rate_send = (msg_sent - metrics.last_messages_sent) / duration_sec;
                double msg_rate_recv = (msg_recv - metrics.last_messages_received) / duration_sec;
                double bandwidth_send = (bytes_sent - metrics.last_bytes_sent) / 1024.0 / 1024.0 / duration_sec;
                double bandwidth_recv = (bytes_recv - metrics.last_bytes_received) / 1024.0 / 1024.0 / duration_sec;
                
                metrics.last_messages_sent = msg_sent;
                metrics.last_messages_received = msg_recv;
                metrics.last_bytes_sent = bytes_sent;
                metrics.last_bytes_received = bytes_recv;
                metrics.last_sample_time = now;
                
                // Get latency percentiles
                auto percentiles = metrics.latency_histogram.get_percentiles();
                
                // Output to file
                if (config_.file_output && output_file_.is_open()) {
                    output_file_ << timestamp << ","
                                << name << ","
                                << static_cast<int>(metrics.transport) << ","
                                << msg_sent << ","
                                << msg_recv << ","
                                << bytes_sent << ","
                                << bytes_recv << ","
                                << std::fixed << std::setprecision(2)
                                << msg_rate_send << ","
                                << msg_rate_recv << ","
                                << bandwidth_send << ","
                                << bandwidth_recv << ","
                                << percentiles.p50 / 1000.0 << ","
                                << percentiles.p99 / 1000.0 << ","
                                << metrics.bytes_used.load() << ","
                                << metrics.bytes_available.load() << "\n";
                    output_file_.flush();
                }
                
                // Output to console
                if (config_.console_output) {
                    std::cout << "[" << name << "] "
                             << "Rate: " << std::fixed << std::setprecision(0)
                             << msg_rate_send << "/" << msg_rate_recv << " msg/s, "
                             << std::setprecision(2)
                             << bandwidth_send << "/" << bandwidth_recv << " MB/s, "
                             << "Lat(µs): " << percentiles.p50 / 1000.0
                             << "/" << percentiles.p99 / 1000.0 << " (p50/p99)"
                             << std::endl;
                }
            }
        }
    }
    
    void dashboard_loop() {
        while (running_) {
            // Clear screen (ANSI escape codes)
            std::cout << "\033[2J\033[H";
            
            std::cout << "=== Psyne Metrics Dashboard ===" << std::endl;
            std::cout << "Time: " << std::chrono::system_clock::now() << std::endl;
            std::cout << std::endl;
            
            {
                std::lock_guard<std::mutex> lock(mutex_);
                
                // Header
                std::cout << std::left
                         << std::setw(20) << "Channel"
                         << std::setw(10) << "Transport"
                         << std::setw(15) << "Send Rate"
                         << std::setw(15) << "Recv Rate"
                         << std::setw(15) << "Bandwidth"
                         << std::setw(20) << "Latency (p50/p99)"
                         << std::setw(15) << "Buffer Usage"
                         << std::endl;
                
                std::cout << std::string(110, '-') << std::endl;
                
                // Channel data
                for (const auto& [name, metrics] : channels_) {
                    auto now = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                        now - metrics.last_sample_time).count();
                    
                    if (duration > 0) {
                        double msg_rate_send = (metrics.messages_sent - metrics.last_messages_sent) / duration;
                        double msg_rate_recv = (metrics.messages_received - metrics.last_messages_received) / duration;
                        double bandwidth = (metrics.bytes_sent - metrics.last_bytes_sent) / 1024.0 / 1024.0 / duration;
                        
                        auto percentiles = metrics.latency_histogram.get_percentiles();
                        double buffer_usage = 100.0 * metrics.bytes_used / 
                                            (metrics.bytes_used + metrics.bytes_available);
                        
                        std::cout << std::left
                                 << std::setw(20) << name
                                 << std::setw(10) << transport_to_string(metrics.transport)
                                 << std::setw(15) << std::fixed << std::setprecision(0) 
                                 << msg_rate_send
                                 << std::setw(15) << msg_rate_recv
                                 << std::setw(15) << std::setprecision(2) << bandwidth
                                 << std::setw(20) << (std::to_string(percentiles.p50/1000) + "/" + 
                                                     std::to_string(percentiles.p99/1000) + "µs")
                                 << std::setw(15) << std::setprecision(1) << buffer_usage << "%"
                                 << std::endl;
                    }
                }
                
                // Summary statistics
                std::cout << std::endl;
                std::cout << "=== Summary ===" << std::endl;
                
                uint64_t total_messages = 0;
                uint64_t total_bytes = 0;
                for (const auto& [name, metrics] : channels_) {
                    total_messages += metrics.messages_sent + metrics.messages_received;
                    total_bytes += metrics.bytes_sent + metrics.bytes_received;
                }
                
                std::cout << "Total messages: " << total_messages << std::endl;
                std::cout << "Total data: " << total_bytes / 1024 / 1024 << " MB" << std::endl;
                
                if (config_.memory_tracking) {
                    std::cout << std::endl;
                    std::cout << "=== Memory Usage ===" << std::endl;
                    for (const auto& [name, metrics] : channels_) {
                        std::cout << name << ": " 
                                 << metrics.current_allocations << " allocations, "
                                 << metrics.total_allocation_size / 1024 << " KB"
                                 << " (peak: " << metrics.peak_allocations << ")"
                                 << std::endl;
                    }
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.sampling_interval_ms));
        }
    }
    
    void add_event(EventType type, const std::string& channel, uint64_t sequence,
                  size_t size, const std::string& extra) {
        if (!config_.event_tracing) return;
        
        std::lock_guard<std::mutex> lock(event_mutex_);
        
        if (events_.size() >= config_.event_buffer_size) {
            events_.erase(events_.begin()); // Remove oldest
        }
        
        TracedEvent event;
        event.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        event.type = type;
        event.channel_name = channel;
        event.sequence = sequence;
        event.size = size;
        event.thread_id = std::this_thread::get_id();
        event.extra_info = extra;
        
        events_.push_back(event);
    }
    
    std::string transport_to_string(ChannelTransport transport) {
        switch (transport) {
            case ChannelTransport::IN_PROCESS: return "InProc";
            case ChannelTransport::IPC: return "IPC";
            case ChannelTransport::TCP: return "TCP";
            case ChannelTransport::UDP: return "UDP";
            case ChannelTransport::RDMA: return "RDMA";
            default: return "Unknown";
        }
    }
    
    MetricsConfig config_;
    std::map<std::string, ChannelMetrics> channels_;
    std::vector<TracedEvent> events_;
    
    std::mutex mutex_;
    std::mutex event_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
    
    std::thread collector_thread_;
    std::thread dashboard_thread_;
    std::ofstream output_file_;
};

// Convenience macros for metrics collection
#ifdef PSYNE_DEBUG_METRICS
#define PSYNE_METRICS_REGISTER_CHANNEL(name, transport, capacity) \
    psyne::debug::MetricsCollector::instance().register_channel(name, transport, capacity)
#define PSYNE_METRICS_RECORD_SEND(channel, size, seq) \
    psyne::debug::MetricsCollector::instance().record_send(channel, size, seq)
#define PSYNE_METRICS_RECORD_RECEIVE(channel, size, seq, latency) \
    psyne::debug::MetricsCollector::instance().record_receive(channel, size, seq, latency)
#define PSYNE_METRICS_UPDATE_BUFFER(channel, used, available) \
    psyne::debug::MetricsCollector::instance().update_buffer_state(channel, used, available)
#else
#define PSYNE_METRICS_REGISTER_CHANNEL(name, transport, capacity)
#define PSYNE_METRICS_RECORD_SEND(channel, size, seq)
#define PSYNE_METRICS_RECORD_RECEIVE(channel, size, seq, latency)
#define PSYNE_METRICS_UPDATE_BUFFER(channel, used, available)
#endif

} // namespace debug
} // namespace psyne