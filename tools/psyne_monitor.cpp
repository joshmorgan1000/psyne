/**
 * @file psyne_monitor.cpp
 * @brief Real-time monitoring tool for Psyne channels
 *
 * Provides live monitoring of Psyne channel performance with:
 * - Terminal-based dashboard
 * - Performance graphs
 * - Alert thresholds
 * - CSV export
 */

#include "psyne/psyne.hpp"
#include "psyne/debug/metrics_collector.hpp"
#include <algorithm>
#include <csignal>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>
#include <vector>

using namespace psyne;
using namespace psyne::debug;

// Global flag for clean shutdown
std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    g_running = false;
}

// Terminal colors and controls
namespace term {
    const std::string CLEAR = "\033[2J\033[H";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    
    void move_cursor(int row, int col) {
        std::cout << "\033[" << row << ";" << col << "H";
    }
    
    std::string color_value(double value, double warn_threshold, double error_threshold,
                           bool inverse = false) {
        if (inverse) {
            if (value < warn_threshold) return GREEN;
            if (value < error_threshold) return YELLOW;
            return RED;
        } else {
            if (value > error_threshold) return RED;
            if (value > warn_threshold) return YELLOW;
            return GREEN;
        }
    }
}

// Time series data for graphing
template<typename T>
class TimeSeries {
public:
    TimeSeries(size_t max_points = 60) : max_points_(max_points) {}
    
    void add(T value) {
        if (data_.size() >= max_points_) {
            data_.pop_front();
        }
        data_.push_back(value);
    }
    
    std::vector<T> get_data() const {
        return std::vector<T>(data_.begin(), data_.end());
    }
    
    T get_max() const {
        if (data_.empty()) return T{};
        return *std::max_element(data_.begin(), data_.end());
    }
    
    T get_min() const {
        if (data_.empty()) return T{};
        return *std::min_element(data_.begin(), data_.end());
    }
    
    T get_latest() const {
        return data_.empty() ? T{} : data_.back();
    }
    
private:
    size_t max_points_;
    std::deque<T> data_;
};

// ASCII graph renderer
class AsciiGraph {
public:
    static std::string render(const std::vector<double>& data, 
                            int width = 50, int height = 10,
                            const std::string& label = "") {
        if (data.empty()) return "";
        
        std::stringstream ss;
        
        // Find min/max
        double min_val = *std::min_element(data.begin(), data.end());
        double max_val = *std::max_element(data.begin(), data.end());
        double range = max_val - min_val;
        if (range == 0) range = 1;
        
        // Create graph
        std::vector<std::string> lines(height);
        for (int i = 0; i < height; ++i) {
            lines[i] = std::string(width, ' ');
        }
        
        // Plot points
        int step = std::max(1, static_cast<int>(data.size() / width));
        for (int x = 0; x < width && x * step < data.size(); ++x) {
            double value = data[x * step];
            int y = height - 1 - static_cast<int>((value - min_val) / range * (height - 1));
            if (y >= 0 && y < height) {
                lines[y][x] = '*';
            }
        }
        
        // Add axes and labels
        ss << label << " (max: " << std::fixed << std::setprecision(2) 
           << max_val << ")" << std::endl;
        
        for (const auto& line : lines) {
            ss << "|" << line << std::endl;
        }
        ss << "+" << std::string(width, '-') << std::endl;
        
        return ss.str();
    }
};

// Channel monitor data
struct ChannelMonitor {
    std::string name;
    TimeSeries<double> send_rate;
    TimeSeries<double> recv_rate;
    TimeSeries<double> bandwidth;
    TimeSeries<double> latency_p50;
    TimeSeries<double> latency_p99;
    TimeSeries<double> buffer_usage;
    
    uint64_t total_errors = 0;
    uint64_t allocation_failures = 0;
    
    void update(const ChannelMetrics& metrics, const ChannelMetrics& prev, double dt) {
        if (dt > 0) {
            double msg_send_rate = (metrics.messages_sent - prev.messages_sent) / dt;
            double msg_recv_rate = (metrics.messages_received - prev.messages_received) / dt;
            double bw = (metrics.bytes_sent - prev.bytes_sent + 
                        metrics.bytes_received - prev.bytes_received) / 1024.0 / 1024.0 / dt;
            
            send_rate.add(msg_send_rate);
            recv_rate.add(msg_recv_rate);
            bandwidth.add(bw);
            
            auto percentiles = metrics.latency_histogram.get_percentiles();
            latency_p50.add(percentiles.p50 / 1000.0); // Convert to microseconds
            latency_p99.add(percentiles.p99 / 1000.0);
            
            double usage = 100.0 * metrics.bytes_used / 
                          (metrics.bytes_used + metrics.bytes_available + 1);
            buffer_usage.add(usage);
            
            allocation_failures = metrics.allocation_failures;
        }
    }
};

// Alert system
struct Alert {
    enum Level { INFO, WARNING, ERROR };
    Level level;
    std::string channel;
    std::string message;
    std::chrono::steady_clock::time_point timestamp;
};

class AlertManager {
public:
    void check_metrics(const std::string& channel, const ChannelMonitor& monitor) {
        // Check for high latency
        if (monitor.latency_p99.get_latest() > 1000) { // 1ms
            add_alert(Alert::WARNING, channel, 
                     "High latency: " + std::to_string(monitor.latency_p99.get_latest()) + "µs");
        }
        
        // Check for buffer usage
        if (monitor.buffer_usage.get_latest() > 90) {
            add_alert(Alert::ERROR, channel, 
                     "Buffer critically full: " + 
                     std::to_string(static_cast<int>(monitor.buffer_usage.get_latest())) + "%");
        } else if (monitor.buffer_usage.get_latest() > 75) {
            add_alert(Alert::WARNING, channel, 
                     "Buffer usage high: " + 
                     std::to_string(static_cast<int>(monitor.buffer_usage.get_latest())) + "%");
        }
        
        // Check for allocation failures
        if (monitor.allocation_failures > 0) {
            add_alert(Alert::ERROR, channel, 
                     "Allocation failures: " + std::to_string(monitor.allocation_failures));
        }
        
        // Check for rate imbalance
        double send_rate = monitor.send_rate.get_latest();
        double recv_rate = monitor.recv_rate.get_latest();
        if (send_rate > 0 && recv_rate > 0) {
            double imbalance = std::abs(send_rate - recv_rate) / std::max(send_rate, recv_rate);
            if (imbalance > 0.2) {
                add_alert(Alert::WARNING, channel, 
                         "Rate imbalance: send=" + std::to_string(static_cast<int>(send_rate)) +
                         " recv=" + std::to_string(static_cast<int>(recv_rate)));
            }
        }
    }
    
    std::vector<Alert> get_recent_alerts(size_t count = 5) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Remove old alerts (older than 1 minute)
        auto now = std::chrono::steady_clock::now();
        alerts_.erase(
            std::remove_if(alerts_.begin(), alerts_.end(),
                          [now](const Alert& a) {
                              return now - a.timestamp > std::chrono::minutes(1);
                          }),
            alerts_.end()
        );
        
        // Return most recent
        size_t start = alerts_.size() > count ? alerts_.size() - count : 0;
        return std::vector<Alert>(alerts_.begin() + start, alerts_.end());
    }
    
private:
    void add_alert(Alert::Level level, const std::string& channel, 
                  const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Avoid duplicate alerts
        for (const auto& existing : alerts_) {
            if (existing.channel == channel && existing.message == message &&
                std::chrono::steady_clock::now() - existing.timestamp < std::chrono::seconds(10)) {
                return;
            }
        }
        
        alerts_.push_back({level, channel, message, std::chrono::steady_clock::now()});
    }
    
    std::vector<Alert> alerts_;
    mutable std::mutex mutex_;
};

// Main monitoring application
class PsyneMonitor {
public:
    PsyneMonitor(bool enable_graphs = true, bool enable_csv = false,
                 const std::string& csv_file = "psyne_monitor.csv")
        : enable_graphs_(enable_graphs), enable_csv_(enable_csv), csv_file_(csv_file) {
        
        if (enable_csv_) {
            csv_output_.open(csv_file_, std::ios::out | std::ios::trunc);
            csv_output_ << "timestamp,channel,send_rate,recv_rate,bandwidth_mbps,"
                       << "latency_p50_us,latency_p99_us,buffer_usage_pct\n";
        }
    }
    
    void run() {
        // Configure metrics collector
        MetricsConfig config;
        config.enabled = true;
        config.console_output = false;
        config.file_output = false;
        config.sampling_interval_ms = 1000;
        config.detailed_histograms = true;
        config.memory_tracking = true;
        config.event_tracing = false;
        
        MetricsCollector::instance().configure(config);
        MetricsCollector::instance().start();
        
        // Main monitoring loop
        auto last_update = std::chrono::steady_clock::now();
        std::map<std::string, ChannelMetrics> prev_metrics;
        
        while (g_running) {
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - last_update).count();
            
            // Get current metrics
            auto current_metrics = MetricsCollector::instance().get_snapshot();
            
            // Update monitors
            for (const auto& [name, metrics] : current_metrics) {
                auto& monitor = monitors_[name];
                monitor.name = name;
                
                if (prev_metrics.count(name) > 0) {
                    monitor.update(metrics, prev_metrics[name], dt);
                    alert_manager_.check_metrics(name, monitor);
                }
                
                // CSV output
                if (enable_csv_ && csv_output_.is_open()) {
                    csv_output_ << std::chrono::system_clock::now() << ","
                               << name << ","
                               << monitor.send_rate.get_latest() << ","
                               << monitor.recv_rate.get_latest() << ","
                               << monitor.bandwidth.get_latest() << ","
                               << monitor.latency_p50.get_latest() << ","
                               << monitor.latency_p99.get_latest() << ","
                               << monitor.buffer_usage.get_latest() << "\n";
                    csv_output_.flush();
                }
            }
            
            // Render dashboard
            render_dashboard();
            
            prev_metrics = current_metrics;
            last_update = now;
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        MetricsCollector::instance().stop();
    }
    
private:
    void render_dashboard() {
        std::cout << term::CLEAR;
        
        // Header
        std::cout << term::BOLD << term::CYAN 
                  << "=== Psyne Channel Monitor ===" 
                  << term::RESET << std::endl;
        std::cout << "Time: " << std::chrono::system_clock::now() << std::endl;
        std::cout << "Press Ctrl+C to exit" << std::endl;
        std::cout << std::endl;
        
        // Channel summary table
        render_summary_table();
        
        // Alerts
        render_alerts();
        
        // Graphs (if enabled)
        if (enable_graphs_ && !monitors_.empty()) {
            render_graphs();
        }
    }
    
    void render_summary_table() {
        std::cout << term::BOLD << "Channel Summary:" << term::RESET << std::endl;
        std::cout << std::left
                  << std::setw(20) << "Channel"
                  << std::setw(12) << "Send Rate"
                  << std::setw(12) << "Recv Rate"
                  << std::setw(12) << "Bandwidth"
                  << std::setw(15) << "Latency (µs)"
                  << std::setw(12) << "Buffer %"
                  << std::setw(10) << "Errors"
                  << std::endl;
        
        std::cout << std::string(95, '-') << std::endl;
        
        for (const auto& [name, monitor] : monitors_) {
            double send = monitor.send_rate.get_latest();
            double recv = monitor.recv_rate.get_latest();
            double bw = monitor.bandwidth.get_latest();
            double lat_p50 = monitor.latency_p50.get_latest();
            double lat_p99 = monitor.latency_p99.get_latest();
            double buf = monitor.buffer_usage.get_latest();
            
            std::cout << std::left << std::setw(20) << name;
            
            // Send rate with color
            std::cout << term::color_value(send, 1000000, 5000000) 
                      << std::setw(12) << format_rate(send) << term::RESET;
            
            // Recv rate
            std::cout << term::color_value(recv, 1000000, 5000000)
                      << std::setw(12) << format_rate(recv) << term::RESET;
            
            // Bandwidth
            std::cout << std::setw(12) << format_bandwidth(bw);
            
            // Latency
            std::cout << term::color_value(lat_p99, 100, 1000, true)
                      << std::setw(15) << (std::to_string(static_cast<int>(lat_p50)) + "/" + 
                                          std::to_string(static_cast<int>(lat_p99)))
                      << term::RESET;
            
            // Buffer usage
            std::cout << term::color_value(buf, 75, 90)
                      << std::setw(12) << std::fixed << std::setprecision(1) 
                      << buf << "%" << term::RESET;
            
            // Errors
            if (monitor.allocation_failures > 0) {
                std::cout << term::RED << std::setw(10) 
                          << monitor.allocation_failures << term::RESET;
            } else {
                std::cout << term::GREEN << std::setw(10) << "0" << term::RESET;
            }
            
            std::cout << std::endl;
        }
    }
    
    void render_alerts() {
        auto alerts = alert_manager_.get_recent_alerts();
        if (!alerts.empty()) {
            std::cout << std::endl;
            std::cout << term::BOLD << "Recent Alerts:" << term::RESET << std::endl;
            
            for (const auto& alert : alerts) {
                std::string color;
                std::string level_str;
                switch (alert.level) {
                    case Alert::INFO: 
                        color = term::CYAN; 
                        level_str = "[INFO]"; 
                        break;
                    case Alert::WARNING: 
                        color = term::YELLOW; 
                        level_str = "[WARN]"; 
                        break;
                    case Alert::ERROR: 
                        color = term::RED; 
                        level_str = "[ERROR]"; 
                        break;
                }
                
                auto age = std::chrono::steady_clock::now() - alert.timestamp;
                auto age_sec = std::chrono::duration_cast<std::chrono::seconds>(age).count();
                
                std::cout << color << level_str << " " << alert.channel 
                          << ": " << alert.message 
                          << " (" << age_sec << "s ago)"
                          << term::RESET << std::endl;
            }
        }
    }
    
    void render_graphs() {
        std::cout << std::endl;
        std::cout << term::BOLD << "Performance Graphs:" << term::RESET << std::endl;
        
        // Select most active channel
        std::string active_channel;
        double max_activity = 0;
        for (const auto& [name, monitor] : monitors_) {
            double activity = monitor.send_rate.get_latest() + monitor.recv_rate.get_latest();
            if (activity > max_activity) {
                max_activity = activity;
                active_channel = name;
            }
        }
        
        if (!active_channel.empty()) {
            const auto& monitor = monitors_[active_channel];
            
            // Message rate graph
            auto send_data = monitor.send_rate.get_data();
            std::cout << AsciiGraph::render(send_data, 60, 8, 
                                           active_channel + " Send Rate (msg/s)");
            
            // Latency graph
            auto latency_data = monitor.latency_p99.get_data();
            std::cout << AsciiGraph::render(latency_data, 60, 8,
                                           active_channel + " Latency p99 (µs)");
        }
    }
    
    std::string format_rate(double rate) {
        if (rate > 1000000) {
            return std::to_string(static_cast<int>(rate / 1000000)) + "M/s";
        } else if (rate > 1000) {
            return std::to_string(static_cast<int>(rate / 1000)) + "K/s";
        } else {
            return std::to_string(static_cast<int>(rate)) + "/s";
        }
    }
    
    std::string format_bandwidth(double mbps) {
        if (mbps > 1000) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << (mbps / 1000) << " GB/s";
            return ss.str();
        } else {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << mbps << " MB/s";
            return ss.str();
        }
    }
    
    std::map<std::string, ChannelMonitor> monitors_;
    AlertManager alert_manager_;
    bool enable_graphs_;
    bool enable_csv_;
    std::string csv_file_;
    std::ofstream csv_output_;
};

int main(int argc, char* argv[]) {
    // Install signal handler
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // Parse command line arguments
    bool enable_graphs = true;
    bool enable_csv = false;
    std::string csv_file = "psyne_monitor.csv";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-graphs") {
            enable_graphs = false;
        } else if (arg == "--csv" && i + 1 < argc) {
            enable_csv = true;
            csv_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --no-graphs      Disable performance graphs" << std::endl;
            std::cout << "  --csv <file>     Export metrics to CSV file" << std::endl;
            std::cout << "  --help           Show this help message" << std::endl;
            return 0;
        }
    }
    
    try {
        PsyneMonitor monitor(enable_graphs, enable_csv, csv_file);
        monitor.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}