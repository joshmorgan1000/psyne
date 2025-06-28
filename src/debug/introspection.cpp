#include "introspection.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <iostream>

namespace psyne {
namespace debug {

// Static member definitions
std::map<std::string, std::vector<MessageTracer::TraceEvent>> MessageTracer::traces_;
std::map<std::string, bool> MessageTracer::tracing_enabled_;
std::map<std::string, std::shared_ptr<Channel>> DebugConsole::registered_channels_;
std::map<std::string, std::function<std::string(const std::vector<std::string>&)>> DebugConsole::commands_;

// BufferUsage implementation
std::string BufferUsage::visual_bar() const {
    const size_t bar_width = 40;
    size_t filled = static_cast<size_t>((utilization_percent / 100.0) * bar_width);
    
    std::string bar = "[";
    for (size_t i = 0; i < bar_width; ++i) {
        if (i < filled) {
            bar += "█";
        } else {
            bar += "░";
        }
    }
    bar += "] " + std::to_string(static_cast<int>(utilization_percent)) + "%";
    return bar;
}

// ChannelDiagnostics implementation
std::string ChannelDiagnostics::summary() const {
    std::ostringstream oss;
    oss << "Channel: " << uri << " [";
    
    switch (health) {
        case ChannelHealth::Healthy: oss << "HEALTHY"; break;
        case ChannelHealth::Warning: oss << "WARNING"; break;
        case ChannelHealth::Critical: oss << "CRITICAL"; break;
        case ChannelHealth::Disconnected: oss << "DISCONNECTED"; break;
        case ChannelHealth::Unknown: oss << "UNKNOWN"; break;
    }
    
    oss << "] Buffer: " << std::fixed << std::setprecision(1) 
        << buffer_usage.utilization_percent << "%";
    
    if (has_errors()) {
        oss << " (" << errors.size() << " errors)";
    }
    if (has_warnings()) {
        oss << " (" << warnings.size() << " warnings)";
    }
    
    return oss.str();
}

// ChannelInspector implementation
ChannelDiagnostics ChannelInspector::inspect(const Channel& channel) {
    ChannelDiagnostics diag;
    diag.uri = channel.uri();
    diag.type = channel.type();
    diag.mode = channel.mode();
    diag.buffer_usage = get_buffer_usage(channel);
    diag.health = analyze_health(channel);
    diag.last_activity = std::chrono::steady_clock::now();
    
    if (channel.has_metrics()) {
        diag.current_metrics = channel.get_metrics();
    }
    
    // Add warnings and errors based on analysis
    auto health_issues = health_check(channel);
    for (const auto& issue : health_issues) {
        if (issue.find("WARNING") != std::string::npos) {
            diag.warnings.push_back(issue);
        } else if (issue.find("ERROR") != std::string::npos) {
            diag.errors.push_back(issue);
        }
    }
    
    return diag;
}

BufferUsage ChannelInspector::get_buffer_usage(const Channel& channel) {
    BufferUsage usage;
    
    // For now, provide mock data since we don't have direct buffer access
    // In a real implementation, this would access the ring buffer state
    usage.total_size = 1024 * 1024; // 1MB default
    
    if (channel.has_metrics()) {
        auto metrics = channel.get_metrics();
        // Estimate usage based on message activity
        usage.used_bytes = std::min(static_cast<size_t>(metrics.bytes_sent * 0.1), usage.total_size);
        usage.available_bytes = usage.total_size - usage.used_bytes;
        usage.utilization_percent = (static_cast<double>(usage.used_bytes) / usage.total_size) * 100.0;
    }
    
    return usage;
}

ChannelHealth ChannelInspector::analyze_health(const Channel& channel) {
    if (channel.is_stopped()) {
        return ChannelHealth::Disconnected;
    }
    
    if (channel.has_metrics()) {
        auto metrics = channel.get_metrics();
        auto buffer_usage = get_buffer_usage(channel);
        
        // Critical: Buffer over 90% full
        if (buffer_usage.utilization_percent > 90.0) {
            return ChannelHealth::Critical;
        }
        
        // Warning: Many send blocks indicate congestion
        if (metrics.send_blocks > metrics.messages_sent * 0.1) {
            return ChannelHealth::Warning;
        }
        
        // Warning: Buffer over 75% full
        if (buffer_usage.utilization_percent > 75.0) {
            return ChannelHealth::Warning;
        }
    }
    
    return ChannelHealth::Healthy;
}

std::string ChannelInspector::get_status_report(const Channel& channel) {
    auto diag = inspect(channel);
    std::ostringstream report;
    
    report << "=== Channel Status Report ===" << std::endl;
    report << "URI: " << diag.uri << std::endl;
    report << "Type: " << (diag.type == ChannelType::SingleType ? "SingleType" : "MultiType") << std::endl;
    report << "Mode: ";
    switch (diag.mode) {
        case ChannelMode::SPSC: report << "SPSC"; break;
        case ChannelMode::SPMC: report << "SPMC"; break;
        case ChannelMode::MPSC: report << "MPSC"; break;
        case ChannelMode::MPMC: report << "MPMC"; break;
    }
    report << std::endl;
    
    report << "Health: ";
    switch (diag.health) {
        case ChannelHealth::Healthy: report << "🟢 HEALTHY"; break;
        case ChannelHealth::Warning: report << "🟡 WARNING"; break;
        case ChannelHealth::Critical: report << "🔴 CRITICAL"; break;
        case ChannelHealth::Disconnected: report << "⚫ DISCONNECTED"; break;
        case ChannelHealth::Unknown: report << "❓ UNKNOWN"; break;
    }
    report << std::endl << std::endl;
    
    report << "Buffer Usage:" << std::endl;
    report << "  " << diag.buffer_usage.visual_bar() << std::endl;
    report << "  Used: " << diag.buffer_usage.used_bytes << " bytes" << std::endl;
    report << "  Available: " << diag.buffer_usage.available_bytes << " bytes" << std::endl;
    report << std::endl;
    
    if (channel.has_metrics()) {
        report << "Metrics:" << std::endl;
        report << "  Messages sent: " << diag.current_metrics.messages_sent << std::endl;
        report << "  Messages received: " << diag.current_metrics.messages_received << std::endl;
        report << "  Bytes sent: " << diag.current_metrics.bytes_sent << std::endl;
        report << "  Bytes received: " << diag.current_metrics.bytes_received << std::endl;
        report << "  Send blocks: " << diag.current_metrics.send_blocks << std::endl;
        report << "  Receive blocks: " << diag.current_metrics.receive_blocks << std::endl;
        report << std::endl;
    }
    
    if (diag.has_warnings()) {
        report << "Warnings:" << std::endl;
        for (const auto& warning : diag.warnings) {
            report << "  ⚠️  " << warning << std::endl;
        }
        report << std::endl;
    }
    
    if (diag.has_errors()) {
        report << "Errors:" << std::endl;
        for (const auto& error : diag.errors) {
            report << "  ❌ " << error << std::endl;
        }
        report << std::endl;
    }
    
    return report.str();
}

std::string ChannelInspector::visualize_buffer(const Channel& channel, size_t width) {
    auto usage = get_buffer_usage(channel);
    
    std::ostringstream vis;
    vis << "Buffer Visualization (1MB):" << std::endl;
    
    size_t filled = static_cast<size_t>((usage.utilization_percent / 100.0) * width);
    
    vis << "┌";
    for (size_t i = 0; i < width; ++i) vis << "─";
    vis << "┐" << std::endl;
    
    vis << "│";
    for (size_t i = 0; i < width; ++i) {
        if (i < filled) {
            vis << "█";
        } else {
            vis << " ";
        }
    }
    vis << "│ " << std::fixed << std::setprecision(1) << usage.utilization_percent << "%" << std::endl;
    
    vis << "└";
    for (size_t i = 0; i < width; ++i) vis << "─";
    vis << "┘" << std::endl;
    
    vis << "Used: " << usage.used_bytes << " bytes, Available: " << usage.available_bytes << " bytes";
    
    return vis.str();
}

std::vector<std::string> ChannelInspector::health_check(const Channel& channel) {
    std::vector<std::string> issues;
    
    if (channel.is_stopped()) {
        issues.push_back("ERROR: Channel is stopped");
        return issues;
    }
    
    auto buffer_usage = get_buffer_usage(channel);
    
    if (buffer_usage.utilization_percent > 95.0) {
        issues.push_back("ERROR: Buffer nearly full (" + 
                        std::to_string(static_cast<int>(buffer_usage.utilization_percent)) + "%)");
    } else if (buffer_usage.utilization_percent > 80.0) {
        issues.push_back("WARNING: High buffer usage (" + 
                        std::to_string(static_cast<int>(buffer_usage.utilization_percent)) + "%)");
    }
    
    if (channel.has_metrics()) {
        auto metrics = channel.get_metrics();
        
        if (metrics.send_blocks > 100) {
            issues.push_back("WARNING: High send blocking (" + 
                           std::to_string(metrics.send_blocks) + " blocks)");
        }
        
        if (metrics.receive_blocks > 100) {
            issues.push_back("WARNING: High receive blocking (" + 
                           std::to_string(metrics.receive_blocks) + " blocks)");
        }
        
        // Check for message rate issues
        if (metrics.messages_sent > 0 && metrics.send_blocks > metrics.messages_sent * 0.2) {
            issues.push_back("WARNING: Send success rate low - consider increasing buffer size");
        }
    }
    
    return issues;
}

// PerformanceProfiler implementation
PerformanceProfiler::PerformanceProfiler(const Channel& channel) : channel_(&channel) {}

void PerformanceProfiler::start() {
    start_time_ = std::chrono::steady_clock::now();
    profiling_ = true;
    last_snapshot_ = snapshot();
}

void PerformanceProfiler::stop() {
    profiling_ = false;
}

PerformanceSnapshot PerformanceProfiler::snapshot() {
    PerformanceSnapshot snap;
    snap.timestamp = std::chrono::steady_clock::now();
    
    if (channel_->has_metrics()) {
        snap.metrics = channel_->get_metrics();
        snap.buffer_usage = ChannelInspector::get_buffer_usage(*channel_);
        
        // Calculate rates since last snapshot
        if (!history_.empty()) {
            auto& last = history_.back();
            auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
                snap.timestamp - last.timestamp).count();
            
            if (time_diff > 0) {
                auto msg_diff = snap.metrics.messages_sent - last.metrics.messages_sent;
                auto byte_diff = snap.metrics.bytes_sent - last.metrics.bytes_sent;
                
                snap.messages_per_second = static_cast<double>(msg_diff) / time_diff;
                snap.bytes_per_second = static_cast<double>(byte_diff) / time_diff;
                
                if (msg_diff > 0) {
                    snap.average_message_size = static_cast<double>(byte_diff) / msg_diff;
                }
            }
        }
    }
    
    if (profiling_) {
        history_.push_back(snap);
        
        // Limit history size
        if (history_.size() > 1000) {
            history_.erase(history_.begin());
        }
    }
    
    return snap;
}

std::string PerformanceProfiler::generate_report() const {
    std::ostringstream report;
    report << "=== Performance Report ===" << std::endl;
    
    if (history_.empty()) {
        report << "No performance data available." << std::endl;
        return report.str();
    }
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        history_.back().timestamp - history_.front().timestamp).count();
    
    report << "Profiling duration: " << duration << " seconds" << std::endl;
    report << "Samples collected: " << history_.size() << std::endl;
    
    if (history_.size() > 1) {
        double max_msg_rate = 0, avg_msg_rate = 0;
        double max_byte_rate = 0, avg_byte_rate = 0;
        
        for (const auto& snap : history_) {
            max_msg_rate = std::max(max_msg_rate, snap.messages_per_second);
            avg_msg_rate += snap.messages_per_second;
            max_byte_rate = std::max(max_byte_rate, snap.bytes_per_second);
            avg_byte_rate += snap.bytes_per_second;
        }
        
        avg_msg_rate /= history_.size();
        avg_byte_rate /= history_.size();
        
        report << std::endl << "Performance Summary:" << std::endl;
        report << "  Peak message rate: " << std::fixed << std::setprecision(1) 
               << max_msg_rate << " msg/sec" << std::endl;
        report << "  Average message rate: " << avg_msg_rate << " msg/sec" << std::endl;
        report << "  Peak throughput: " << (max_byte_rate / 1024.0) << " KB/sec" << std::endl;
        report << "  Average throughput: " << (avg_byte_rate / 1024.0) << " KB/sec" << std::endl;
    }
    
    const auto& final = history_.back();
    report << std::endl << "Final Metrics:" << std::endl;
    report << "  Total messages sent: " << final.metrics.messages_sent << std::endl;
    report << "  Total bytes sent: " << final.metrics.bytes_sent << std::endl;
    report << "  Buffer utilization: " << std::fixed << std::setprecision(1) 
           << final.buffer_usage.utilization_percent << "%" << std::endl;
    
    return report.str();
}

// MessageTracer implementation
void MessageTracer::enable_tracing(const std::string& channel_uri) {
    tracing_enabled_[channel_uri] = true;
}

void MessageTracer::disable_tracing(const std::string& channel_uri) {
    tracing_enabled_[channel_uri] = false;
}

void MessageTracer::trace_event(const std::string& channel_uri, 
                                const std::string& event_type,
                                size_t message_size,
                                uint32_t message_type,
                                const std::string& details) {
    if (tracing_enabled_[channel_uri]) {
        TraceEvent event;
        event.timestamp = std::chrono::steady_clock::now();
        event.channel_uri = channel_uri;
        event.event_type = event_type;
        event.message_size = message_size;
        event.message_type = message_type;
        event.details = details;
        
        traces_[channel_uri].push_back(event);
        
        // Limit trace history
        if (traces_[channel_uri].size() > 10000) {
            traces_[channel_uri].erase(traces_[channel_uri].begin());
        }
    }
}

std::vector<MessageTracer::TraceEvent> MessageTracer::get_trace_history(const std::string& channel_uri) {
    return traces_[channel_uri];
}

void MessageTracer::clear_traces(const std::string& channel_uri) {
    if (channel_uri.empty()) {
        traces_.clear();
    } else {
        traces_[channel_uri].clear();
    }
}

std::string MessageTracer::generate_trace_report(const std::string& channel_uri) {
    std::ostringstream report;
    report << "=== Message Trace Report ===" << std::endl;
    report << "Channel: " << channel_uri << std::endl;
    
    auto& trace = traces_[channel_uri];
    if (trace.empty()) {
        report << "No trace events recorded." << std::endl;
        return report.str();
    }
    
    report << "Events: " << trace.size() << std::endl << std::endl;
    
    // Show recent events
    size_t start = trace.size() > 20 ? trace.size() - 20 : 0;
    for (size_t i = start; i < trace.size(); ++i) {
        const auto& event = trace[i];
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            event.timestamp.time_since_epoch()).count();
        
        report << "[" << ms << "ms] " << event.event_type 
               << " size=" << event.message_size << " type=" << event.message_type;
        if (!event.details.empty()) {
            report << " (" << event.details << ")";
        }
        report << std::endl;
    }
    
    return report.str();
}

// Debug utility functions
void print_diagnostics(const ChannelDiagnostics& diag, bool verbose) {
    std::cout << diag.summary() << std::endl;
    
    if (verbose) {
        std::cout << "  Buffer: " << diag.buffer_usage.visual_bar() << std::endl;
        
        if (diag.has_warnings()) {
            for (const auto& warning : diag.warnings) {
                std::cout << "  ⚠️  " << warning << std::endl;
            }
        }
        
        if (diag.has_errors()) {
            for (const auto& error : diag.errors) {
                std::cout << "  ❌ " << error << std::endl;
            }
        }
    }
}

std::string create_dashboard() {
    std::ostringstream dashboard;
    dashboard << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    dashboard << "║                    PSYNE DEBUG DASHBOARD                     ║" << std::endl;
    dashboard << "╠══════════════════════════════════════════════════════════════╣" << std::endl;
    dashboard << "║ Debug utilities active and ready for channel monitoring      ║" << std::endl;
    dashboard << "║                                                              ║" << std::endl;
    dashboard << "║ Available Tools:                                             ║" << std::endl;
    dashboard << "║   • Channel Inspector                                        ║" << std::endl;
    dashboard << "║   • Performance Profiler                                     ║" << std::endl;
    dashboard << "║   • Message Tracer                                           ║" << std::endl;
    dashboard << "║   • Health Monitor                                           ║" << std::endl;
    dashboard << "║   • Buffer Visualizer                                        ║" << std::endl;
    dashboard << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    return dashboard.str();
}

// ChannelMonitor implementation
void ChannelMonitor::add_channel(const std::string& name, std::shared_ptr<Channel> channel) {
    channels_[name] = channel;
}

void ChannelMonitor::start_monitoring(std::chrono::milliseconds interval) {
    if (monitoring_) return;
    
    monitoring_ = true;
    monitor_thread_ = std::thread([this, interval]() { monitor_loop(interval); });
}

void ChannelMonitor::stop_monitoring() {
    monitoring_ = false;
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

void ChannelMonitor::set_alert_callback(
    std::function<void(const std::string&, ChannelHealth, const std::string&)> callback) {
    alert_callback_ = callback;
}

std::string ChannelMonitor::get_status_summary() {
    std::ostringstream summary;
    summary << "Channel Monitor Status:" << std::endl;
    
    for (const auto& [name, channel] : channels_) {
        auto health = ChannelInspector::analyze_health(*channel);
        summary << "  " << name << ": ";
        
        switch (health) {
            case ChannelHealth::Healthy: summary << "🟢 HEALTHY"; break;
            case ChannelHealth::Warning: summary << "🟡 WARNING"; break;
            case ChannelHealth::Critical: summary << "🔴 CRITICAL"; break;
            case ChannelHealth::Disconnected: summary << "⚫ DISCONNECTED"; break;
            case ChannelHealth::Unknown: summary << "❓ UNKNOWN"; break;
        }
        
        summary << std::endl;
    }
    
    return summary.str();
}

void ChannelMonitor::monitor_loop(std::chrono::milliseconds interval) {
    while (monitoring_) {
        for (const auto& [name, channel] : channels_) {
            auto health = ChannelInspector::analyze_health(*channel);
            
            if (health != ChannelHealth::Healthy && alert_callback_) {
                auto issues = ChannelInspector::health_check(*channel);
                std::string issue_summary;
                for (const auto& issue : issues) {
                    issue_summary += issue + "; ";
                }
                
                alert_callback_(name, health, issue_summary);
            }
        }
        
        std::this_thread::sleep_for(interval);
    }
}

} // namespace debug
} // namespace psyne