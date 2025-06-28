#include <psyne/psyne.hpp>
#include "../src/debug/introspection.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>

using namespace psyne;

void test_channel_inspection() {
    std::cout << "=== Channel Inspection Demo ===" << std::endl;
    
    // Create a channel with metrics enabled
    auto channel = create_channel("memory://debug_test", 1024*1024, 
                                 ChannelMode::SPSC, ChannelType::SingleType, true);
    
    // Get initial status
    std::cout << "\nInitial Channel Status:" << std::endl;
    auto initial_diag = debug::ChannelInspector::inspect(*channel);
    debug::print_diagnostics(initial_diag, true);
    
    // Show buffer visualization
    std::cout << "\nBuffer Visualization:" << std::endl;
    std::cout << debug::ChannelInspector::visualize_buffer(*channel) << std::endl;
    
    // Send some messages to change state
    std::cout << "\nSending test messages..." << std::endl;
    for (int i = 0; i < 50; ++i) {
        FloatVector msg(*channel);
        msg.resize(100);
        for (size_t j = 0; j < msg.size(); ++j) {
            msg[j] = static_cast<float>(i * 100 + j);
        }
        channel->send(msg);
    }
    
    // Receive half the messages
    for (int i = 0; i < 25; ++i) {
        auto msg = channel->receive<FloatVector>();
        if (!msg) break;
    }
    
    // Check status after activity
    std::cout << "\nChannel Status After Activity:" << std::endl;
    auto active_diag = debug::ChannelInspector::inspect(*channel);
    debug::print_diagnostics(active_diag, true);
    
    // Show detailed status report
    std::cout << "\nDetailed Status Report:" << std::endl;
    std::cout << debug::ChannelInspector::get_status_report(*channel) << std::endl;
    
    // Run health check
    std::cout << "Health Check Results:" << std::endl;
    auto health_issues = debug::ChannelInspector::health_check(*channel);
    if (health_issues.empty()) {
        std::cout << "✅ No health issues detected" << std::endl;
    } else {
        for (const auto& issue : health_issues) {
            std::cout << "⚠️  " << issue << std::endl;
        }
    }
}

void test_performance_profiling() {
    std::cout << "\n=== Performance Profiling Demo ===" << std::endl;
    
    auto channel = create_channel("memory://perf_test", 1024*1024, 
                                 ChannelMode::SPSC, ChannelType::SingleType, true);
    
    // Start profiling
    debug::PerformanceProfiler profiler(*channel);
    profiler.start();
    
    std::cout << "Starting performance test..." << std::endl;
    
    // Simulate workload with timing
    auto start_time = std::chrono::steady_clock::now();
    
    for (int batch = 0; batch < 5; ++batch) {
        std::cout << "Batch " << (batch + 1) << "/5..." << std::endl;
        
        // Send burst of messages
        for (int i = 0; i < 20; ++i) {
            FloatVector msg(*channel);
            msg.resize(50);
            for (size_t j = 0; j < msg.size(); ++j) {
                msg[j] = static_cast<float>(batch * 1000 + i * 50 + j);
            }
            channel->send(msg);
        }
        
        // Receive messages
        for (int i = 0; i < 20; ++i) {
            auto msg = channel->receive<FloatVector>();
            if (!msg) break;
        }
        
        // Take a snapshot
        auto snapshot = profiler.snapshot();
        std::cout << "  Messages/sec: " << std::fixed << std::setprecision(1) 
                  << snapshot.messages_per_second << std::endl;
        std::cout << "  Throughput: " << (snapshot.bytes_per_second / 1024.0) 
                  << " KB/sec" << std::endl;
        
        // Brief pause between batches
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    profiler.stop();
    
    std::cout << "\nPerformance test completed in " << duration.count() << "ms" << std::endl;
    std::cout << profiler.generate_report() << std::endl;
}

void test_message_tracing() {
    std::cout << "\n=== Message Tracing Demo ===" << std::endl;
    
    const std::string channel_uri = "memory://trace_test";
    auto channel = create_channel(channel_uri, 1024*1024, 
                                 ChannelMode::SPSC, ChannelType::SingleType, true);
    
    // Enable tracing
    debug::MessageTracer::enable_tracing(channel_uri);
    std::cout << "Message tracing enabled for " << channel_uri << std::endl;
    
    // Simulate message events
    debug::MessageTracer::trace_event(channel_uri, "send", 400, 1, "FloatVector with 100 elements");
    debug::MessageTracer::trace_event(channel_uri, "receive", 400, 1, "Successfully received");
    debug::MessageTracer::trace_event(channel_uri, "send", 800, 1, "FloatVector with 200 elements");
    debug::MessageTracer::trace_event(channel_uri, "send", 1200, 1, "FloatVector with 300 elements");
    debug::MessageTracer::trace_event(channel_uri, "receive", 800, 1, "Successfully received");
    debug::MessageTracer::trace_event(channel_uri, "drop", 1200, 1, "Buffer full, message dropped");
    
    // Add a small delay to see timestamp differences
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    debug::MessageTracer::trace_event(channel_uri, "error", 0, 0, "Connection timeout");
    
    // Generate trace report
    std::cout << "\nTrace Report:" << std::endl;
    std::cout << debug::MessageTracer::generate_trace_report(channel_uri) << std::endl;
    
    // Cleanup
    debug::MessageTracer::disable_tracing(channel_uri);
    debug::MessageTracer::clear_traces(channel_uri);
}

void test_channel_monitoring() {
    std::cout << "\n=== Channel Monitoring Demo ===" << std::endl;
    
    // Create multiple channels
    auto channel1 = create_channel("memory://monitor1", 1024*1024, 
                                  ChannelMode::SPSC, ChannelType::SingleType, true);
    auto channel2 = create_channel("memory://monitor2", 512*1024, 
                                  ChannelMode::SPSC, ChannelType::SingleType, true);
    
    // Create shared_ptr copies for monitoring
    auto shared_channel1 = std::shared_ptr<Channel>(std::move(channel1));
    auto shared_channel2 = std::shared_ptr<Channel>(std::move(channel2));
    
    // Set up monitor
    debug::ChannelMonitor monitor;
    monitor.add_channel("Channel1", shared_channel1);
    monitor.add_channel("Channel2", shared_channel2);
    
    // Set alert callback
    monitor.set_alert_callback([](const std::string& name, debug::ChannelHealth health, const std::string& details) {
        std::cout << "🚨 ALERT: " << name << " health changed to ";
        switch (health) {
            case debug::ChannelHealth::Warning: std::cout << "WARNING"; break;
            case debug::ChannelHealth::Critical: std::cout << "CRITICAL"; break;
            case debug::ChannelHealth::Disconnected: std::cout << "DISCONNECTED"; break;
            default: std::cout << "UNKNOWN"; break;
        }
        std::cout << " - " << details << std::endl;
    });
    
    std::cout << "Starting monitoring..." << std::endl;
    monitor.start_monitoring(std::chrono::milliseconds(500));
    
    // Show initial status
    std::cout << monitor.get_status_summary() << std::endl;
    
    // Simulate some activity
    std::cout << "Simulating channel activity..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        FloatVector msg1(*shared_channel1);
        msg1.resize(100);
        shared_channel1->send(msg1);
        
        FloatVector msg2(*shared_channel2);
        msg2.resize(50);
        shared_channel2->send(msg2);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Show final status
    std::cout << "\nFinal monitoring status:" << std::endl;
    std::cout << monitor.get_status_summary() << std::endl;
    
    monitor.stop_monitoring();
    std::cout << "Monitoring stopped." << std::endl;
}

void test_debug_dashboard() {
    std::cout << "\n=== Debug Dashboard Demo ===" << std::endl;
    
    // Create some channels and register them for debugging
    auto channel1 = create_channel("memory://dash1", 1024*1024, 
                                  ChannelMode::SPSC, ChannelType::SingleType, true);
    auto channel2 = create_channel("memory://dash2", 512*1024, 
                                  ChannelMode::SPSC, ChannelType::SingleType, true);
    
    // The debug console would normally register these, but for demo we'll just show the concept
    std::cout << "Debug Dashboard:" << std::endl;
    std::cout << debug::create_dashboard() << std::endl;
    
    // Show individual channel diagnostics
    std::cout << "\nChannel Diagnostics:" << std::endl;
    auto diag1 = debug::ChannelInspector::inspect(*channel1);
    auto diag2 = debug::ChannelInspector::inspect(*channel2);
    
    std::cout << "Channel 1: " << diag1.summary() << std::endl;
    std::cout << "Channel 2: " << diag2.summary() << std::endl;
}

int main() {
    try {
        std::cout << "Psyne Debug and Introspection Demo" << std::endl;
        std::cout << "==================================" << std::endl;
        
        test_channel_inspection();
        test_performance_profiling();
        test_message_tracing();
        test_channel_monitoring();
        test_debug_dashboard();
        
        std::cout << "\n✅ All debugging features demonstrated successfully!" << std::endl;
        std::cout << "\nDebugging utilities available:" << std::endl;
        std::cout << "  • Channel inspection and health monitoring" << std::endl;
        std::cout << "  • Performance profiling and metrics" << std::endl;
        std::cout << "  • Message flow tracing" << std::endl;
        std::cout << "  • Real-time monitoring with alerts" << std::endl;
        std::cout << "  • Visual buffer usage displays" << std::endl;
        std::cout << "  • Comprehensive diagnostic reports" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}