/**
 * @file metrics_demo.cpp
 * @brief Demonstration of Psyne debug metrics collection
 *
 * Shows how to use the metrics system for debugging and performance analysis
 */

#include "psyne/psyne.hpp"
#include "psyne/debug/metrics_collector.hpp"
#include "logger.hpp"
#include <chrono>
#include <random>
#include <thread>
#include <vector>

using namespace psyne;
using namespace psyne::debug;

// Test message with variable payload
struct TestMessage {
    uint64_t sequence;
    uint64_t timestamp_ns;
    uint32_t producer_id;
    uint32_t payload_size;
    uint8_t payload[];
    
    void fill_pattern() {
        for (uint32_t i = 0; i < payload_size; ++i) {
            payload[i] = (sequence + i) & 0xFF;
        }
    }
    
    bool verify_pattern() const {
        for (uint32_t i = 0; i < payload_size; ++i) {
            if (payload[i] != ((sequence + i) & 0xFF)) {
                return false;
            }
        }
        return true;
    }
};

// Producer with configurable behavior
void producer_thread(uint32_t id, const std::string& channel_name,
                    size_t message_count, size_t message_size,
                    bool variable_size, bool introduce_delays) {
    try {
        // Create channel
        ChannelConfig config;
        config.name = channel_name;
        config.size_mb = 16;
        config.mode = ChannelMode::SPSC;
        config.transport = ChannelTransport::IPC;
        config.is_producer = true;
        config.blocking = true;
        
        auto channel = Channel<TestMessage>::create(config);
        
        // Register with metrics collector
        MetricsCollector::instance().register_channel(channel_name, 
                                                     config.transport,
                                                     config.size_mb * 1024 * 1024);
        
        log_info("[Producer ", id, "] Starting on channel ", channel_name);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> size_dist(64, message_size);
        std::uniform_int_distribution<> delay_dist(0, 10);
        
        for (size_t i = 0; i < message_count; ++i) {
            // Determine message size
            size_t payload_size = variable_size ? size_dist(gen) : message_size;
            size_t total_size = sizeof(TestMessage) + payload_size;
            
            // Allocate message
            auto start_alloc = std::chrono::high_resolution_clock::now();
            
            std::optional<typename Channel<TestMessage>::Message> msg;
            int retry_count = 0;
            
            while (!msg) {
                msg = channel->try_allocate(total_size);
                if (!msg) {
                    MetricsCollector::instance().record_allocation_failure(
                        channel_name, total_size);
                    
                    if (++retry_count > 100) {
                        log_error("[Producer ", id, 
                                  "] Failed to allocate after 100 retries");
                        break;
                    }
                    
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
            
            if (!msg) continue;
            
            auto alloc_time = std::chrono::high_resolution_clock::now() - start_alloc;
            
            // Fill message
            (*msg)->sequence = i;
            (*msg)->timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            (*msg)->producer_id = id;
            (*msg)->payload_size = payload_size;
            (*msg)->fill_pattern();
            
            // Track allocation in metrics
            MetricsCollector::instance().record_allocation(channel_name, total_size);
            
            // Send message
            msg->send();
            
            // Record send metrics
            MetricsCollector::instance().record_send(channel_name, total_size, i);
            
            // Update buffer state periodically
            if (i % 100 == 0) {
                size_t used = channel->get_bytes_used();
                size_t available = channel->get_bytes_available();
                MetricsCollector::instance().update_buffer_state(
                    channel_name, used, available);
            }
            
            // Introduce delays if requested
            if (introduce_delays && delay_dist(gen) > 7) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(delay_dist(gen)));
            }
            
            // Progress reporting
            if (i % 1000 == 0 && i > 0) {
                log_info("[Producer ", id, "] Sent ", i, " messages");
            }
        }
        
        log_info("[Producer ", id, "] Completed ", message_count, " messages");
        
    } catch (const std::exception& e) {
        log_error("[Producer ", id, "] Error: ", e.what());
    }
}

// Consumer with metrics tracking
void consumer_thread(uint32_t id, const std::string& channel_name,
                    size_t expected_messages, bool verify_data) {
    try {
        // Create channel
        ChannelConfig config;
        config.name = channel_name;
        config.size_mb = 16;
        config.mode = ChannelMode::SPSC;
        config.transport = ChannelTransport::IPC;
        config.is_producer = false;
        config.blocking = true;
        
        auto channel = Channel<TestMessage>::create(config);
        
        log_info("[Consumer ", id, "] Starting on channel ", channel_name);
        
        size_t received = 0;
        size_t errors = 0;
        
        while (received < expected_messages) {
            // Receive message
            auto msg = channel->receive();
            
            // Calculate latency
            uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            uint64_t latency = now - msg->timestamp_ns;
            
            // Verify data if requested
            if (verify_data && !msg->verify_pattern()) {
                log_error("[Consumer ", id, "] Data corruption in message ",
                          msg->sequence);
                errors++;
            }
            
            // Track deallocation
            size_t msg_size = sizeof(TestMessage) + msg->payload_size;
            MetricsCollector::instance().record_deallocation(channel_name, msg_size);
            
            // Record receive metrics
            MetricsCollector::instance().record_receive(
                channel_name, msg_size, msg->sequence, latency);
            
            received++;
            
            // Progress reporting
            if (received % 1000 == 0) {
                log_info("[Consumer ", id, "] Received ", received,
                          " messages, latency: ", latency / 1000, "µs");
            }
        }
        
        log_info("[Consumer ", id, "] Completed. Received: ",
                  received, ", Errors: ", errors);
        
    } catch (const std::exception& e) {
        log_error("[Consumer ", id, "] Error: ", e.what());
    }
}

// Stress test scenario
void stress_test_scenario() {
    log_info("=== Stress Test Scenario ===");
    log_info("Multiple producers sending to multiple channels");
    
    const size_t num_channels = 4;
    const size_t messages_per_channel = 10000;
    const size_t message_size = 1024;
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    
    for (size_t i = 0; i < num_channels; ++i) {
        std::string channel_name = "stress_channel_" + std::to_string(i);
        
        producers.emplace_back(producer_thread, i, channel_name,
                             messages_per_channel, message_size,
                             true, true); // variable size and delays
        
        consumers.emplace_back(consumer_thread, i, channel_name,
                             messages_per_channel, true);
    }
    
    // Wait for completion
    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();
}

// Latency test scenario
void latency_test_scenario() {
    log_info("=== Latency Test Scenario ===");
    log_info("Single producer/consumer with small messages");
    
    const std::string channel_name = "latency_test_channel";
    const size_t message_count = 100000;
    const size_t message_size = 64;
    
    std::thread producer(producer_thread, 0, channel_name,
                        message_count, message_size, false, false);
    
    std::thread consumer(consumer_thread, 0, channel_name,
                        message_count, false);
    
    producer.join();
    consumer.join();
}

// Throughput test scenario
void throughput_test_scenario() {
    log_info("=== Throughput Test Scenario ===");
    log_info("Maximum rate with optimal message size");
    
    const std::string channel_name = "throughput_test_channel";
    const size_t message_count = 1000000;
    const size_t message_size = 4096;
    
    std::thread producer(producer_thread, 0, channel_name,
                        message_count, message_size, false, false);
    
    std::thread consumer(consumer_thread, 0, channel_name,
                        message_count, false);
    
    producer.join();
    consumer.join();
}

int main(int argc, char* argv[]) {
    log_info("Psyne Debug Metrics Demo");
    log_info("========================");
    
    // Configure metrics collection
    MetricsConfig config;
    config.enabled = true;
    config.console_output = true;
    config.file_output = true;
    config.output_file = "metrics_demo.csv";
    config.sampling_interval_ms = 500;
    config.detailed_histograms = true;
    config.memory_tracking = true;
    config.event_tracing = true;
    config.live_dashboard = false; // Use separate monitor tool for dashboard
    
    MetricsCollector::instance().configure(config);
    MetricsCollector::instance().start();
    
    // Parse command line
    std::string scenario = "all";
    if (argc > 1) {
        scenario = argv[1];
    }
    
    // Run scenarios
    if (scenario == "latency" || scenario == "all") {
        latency_test_scenario();
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    
    if (scenario == "throughput" || scenario == "all") {
        throughput_test_scenario();
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    
    if (scenario == "stress" || scenario == "all") {
        stress_test_scenario();
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    
    // Stop metrics collection
    MetricsCollector::instance().stop();
    
    // Print final summary
    log_info("=== Final Metrics Summary ===");
    
    auto final_metrics = MetricsCollector::instance().get_snapshot();
    for (const auto& [name, metrics] : final_metrics) {
        log_info("Channel: ", name);
        log_info("  Messages sent: ", metrics.messages_sent);
        log_info("  Messages received: ", metrics.messages_received);
        log_info("  Total bytes: ", (metrics.bytes_sent + metrics.bytes_received));
        log_info("  Allocation failures: ", metrics.allocation_failures);
        
        auto percentiles = metrics.latency_histogram.get_percentiles();
        log_info("  Latency (µs): min=", percentiles.min / 1000.0,
                  ", p50=", percentiles.p50 / 1000.0,
                  ", p90=", percentiles.p90 / 1000.0,
                  ", p99=", percentiles.p99 / 1000.0,
                  ", max=", percentiles.max / 1000.0);
        
        if (config.memory_tracking) {
            log_info("  Peak allocations: ", metrics.peak_allocations);
            log_info("  Current allocations: ", metrics.current_allocations);
        }
    }
    
    // Dump events if tracing was enabled
    if (config.event_tracing) {
        log_info("=== Event Trace Sample ===");
        auto events = MetricsCollector::instance().get_events();
        
        // Show last 10 events
        size_t start = events.size() > 10 ? events.size() - 10 : 0;
        for (size_t i = start; i < events.size(); ++i) {
            const auto& event = events[i];
            std::ostringstream event_msg;
            event_msg << "[" << event.timestamp_ns << "] "
                      << event.channel_name << " - ";
            
            switch (event.type) {
                case EventType::MESSAGE_ALLOCATE:
                    event_msg << "ALLOCATE";
                    break;
                case EventType::MESSAGE_COMMIT:
                    event_msg << "COMMIT";
                    break;
                case EventType::MESSAGE_RECEIVE:
                    event_msg << "RECEIVE";
                    break;
                case EventType::BUFFER_FULL:
                    event_msg << "BUFFER_FULL";
                    break;
                default:
                    event_msg << "OTHER";
            }
            
            if (!event.extra_info.empty()) {
                event_msg << " (" << event.extra_info << ")";
            }
            log_info(event_msg.str());
        }
    }
    
    log_info("Metrics have been exported to: ", config.output_file);
    log_info("Use psyne_monitor tool for real-time visualization");
    
    return 0;
}