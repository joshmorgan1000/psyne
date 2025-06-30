/**
 * @file filtered_fanout_demo.cpp
 * @brief Demonstration of the filtered fanout dispatcher pattern
 *
 * This example shows how to use the FilteredFanoutDispatcher to:
 * - Route messages based on custom predicates
 * - Process messages in parallel with multiple handlers
 * - Automatically route responses back to senders
 * - Aggregate multiple responses into a single reply
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include <psyne/psyne.hpp>
#include "../src/patterns/filtered_fanout_dispatcher.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace psyne;
using namespace psyne::patterns;

// Example message types
class SensorData : public Message<SensorData> {
public:
    static constexpr uint32_t message_type = 1001;
    
    struct Data {
        uint32_t sensor_id;
        float temperature;
        float humidity;
        uint64_t timestamp;
        char location[32];
    };
    
    static size_t calculate_size() noexcept {
        return sizeof(Data);
    }
    
    Data& data() { return *reinterpret_cast<Data*>(Message::data()); }
    const Data& data() const { return *reinterpret_cast<const Data*>(Message::data()); }
};

class AlertNotification : public Message<AlertNotification> {
public:
    static constexpr uint32_t message_type = 1002;
    
    enum class AlertLevel : uint32_t {
        INFO = 0,
        WARNING = 1,
        CRITICAL = 2
    };
    
    struct Data {
        AlertLevel level;
        uint32_t source_sensor;
        float threshold_value;
        float actual_value;
        char message[128];
    };
    
    static size_t calculate_size() noexcept {
        return sizeof(Data);
    }
    
    Data& data() { return *reinterpret_cast<Data*>(Message::data()); }
    const Data& data() const { return *reinterpret_cast<const Data*>(Message::data()); }
};

class ProcessingResult : public Message<ProcessingResult> {
public:
    static constexpr uint32_t message_type = 1003;
    
    struct Data {
        char processor_name[32];
        uint32_t items_processed;
        float processing_time_ms;
        bool success;
        char details[64];
    };
    
    static size_t calculate_size() noexcept {
        return sizeof(Data);
    }
    
    Data& data() { return *reinterpret_cast<Data*>(Message::data()); }
    const Data& data() const { return *reinterpret_cast<const Data*>(Message::data()); }
};

// Example processors that will handle messages
class TemperatureMonitor {
public:
    TemperatureMonitor(float threshold) : threshold_(threshold) {}
    
    ProcessingResult process(const SensorData& sensor) {
        std::cout << "[Temperature Monitor] Processing sensor " << sensor.data().sensor_id
                  << " temp=" << sensor.data().temperature << "°C" << std::endl;
        
        // Simulate processing
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Create response
        auto channel = Channel::create("memory://temp_response", 1024 * 1024);
        ProcessingResult result(*channel);
        
        auto& res_data = result.data();
        std::strcpy(res_data.processor_name, "TemperatureMonitor");
        res_data.items_processed = 1;
        res_data.processing_time_ms = 50.0f;
        res_data.success = true;
        
        if (sensor.data().temperature > threshold_) {
            std::snprintf(res_data.details, sizeof(res_data.details),
                         "High temp alert: %.1f°C > %.1f°C",
                         sensor.data().temperature, threshold_);
        } else {
            std::strcpy(res_data.details, "Temperature within normal range");
        }
        
        return result;
    }
    
private:
    float threshold_;
};

class HumidityAnalyzer {
public:
    void analyze(const SensorData& sensor) {
        std::cout << "[Humidity Analyzer] Analyzing sensor " << sensor.data().sensor_id
                  << " humidity=" << sensor.data().humidity << "%" << std::endl;
        
        // This processor doesn't send responses - fire and forget
        if (sensor.data().humidity > 80.0f) {
            std::cout << "  WARNING: High humidity detected!" << std::endl;
        }
        
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
};

class DataLogger {
public:
    ProcessingResult log_data(const SensorData& sensor) {
        std::cout << "[Data Logger] Logging: sensor=" << sensor.data().sensor_id
                  << " location=" << sensor.data().location
                  << " [T=" << sensor.data().temperature 
                  << " H=" << sensor.data().humidity << "]" << std::endl;
        
        // Simulate database write
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        logged_count_++;
        
        // Return log result
        auto channel = Channel::create("memory://log_response", 1024 * 1024);
        ProcessingResult result(*channel);
        
        auto& res_data = result.data();
        std::strcpy(res_data.processor_name, "DataLogger");
        res_data.items_processed = logged_count_;
        res_data.processing_time_ms = 20.0f;
        res_data.success = true;
        std::snprintf(res_data.details, sizeof(res_data.details),
                     "Logged entry #%u", logged_count_);
        
        return result;
    }
    
private:
    std::atomic<uint32_t> logged_count_{0};
};

void demonstrate_filtered_fanout() {
    std::cout << "\n=== Filtered Fanout Dispatcher Demo ===\n" << std::endl;
    
    // Create channels
    auto sensor_channel = Channel::create("memory://sensors", 1024 * 1024);
    auto response_channel = Channel::create("memory://responses", 1024 * 1024);
    
    // Create dispatcher with 4 worker threads
    FilteredFanoutDispatcher dispatcher(sensor_channel, 4);
    
    // Create processors
    TemperatureMonitor temp_monitor(30.0f);  // Alert on temps > 30°C
    HumidityAnalyzer humidity_analyzer;
    DataLogger logger;
    
    // Add routes with different predicates
    
    // Route 1: High temperature sensors (> 25°C) go to temperature monitor
    dispatcher.add_typed_route<SensorData>(
        "HighTempMonitor",
        [](const SensorData& msg) {
            return msg.data().temperature > 25.0f;
        },
        [&temp_monitor](const SensorData& msg) {
            return temp_monitor.process(msg);
        },
        PsynePool::HIGH_PRIORITY  // High priority for alerts
    );
    
    // Route 2: High humidity (> 70%) goes to analyzer (no response)
    dispatcher.add_typed_route<SensorData>(
        "HumidityAnalyzer", 
        [](const SensorData& msg) {
            return msg.data().humidity > 70.0f;
        },
        [&humidity_analyzer](const SensorData& msg) {
            humidity_analyzer.analyze(msg);
        }
    );
    
    // Route 3: All sensors from "Lab-A" go to logger
    dispatcher.add_typed_route<SensorData>(
        "LabALogger",
        [](const SensorData& msg) {
            return std::strstr(msg.data().location, "Lab-A") != nullptr;
        },
        [&logger](const SensorData& msg) {
            return logger.log_data(msg);
        }
    );
    
    // Route 4: Critical sensors always get logged (overlaps with route 3)
    dispatcher.add_typed_route<SensorData>(
        "CriticalLogger",
        [](const SensorData& msg) {
            return msg.data().sensor_id < 100;  // IDs < 100 are critical
        },
        [&logger](const SensorData& msg) {
            return logger.log_data(msg);
        },
        PsynePool::HIGH_PRIORITY
    );
    
    // Start dispatcher
    dispatcher.start();
    
    std::cout << "Dispatcher started with 4 worker threads\n" << std::endl;
    
    // Simulate sending sensor data
    std::cout << "Sending sensor data..." << std::endl;
    
    std::vector<std::pair<uint32_t, std::string>> test_sensors = {
        {1, "Lab-A-Reactor"},      // Critical + Lab-A (2 routes)
        {101, "Lab-A-Storage"},    // Lab-A only (1 route)
        {102, "Lab-B-Office"},     // No routes match
        {2, "Lab-C-Server"},       // Critical only (1 route)
        {201, "Lab-A-Entrance"}    // Lab-A only (1 route)
    };
    
    for (const auto& [sensor_id, location] : test_sensors) {
        SensorData sensor(*sensor_channel);
        auto& data = sensor.data();
        
        data.sensor_id = sensor_id;
        data.temperature = 20.0f + (sensor_id % 20);  // Varying temps
        data.humidity = 60.0f + (sensor_id % 30);     // Varying humidity
        data.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        std::strncpy(data.location, location.c_str(), sizeof(data.location) - 1);
        
        std::cout << "\nSending: Sensor " << sensor_id 
                  << " [" << location << "] "
                  << "T=" << data.temperature << "°C "
                  << "H=" << data.humidity << "%" << std::endl;
        
        // Build message with reply info
        auto msg_data = MessageBuilder<SensorData>::build_with_reply_info(
            sensor, 
            response_channel->uri(),
            sensor_id  // Use sensor ID as correlation ID
        );
        
        // Send via raw interface to include reply info
        auto slot = sensor_channel->reserve_write_slot(msg_data.size());
        if (slot != BUFFER_FULL) {
            auto span = sensor_channel->get_write_span(msg_data.size());
            std::memcpy(span.data(), msg_data.data(), msg_data.size());
            sensor_channel->notify_message_ready(slot, msg_data.size());
        }
        
        // Small delay between sends
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Wait for processing and collect responses
    std::cout << "\n--- Waiting for responses ---" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Read aggregated responses
    std::cout << "\n--- Aggregated Responses ---" << std::endl;
    
    while (true) {
        size_t size;
        uint32_t type;
        void* response_data = response_channel->receive_message(size, type);
        
        if (!response_data) break;
        
        // Parse aggregated response
        const uint8_t* data = static_cast<const uint8_t*>(response_data);
        uint32_t response_count = *reinterpret_cast<const uint32_t*>(data);
        data += sizeof(uint32_t);
        
        std::cout << "\nReceived " << response_count << " responses:" << std::endl;
        
        for (uint32_t i = 0; i < response_count; ++i) {
            uint32_t resp_size = *reinterpret_cast<const uint32_t*>(data);
            data += sizeof(uint32_t);
            
            if (resp_size == 0) {
                std::cout << "  - Empty response (fire-and-forget handler)" << std::endl;
            } else if (resp_size == sizeof(ProcessingResult::Data)) {
                const auto* result = reinterpret_cast<const ProcessingResult::Data*>(data);
                std::cout << "  - " << result->processor_name 
                          << ": " << result->details
                          << " (took " << result->processing_time_ms << "ms)" << std::endl;
            }
            
            data += resp_size;
        }
        
        response_channel->release_message(response_data);
    }
    
    // Show metrics
    auto metrics = dispatcher.get_metrics();
    std::cout << "\n--- Dispatcher Metrics ---" << std::endl;
    std::cout << "Messages received: " << metrics.messages_received << std::endl;
    std::cout << "Messages dispatched: " << metrics.messages_dispatched << std::endl;
    std::cout << "No matches: " << metrics.no_matches << std::endl;
    std::cout << "Active routes: " << metrics.active_routes << std::endl;
    std::cout << "Pending tasks: " << metrics.pending_tasks << std::endl;
    
    // Stop dispatcher
    dispatcher.stop();
    std::cout << "\nDispatcher stopped." << std::endl;
}

void demonstrate_alert_routing() {
    std::cout << "\n\n=== Alert Priority Routing Demo ===\n" << std::endl;
    
    auto alert_channel = Channel::create("memory://alerts", 1024 * 1024);
    auto alert_response = Channel::create("memory://alert_responses", 1024 * 1024);
    
    FilteredFanoutDispatcher alert_dispatcher(alert_channel, 2);  // 2 threads for alerts
    
    // Add alert routes based on severity
    
    // Critical alerts - immediate action
    alert_dispatcher.add_typed_route<AlertNotification>(
        "CriticalHandler",
        [](const AlertNotification& alert) {
            return alert.data().level == AlertNotification::AlertLevel::CRITICAL;
        },
        [](const AlertNotification& alert) {
            std::cout << "[CRITICAL ALERT] Sensor " << alert.data().source_sensor
                      << " - " << alert.data().message << std::endl;
            std::cout << "  >>> TRIGGERING EMERGENCY RESPONSE <<<" << std::endl;
            
            // Simulate emergency action
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        },
        PsynePool::HIGH_PRIORITY
    );
    
    // Warning alerts - log and monitor
    alert_dispatcher.add_typed_route<AlertNotification>(
        "WarningLogger",
        [](const AlertNotification& alert) {
            return alert.data().level == AlertNotification::AlertLevel::WARNING;
        },
        [](const AlertNotification& alert) {
            std::cout << "[Warning] Sensor " << alert.data().source_sensor
                      << ": " << alert.data().message
                      << " (value=" << alert.data().actual_value 
                      << ", threshold=" << alert.data().threshold_value << ")" << std::endl;
        }
    );
    
    // All alerts get timestamped
    alert_dispatcher.add_typed_route<AlertNotification>(
        "TimestampLogger",
        [](const AlertNotification&) { return true; },  // Match all
        [](const AlertNotification& alert) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            std::cout << "[Log] " << std::put_time(std::localtime(&time_t), "%H:%M:%S")
                      << " - Alert from sensor " << alert.data().source_sensor << std::endl;
        },
        PsynePool::LOW_PRIORITY
    );
    
    alert_dispatcher.start();
    
    // Send some test alerts
    std::vector<std::tuple<AlertNotification::AlertLevel, uint32_t, std::string, float, float>> test_alerts = {
        {AlertNotification::AlertLevel::INFO, 101, "Normal operation", 25.0f, 30.0f},
        {AlertNotification::AlertLevel::WARNING, 102, "Temperature rising", 35.0f, 30.0f},
        {AlertNotification::AlertLevel::CRITICAL, 103, "OVERHEATING DETECTED", 45.0f, 30.0f},
        {AlertNotification::AlertLevel::WARNING, 104, "Humidity high", 85.0f, 80.0f},
        {AlertNotification::AlertLevel::CRITICAL, 105, "SYSTEM FAILURE", 0.0f, 0.0f}
    };
    
    for (const auto& [level, sensor, msg, actual, threshold] : test_alerts) {
        AlertNotification alert(*alert_channel);
        auto& data = alert.data();
        
        data.level = level;
        data.source_sensor = sensor;
        data.actual_value = actual;
        data.threshold_value = threshold;
        std::strncpy(data.message, msg.c_str(), sizeof(data.message) - 1);
        
        alert.send();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    alert_dispatcher.stop();
}

int main() {
    std::cout << "Filtered Fanout Dispatcher Pattern Demo\n";
    std::cout << "======================================\n";
    
    try {
        // Demo 1: Sensor data with multiple processors
        demonstrate_filtered_fanout();
        
        // Demo 2: Alert routing by priority
        demonstrate_alert_routing();
        
        std::cout << "\nDemo completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}