#include <cstring>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>
#include <map>

using namespace psyne;

// Simple counter message
class CounterMessage : public Message<CounterMessage> {
public:
    static constexpr uint32_t message_type = 200;

    using Message<CounterMessage>::Message;

    static constexpr size_t calculate_size() {
        return sizeof(uint64_t);
    }

    void set_count(uint64_t count) {
        if (data_) {
            *reinterpret_cast<uint64_t *>(data_) = count;
        }
    }

    uint64_t get_count() const {
        if (!data_)
            return 0;
        return *reinterpret_cast<const uint64_t *>(data_);
    }
};

// Channel factory demonstration
class ChannelFactoryDemo {
public:
    // Factory method for creating channels with different configurations
    static std::unique_ptr<Channel> create_optimized_channel(
        const std::string& protocol,
        const std::string& name,
        size_t message_size,
        size_t expected_throughput) {
        
        // Calculate optimal buffer size based on expected throughput
        size_t buffer_size = calculate_buffer_size(message_size, expected_throughput);
        
        // Select mode based on requirements
        ChannelMode mode = select_channel_mode(protocol);
        
        // Build URI
        std::string uri = protocol + "://" + name;
        
        std::cout << "Creating channel:\n";
        std::cout << "  URI: " << uri << "\n";
        std::cout << "  Buffer size: " << buffer_size / 1024 << " KB\n";
        std::cout << "  Mode: " << mode_to_string(mode) << "\n\n";
        
        return create_channel(uri, buffer_size, mode, ChannelType::SingleType);
    }
    
private:
    static size_t calculate_buffer_size(size_t message_size, size_t messages_per_sec) {
        // Buffer for 100ms worth of messages plus overhead
        size_t base_size = message_size * messages_per_sec / 10;
        // Round up to nearest MB
        return ((base_size + 1024 * 1024 - 1) / (1024 * 1024)) * (1024 * 1024);
    }
    
    static ChannelMode select_channel_mode(const std::string& protocol) {
        // IPC and memory channels work well with SPSC
        if (protocol == "memory" || protocol == "ipc") {
            return ChannelMode::SPSC;
        }
        // Network channels often need multi-producer support
        return ChannelMode::MPSC;
    }
    
    static std::string mode_to_string(ChannelMode mode) {
        switch (mode) {
            case ChannelMode::SPSC: return "SPSC (Single Producer, Single Consumer)";
            case ChannelMode::MPSC: return "MPSC (Multi Producer, Single Consumer)";
            case ChannelMode::SPMC: return "SPMC (Single Producer, Multi Consumer)";
            case ChannelMode::MPMC: return "MPMC (Multi Producer, Multi Consumer)";
            default: return "Unknown";
        }
    }
};

void demo_memory_channel() {
    std::cout << "\n=== Memory Channel Demo ===\n";

    // Create optimized in-memory channel
    auto channel = ChannelFactoryDemo::create_optimized_channel(
        "memory", "demo", 
        CounterMessage::calculate_size(), 
        1000 // Expected 1000 messages/sec
    );

    // Producer thread
    std::thread producer([&channel]() {
        for (uint64_t i = 1; i <= 5; ++i) {
            try {
                CounterMessage msg(*channel);
                msg.set_count(i);
                msg.send();
                std::cout << "[Producer] Sent count: " << i << std::endl;
            } catch (const std::runtime_error& e) {
                std::cout << "[Producer] Buffer full!" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // Consumer thread
    std::thread consumer([&channel]() {
        for (int i = 0; i < 5; ++i) {
            size_t msg_size;
            uint32_t msg_type;
            void* msg_data = channel->receive_raw_message(msg_size, msg_type);
            
            if (msg_data && msg_type == CounterMessage::message_type) {
                CounterMessage msg(msg_data, msg_size);
                std::cout << "[Consumer] Received count: " << msg.get_count() << std::endl;
                channel->release_raw_message(msg_data);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                i--; // Retry
            }
        }
    });

    producer.join();
    consumer.join();
}

void demo_channel_registry() {
    std::cout << "\n=== Channel Registry Demo ===\n";
    
    // Simple channel registry for managing multiple channels
    std::map<std::string, std::unique_ptr<Channel>> registry;
    
    // Register different channel types
    registry["control"] = create_channel("memory://control", 1024 * 1024);
    registry["data"] = create_channel("memory://data", 64 * 1024 * 1024);
    registry["metrics"] = create_channel("memory://metrics", 256 * 1024);
    
    std::cout << "Registered channels:\n";
    for (const auto& [name, channel] : registry) {
        std::cout << "  " << name << " -> " << channel->uri() << "\n";
    }
    
    // Send messages to different channels
    try {
        CounterMessage control_msg(*registry["control"]);
        control_msg.set_count(42);
        control_msg.send();
        std::cout << "\nSent control message: 42\n";
        
        CounterMessage data_msg(*registry["data"]);
        data_msg.set_count(12345);
        data_msg.send();
        std::cout << "Sent data message: 12345\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

void demo_adaptive_channel() {
    std::cout << "\n=== Adaptive Channel Configuration Demo ===\n";
    
    // Demonstrate creating channels with different configurations
    // based on usage patterns
    
    struct ChannelConfig {
        std::string name;
        size_t message_size;
        size_t messages_per_sec;
        std::string description;
    };
    
    std::vector<ChannelConfig> configs = {
        {"high_frequency", 64, 100000, "High-frequency trading data"},
        {"large_tensors", 4 * 1024 * 1024, 10, "Large tensor transfers"},
        {"control", 256, 100, "Control messages"},
        {"telemetry", 1024, 1000, "Telemetry data"}
    };
    
    for (const auto& config : configs) {
        std::cout << "\nConfiguration for " << config.description << ":\n";
        auto channel = ChannelFactoryDemo::create_optimized_channel(
            "memory", config.name, 
            config.message_size, 
            config.messages_per_sec
        );
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Psyne Channel Factory Demo\n";
    std::cout << "==========================\n";

    // Demonstrate different factory patterns
    demo_memory_channel();
    demo_channel_registry();
    demo_adaptive_channel();

    std::cout << "\nDemo completed successfully!\n";
    return 0;
}