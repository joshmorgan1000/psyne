#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <psyne/psyne.hpp>
#include <random>
#include <thread>

using namespace psyne;
using namespace std::chrono_literals;

// Dynamic-size message for variable data
class DynamicDataMessage : public Message<DynamicDataMessage> {
public:
    static constexpr uint32_t message_type = 300;
    
    using Message<DynamicDataMessage>::Message;
    
    // Header structure for dynamic messages
    struct Header {
        uint32_t data_size;
        uint32_t sequence_number;
        uint64_t timestamp;
    };
    
    // Calculate size includes header + dynamic data
    static size_t calculate_size() {
        // Default allocation for dynamic messages
        return sizeof(Header) + 1024;  // 1KB default
    }
    
    void initialize(uint32_t seq, size_t data_size) {
        if (!data_) return;
        
        auto* header = reinterpret_cast<Header*>(data_);
        header->data_size = static_cast<uint32_t>(data_size);
        header->sequence_number = seq;
        header->timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        
        // Initialize data area
        uint8_t* data_area = data_ + sizeof(Header);
        for (size_t i = 0; i < data_size && i < (size_ - sizeof(Header)); ++i) {
            data_area[i] = static_cast<uint8_t>((seq + i) % 256);
        }
    }
    
    Header get_header() const {
        if (!data_) return {};
        return *reinterpret_cast<const Header*>(data_);
    }
    
    std::span<const uint8_t> get_data() const {
        if (!data_ || size_ <= sizeof(Header)) return {};
        const uint8_t* data_start = data_ + sizeof(Header);
        return std::span<const uint8_t>(data_start, size_ - sizeof(Header));
    }
};

void demonstrate_dynamic_messages() {
    std::cout << "\n=== Dynamic Message Allocation Demo ===\n";
    
    // Create channel with large buffer for dynamic allocation
    auto channel = create_channel("memory://dynamic_demo", 
                                  64 * 1024 * 1024,  // 64MB buffer
                                  ChannelMode::SPSC,
                                  ChannelType::SingleType,
                                  true);  // Enable metrics
    
    std::cout << "Created channel with 64MB buffer for dynamic allocations\n\n";
    
    // Random size generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> size_dist(64, 4096);  // 64B to 4KB
    
    // Producer thread - sends messages of varying sizes
    std::thread producer([&]() {
        uint32_t sequence = 0;
        size_t total_sent = 0;
        
        for (int i = 0; i < 100; ++i) {
            size_t data_size = size_dist(gen);
            
            try {
                // Allocate message with specific size
                DynamicDataMessage msg(*channel);
                msg.initialize(sequence++, data_size);
                
                auto header = msg.get_header();
                std::cout << "[Producer] Sending seq=" << header.sequence_number 
                          << ", size=" << data_size << " bytes\n";
                
                msg.send();
                total_sent += data_size;
                
                // Vary sending rate
                if (i % 10 == 0) {
                    std::this_thread::sleep_for(10ms);
                }
            } catch (const std::runtime_error& e) {
                std::cout << "[Producer] Buffer full, waiting...\n";
                std::this_thread::sleep_for(50ms);
                i--; // Retry
            }
        }
        
        std::cout << "[Producer] Total sent: " << total_sent / 1024.0 << " KB\n";
    });
    
    // Consumer thread - receives and validates messages
    std::thread consumer([&]() {
        size_t total_received = 0;
        uint32_t expected_seq = 0;
        
        for (int i = 0; i < 100; ++i) {
            size_t msg_size;
            uint32_t msg_type;
            void* msg_data = channel->receive_message(msg_size, msg_type);
            
            if (msg_data && msg_type == DynamicDataMessage::message_type) {
                DynamicDataMessage msg(msg_data, msg_size);
                auto header = msg.get_header();
                auto data = msg.get_data();
                
                // Validate sequence
                if (header.sequence_number != expected_seq) {
                    std::cout << "[Consumer] Sequence error! Expected " << expected_seq 
                              << ", got " << header.sequence_number << "\n";
                }
                expected_seq = header.sequence_number + 1;
                
                // Validate data
                bool valid = true;
                for (size_t j = 0; j < header.data_size && j < data.size(); ++j) {
                    if (data[j] != static_cast<uint8_t>((header.sequence_number + j) % 256)) {
                        valid = false;
                        break;
                    }
                }
                
                if (!valid) {
                    std::cout << "[Consumer] Data validation failed for seq=" 
                              << header.sequence_number << "\n";
                }
                
                total_received += header.data_size;
                
                if (i % 20 == 0) {
                    std::cout << "[Consumer] Received seq=" << header.sequence_number 
                              << ", size=" << header.data_size << " bytes\n";
                }
                
                channel->release_message(msg_data);
            } else {
                std::this_thread::sleep_for(10ms);
                i--; // Retry
            }
        }
        
        std::cout << "[Consumer] Total received: " << total_received / 1024.0 << " KB\n";
    });
    
    producer.join();
    consumer.join();
    
    // Show final metrics
    if (channel->has_metrics()) {
        auto metrics = channel->get_metrics();
        std::cout << "\nFinal channel metrics:\n";
        std::cout << "  Messages sent: " << metrics.messages_sent << "\n";
        std::cout << "  Messages received: " << metrics.messages_received << "\n";
        std::cout << "  Bytes sent: " << metrics.bytes_sent / 1024.0 << " KB\n";
        std::cout << "  Bytes received: " << metrics.bytes_received / 1024.0 << " KB\n";
    }
}

void demonstrate_buffer_management() {
    std::cout << "\n=== Buffer Management Demo ===\n";
    
    // Create multiple channels with different buffer sizes
    struct ChannelConfig {
        std::string name;
        size_t buffer_size;
        std::string description;
    };
    
    std::vector<ChannelConfig> configs = {
        {"small", 1 * 1024 * 1024, "1MB buffer for small messages"},
        {"medium", 16 * 1024 * 1024, "16MB buffer for medium workload"},
        {"large", 128 * 1024 * 1024, "128MB buffer for large data"}
    };
    
    for (const auto& config : configs) {
        std::cout << "\nTesting " << config.description << ":\n";
        
        auto channel = create_channel("memory://" + config.name,
                                      config.buffer_size,
                                      ChannelMode::SPSC);
        
        // Calculate how many 64KB messages fit
        size_t message_size = 64 * 1024;
        size_t max_messages = config.buffer_size / message_size;
        
        std::cout << "  Can hold approximately " << max_messages 
                  << " messages of 64KB each\n";
        
        // Fill the buffer
        size_t sent = 0;
        while (true) {
            try {
                FloatVector msg(*channel);
                msg.resize(message_size / sizeof(float));
                msg.send();
                sent++;
            } catch (const std::runtime_error& e) {
                break;  // Buffer full
            }
        }
        
        std::cout << "  Actually sent " << sent << " messages before buffer full\n";
        std::cout << "  Efficiency: " << (sent * 100.0 / max_messages) << "%\n";
    }
}

void demonstrate_zero_copy_benefits() {
    std::cout << "\n=== Zero-Copy Benefits Demo ===\n";
    
    auto channel = create_channel("memory://zerocopy", 
                                  32 * 1024 * 1024,
                                  ChannelMode::SPSC);
    
    // Large message (1MB)
    const size_t large_size = 1024 * 1024;
    
    std::cout << "Sending 1MB message with zero-copy...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        FloatVector msg(*channel);
        msg.resize(large_size / sizeof(float));
        
        // Fill with data
        for (size_t i = 0; i < msg.size(); ++i) {
            msg[i] = static_cast<float>(i);
        }
        
        // Get pointer before sending
        void* send_ptr = msg.data();
        
        msg.send();
        
        // Receive
        size_t msg_size;
        uint32_t msg_type;
        void* recv_data = channel->receive_message(msg_size, msg_type);
        
        if (recv_data) {
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
            
            std::cout << "  Time: " << microseconds << " µs\n";
            std::cout << "  Throughput: " << (large_size / 1024.0 / 1024.0) / (microseconds / 1e6) 
                      << " MB/s\n";
            std::cout << "  Send pointer: " << send_ptr << "\n";
            std::cout << "  Recv pointer: " << recv_data << "\n";
            std::cout << "  Same memory? " << (send_ptr == recv_data ? "YES (zero-copy!)" : "NO") << "\n";
            
            channel->release_message(recv_data);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

int main() {
    std::cout << "Psyne Dynamic Allocation Demo\n";
    std::cout << "=============================\n";
    std::cout << "Demonstrating dynamic message sizes and buffer management\n";

    demonstrate_dynamic_messages();
    demonstrate_buffer_management();
    demonstrate_zero_copy_benefits();

    std::cout << "\nDemo completed successfully!\n";
    return 0;
}