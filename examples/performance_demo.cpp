/**
 * @file performance_demo.cpp
 * @brief Basic performance demonstration for Psyne
 * 
 * Shows basic performance measurement and optimization techniques
 * available in Psyne v1.3.0
 */

#include <chrono>
#include <iostream>
#include <numeric>
#include <psyne/psyne.hpp>
#include <thread>
#include <vector>

using namespace psyne;
using namespace std::chrono;

// Simple message type for performance testing
class PerformanceMessage : public Message<PerformanceMessage> {
public:
    static constexpr uint32_t message_type = 100;
    using Message<PerformanceMessage>::Message;

    static size_t calculate_size() {
        return sizeof(float) * 1024; // 1KB of float data
    }

    void set_data(const std::vector<float>& data) {
        auto* ptr = reinterpret_cast<float*>(this->data());
        std::copy(data.begin(), data.end(), ptr);
    }

    std::vector<float> get_data() const {
        auto* ptr = reinterpret_cast<const float*>(this->data());
        return std::vector<float>(ptr, ptr + 1024);
    }
};

void benchmark_memory_channels() {
    std::cout << "=== Memory Channel Performance Test ===\n";
    
    // Test SPSC (Single Producer Single Consumer)
    auto channel = create_channel("memory://perf-test", 64*1024*1024, ChannelMode::SPSC);
    
    const int num_messages = 10000;
    std::vector<float> test_data(1024);
    std::iota(test_data.begin(), test_data.end(), 1.0f);
    
    auto start = high_resolution_clock::now();
    
    // Producer
    std::thread producer([&]() {
        for (int i = 0; i < num_messages; ++i) {
            PerformanceMessage msg(*channel);
            msg.set_data(test_data);
            msg.send();
        }
    });
    
    // Consumer
    int received = 0;
    std::thread consumer([&]() {
        while (received < num_messages) {
            size_t size;
            uint32_t type;
            void* msg_data = channel->receive_raw_message(size, type);
            if (msg_data) {
                received++;
                channel->release_raw_message(msg_data);
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    double throughput = (num_messages * 1024 * sizeof(float)) / (duration.count() / 1e6) / (1024*1024);
    std::cout << "SPSC Throughput: " << throughput << " MB/s\n";
    std::cout << "Messages per second: " << (num_messages * 1e6) / duration.count() << "\n";
}

void benchmark_channel_modes() {
    std::cout << "\n=== Channel Mode Comparison ===\n";
    
    const size_t buffer_size = 16*1024*1024;
    const int num_messages = 1000;
    
    // Test different channel modes
    std::vector<std::pair<ChannelMode, std::string>> modes = {
        {ChannelMode::SPSC, "SPSC (Single Producer/Single Consumer)"},
        {ChannelMode::MPSC, "MPSC (Multi Producer/Single Consumer)"},
        {ChannelMode::SPMC, "SPMC (Single Producer/Multi Consumer)"},
        {ChannelMode::MPMC, "MPMC (Multi Producer/Multi Consumer)"}
    };
    
    for (auto& [mode, name] : modes) {
        auto channel = create_channel("memory://mode-test", buffer_size, mode);
        
        auto start = high_resolution_clock::now();
        
        // Simple send/receive test
        std::thread sender([&]() {
            std::vector<float> data(256, 1.0f);
            for (int i = 0; i < num_messages; ++i) {
                PerformanceMessage msg(*channel);
                msg.set_data(data);
                msg.send();
            }
        });
        
        std::thread receiver([&]() {
            int received = 0;
            while (received < num_messages) {
                size_t size;
                uint32_t type;
                void* msg_data = channel->receive_raw_message(size, type);
                if (msg_data) {
                    received++;
                    channel->release_raw_message(msg_data);
                } else {
                    std::this_thread::yield();
                }
            }
        });
        
        sender.join();
        receiver.join();
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        
        std::cout << name << ": " << duration.count() << " μs\n";
    }
}

void show_performance_tips() {
    std::cout << "\n=== Psyne Performance Optimization Tips ===\n";
    std::cout << "1. Use SPSC channels for single producer/consumer scenarios (fastest)\n";
    std::cout << "2. Pre-allocate large ring buffers to avoid blocking\n";
    std::cout << "3. Use zero-copy messaging with Message<T> classes\n";
    std::cout << "4. Batch operations when possible\n";
    std::cout << "5. Consider memory layout and cache locality\n";
    std::cout << "6. Use memory:// channels for intra-process communication\n";
    std::cout << "7. Profile with channel metrics to identify bottlenecks\n";
}

int main() {
    std::cout << "Psyne Performance Demo - v1.3.0\n";
    std::cout << "=====================================\n\n";
    
    try {
        benchmark_memory_channels();
        benchmark_channel_modes();
        show_performance_tips();
        
        std::cout << "\nPerformance demo completed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}