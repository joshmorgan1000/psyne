/**
 * @file memory_benchmark.cpp
 * @brief Memory efficiency and allocation pattern benchmark
 */

#include <psyne/psyne.hpp>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <iomanip>

namespace psyne_bench {

class MemoryMessage : public psyne::Message<MemoryMessage> {
public:
    static constexpr uint32_t message_type = 500;
    using Message<MemoryMessage>::Message;
    
    static size_t calculate_size() { return 64 * 1024 * 1024; } // 64MB max
};

void test_memory_pattern(const std::string& name, size_t buffer_size, size_t message_size, size_t num_messages) {
    std::cout << "Testing " << name << " (buffer: " << (buffer_size/1024/1024) << "MB, "
              << "msg: " << message_size << "B, count: " << num_messages << ")..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        auto channel = psyne::create_channel("memory://memory_test", buffer_size, psyne::ChannelMode::SPSC);
        
        std::atomic<size_t> total_received{0};
        
        // Consumer
        std::thread consumer([&]() {
            while (total_received < num_messages) {
                auto msg_opt = channel->receive<MemoryMessage>(std::chrono::milliseconds(10));
                if (msg_opt) {
                    total_received++;
                }
            }
        });
        
        // Producer
        std::thread producer([&]() {
            for (size_t i = 0; i < num_messages; ++i) {
                bool sent = false;
                while (!sent) {
                    try {
                        MemoryMessage msg(*channel);
                        // Fill with test pattern
                        if (msg.is_valid() && msg.size() >= message_size) {
                            uint8_t* data = msg.data();
                            for (size_t j = 0; j < std::min(message_size, msg.size()); ++j) {
                                data[j] = static_cast<uint8_t>(j % 256);
                            }
                        }
                        msg.send();
                        sent = true;
                    } catch (...) {
                        std::this_thread::yield();
                    }
                }
            }
        });
        
        producer.join();
        consumer.join();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(end_time - start_time).count();
        
        size_t total_bytes = total_received * message_size;
        double throughput = (total_bytes / (1024.0 * 1024.0)) / elapsed;
        
        std::cout << "  Completed: " << total_received << "/" << num_messages << " messages\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << throughput << " MB/s\n";
        std::cout << "  Elapsed: " << std::fixed << std::setprecision(2) << elapsed << " seconds\n\n";
        
    } catch (const std::exception& e) {
        std::cout << "  FAILED: " << e.what() << "\n\n";
    }
}

} // namespace psyne_bench

int main() {
    std::cout << "=== Psyne Memory Efficiency Benchmark ===\n\n";
    
    using namespace psyne_bench;
    
    // Test different buffer sizes and message patterns
    test_memory_pattern("Small Messages", 64*1024*1024, 1024, 50000);
    test_memory_pattern("Medium Messages", 128*1024*1024, 64*1024, 10000);
    test_memory_pattern("Large Messages", 256*1024*1024, 1024*1024, 1000);
    test_memory_pattern("Huge Messages", 512*1024*1024, 16*1024*1024, 100);
    
    std::cout << "Memory benchmark completed!\n";
    return 0;
}