/**
 * @file scalability_benchmark.cpp
 * @brief Scalability benchmark testing performance as thread count increases
 */

#include <psyne/psyne.hpp>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <iomanip>
#include <atomic>

namespace psyne_bench {

class ScalabilityMessage : public psyne::Message<ScalabilityMessage> {
public:
    static constexpr uint32_t message_type = 600;
    using Message<ScalabilityMessage>::Message;
    
    static size_t calculate_size() { return 16 * 1024 * 1024; } // 16MB max
};

struct ScalabilityResult {
    size_t thread_count;
    double throughput_mbps;
    double message_rate;
    double efficiency_ratio; // Throughput per thread
};

ScalabilityResult test_scalability(const std::string& pattern, psyne::ChannelMode mode,
                                  size_t producers, size_t consumers, size_t message_size) {
    
    std::cout << "Testing " << pattern << " with " << producers << "P/" << consumers 
              << "C, " << message_size << "B messages..." << std::flush;
    
    size_t buffer_size = std::max<size_t>(256*1024*1024, message_size * 1000);
    size_t messages_per_producer = 10000;
    
    auto channel = psyne::create_channel("memory://scalability_test", buffer_size, mode);
    
    std::atomic<size_t> total_sent{0};
    std::atomic<size_t> total_received{0};
    std::atomic<size_t> producers_done{0};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Consumers
    std::vector<std::thread> consumer_threads;
    for (size_t i = 0; i < consumers; ++i) {
        consumer_threads.emplace_back([&]() {
            while (producers_done < producers || total_received < total_sent) {
                auto msg_opt = channel->receive<ScalabilityMessage>(std::chrono::milliseconds(1));
                if (msg_opt) {
                    total_received++;
                }
            }
        });
    }
    
    // Producers
    std::vector<std::thread> producer_threads;
    for (size_t i = 0; i < producers; ++i) {
        producer_threads.emplace_back([&, i]() {
            std::vector<uint8_t> test_data(message_size, static_cast<uint8_t>(i % 256));
            
            for (size_t j = 0; j < messages_per_producer; ++j) {
                bool sent = false;
                while (!sent) {
                    try {
                        ScalabilityMessage msg(*channel);
                        if (msg.is_valid() && msg.size() >= message_size) {
                            std::memcpy(msg.data(), test_data.data(), 
                                       std::min(message_size, msg.size()));
                        }
                        msg.send();
                        total_sent++;
                        sent = true;
                    } catch (...) {
                        std::this_thread::yield();
                    }
                }
            }
            producers_done++;
        });
    }
    
    // Wait for completion
    for (auto& t : producer_threads) t.join();
    for (auto& t : consumer_threads) t.join();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    
    ScalabilityResult result;
    result.thread_count = producers + consumers;
    
    size_t total_bytes = total_received * message_size;
    result.throughput_mbps = (total_bytes / (1024.0 * 1024.0)) / elapsed;
    result.message_rate = total_received / elapsed;
    result.efficiency_ratio = result.throughput_mbps / result.thread_count;
    
    std::cout << " " << std::fixed << std::setprecision(1) 
              << result.throughput_mbps << " MB/s (" 
              << result.efficiency_ratio << " MB/s per thread)" << std::endl;
    
    return result;
}

} // namespace psyne_bench

int main() {
    std::cout << "=== Psyne Scalability Benchmark ===\n\n";
    
    using namespace psyne_bench;
    
    std::vector<ScalabilityResult> results;
    size_t message_size = 4096;
    
    std::cout << "MPSC Scaling (message size: " << message_size << "B):\n";
    for (size_t p = 1; p <= 8; p *= 2) {
        auto result = test_scalability("MPSC-" + std::to_string(p) + "P", 
                                     psyne::ChannelMode::MPSC, p, 1, message_size);
        results.push_back(result);
    }
    
    std::cout << "\nSPMC Scaling:\n";
    for (size_t c = 1; c <= 8; c *= 2) {
        auto result = test_scalability("SPMC-" + std::to_string(c) + "C", 
                                     psyne::ChannelMode::SPMC, 1, c, message_size);
        results.push_back(result);
    }
    
    std::cout << "\nMPMC Scaling:\n";
    for (size_t pc = 1; pc <= 4; pc *= 2) {
        auto result = test_scalability("MPMC-" + std::to_string(pc) + "x" + std::to_string(pc), 
                                     psyne::ChannelMode::MPMC, pc, pc, message_size);
        results.push_back(result);
    }
    
    // Summary
    std::cout << "\n=== Scalability Summary ===\n";
    std::cout << std::setw(15) << "Pattern" 
              << std::setw(10) << "Threads"
              << std::setw(15) << "Throughput (MB/s)"
              << std::setw(20) << "Efficiency (MB/s/thread)\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& result : results) {
        std::cout << std::setw(15) << ""  // Pattern name would go here
                  << std::setw(10) << result.thread_count
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.throughput_mbps
                  << std::setw(20) << std::fixed << std::setprecision(2) << result.efficiency_ratio << "\n";
    }
    
    std::cout << "\nScalability benchmark completed!\n";
    return 0;
}