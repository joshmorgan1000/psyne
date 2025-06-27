#include <psyne/psyne.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <iomanip>

using namespace psyne;
using namespace std::chrono;

// Benchmark configuration
struct BenchmarkConfig {
    size_t message_size;
    size_t num_messages;
    size_t buffer_size;
    std::string channel_type;
};

// Results structure
struct BenchmarkResult {
    double throughput_mbps;
    double messages_per_sec;
    double avg_latency_us;
    double p99_latency_us;
    size_t messages_sent;
    size_t messages_received;
    duration<double> total_time;
};

template<typename ChannelType>
BenchmarkResult run_benchmark(const BenchmarkConfig& config) {
    // Create channel
    ChannelType channel("memory://bench", config.buffer_size, ChannelType::SingleType);
    
    std::atomic<bool> done{false};
    std::atomic<size_t> messages_received{0};
    std::vector<double> latencies;
    latencies.reserve(config.num_messages);
    
    // Consumer thread
    std::thread consumer([&]() {
        while (!done || messages_received < config.num_messages) {
            auto start = high_resolution_clock::now();
            auto msg = channel.template receive_single<FloatVector>(milliseconds(10));
            if (msg) {
                auto end = high_resolution_clock::now();
                auto latency = duration_cast<microseconds>(end - start).count();
                latencies.push_back(latency);
                messages_received++;
            }
        }
    });
    
    // Producer
    auto start_time = high_resolution_clock::now();
    
    for (size_t i = 0; i < config.num_messages; ++i) {
        FloatVector msg(channel);
        msg.resize(config.message_size / sizeof(float));
        
        // Fill with test pattern
        for (size_t j = 0; j < msg.size(); ++j) {
            msg[j] = static_cast<float>(i + j);
        }
        
        channel.send(msg);
    }
    
    done = true;
    consumer.join();
    
    auto end_time = high_resolution_clock::now();
    auto total_time = duration_cast<duration<double>>(end_time - start_time);
    
    // Calculate results
    BenchmarkResult result;
    result.messages_sent = config.num_messages;
    result.messages_received = messages_received;
    result.total_time = total_time;
    
    double total_bytes = config.num_messages * config.message_size;
    result.throughput_mbps = (total_bytes / (1024.0 * 1024.0)) / total_time.count();
    result.messages_per_sec = config.num_messages / total_time.count();
    
    // Calculate latency statistics
    if (!latencies.empty()) {
        double sum = 0;
        for (double l : latencies) sum += l;
        result.avg_latency_us = sum / latencies.size();
        
        // P99 latency
        std::sort(latencies.begin(), latencies.end());
        size_t p99_idx = static_cast<size_t>(latencies.size() * 0.99);
        result.p99_latency_us = latencies[p99_idx];
    }
    
    return result;
}

void print_result(const std::string& name, const BenchmarkConfig& config, const BenchmarkResult& result) {
    std::cout << "\n=== " << name << " ===" << std::endl;
    std::cout << "Message size: " << config.message_size << " bytes" << std::endl;
    std::cout << "Buffer size: " << config.buffer_size / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Messages sent: " << result.messages_sent << std::endl;
    std::cout << "Messages received: " << result.messages_received << std::endl;
    std::cout << "Total time: " << result.total_time.count() << " seconds" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) 
              << result.throughput_mbps << " MB/s" << std::endl;
    std::cout << "Messages/sec: " << std::fixed << std::setprecision(0)
              << result.messages_per_sec << std::endl;
    std::cout << "Avg latency: " << std::fixed << std::setprecision(2)
              << result.avg_latency_us << " μs" << std::endl;
    std::cout << "P99 latency: " << std::fixed << std::setprecision(2)
              << result.p99_latency_us << " μs" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Psyne Throughput Benchmark" << std::endl;
    std::cout << "==========================" << std::endl;
    
    // Test configurations
    std::vector<BenchmarkConfig> configs = {
        // Small messages, high frequency
        {64, 1000000, 8 * 1024 * 1024, "SPSC"},
        {256, 500000, 8 * 1024 * 1024, "SPSC"},
        {1024, 200000, 8 * 1024 * 1024, "SPSC"},
        
        // Large messages
        {4096, 50000, 16 * 1024 * 1024, "SPSC"},
        {16384, 10000, 32 * 1024 * 1024, "SPSC"},
        {65536, 2500, 64 * 1024 * 1024, "SPSC"},
    };
    
    // Run benchmarks
    for (const auto& config : configs) {
        std::cout << "\nRunning benchmark: " << config.message_size << " byte messages..." << std::endl;
        
        if (config.channel_type == "SPSC") {
            auto result = run_benchmark<SPSCChannel>(config);
            print_result("SPSC Channel", config, result);
        }
        
        // Small delay between benchmarks
        std::this_thread::sleep_for(milliseconds(100));
    }
    
    // Also test other channel types with medium-sized messages
    BenchmarkConfig medium_config = {1024, 100000, 16 * 1024 * 1024, ""};
    
    std::cout << "\n\nComparing Channel Types (1KB messages):" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // SPSC
    auto spsc_result = run_benchmark<SPSCChannel>(medium_config);
    print_result("SPSC", medium_config, spsc_result);
    
    // SPMC
    auto spmc_result = run_benchmark<SPMCChannel>(medium_config);
    print_result("SPMC", medium_config, spmc_result);
    
    // MPSC
    auto mpsc_result = run_benchmark<MPSCChannel>(medium_config);
    print_result("MPSC", medium_config, mpsc_result);
    
    // MPMC
    auto mpmc_result = run_benchmark<MPMCChannel>(medium_config);
    print_result("MPMC", medium_config, mpmc_result);
    
    return 0;
}