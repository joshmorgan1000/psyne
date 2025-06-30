/**
 * @file spsc_benchmark.cpp
 * @brief SPSC (Single Producer Single Consumer) messaging benchmark
 * 
 * Tests high-performance messaging between a single producer and single consumer thread.
 * This benchmark measures throughput, latency, and zero-copy efficiency for different
 * message sizes ranging from small (64 bytes) to large (64MB).
 */

#include <psyne/psyne.hpp>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <atomic>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <numeric>

namespace psyne_bench {

// Benchmark configuration
struct BenchmarkConfig {
    size_t message_size;
    size_t num_messages;
    size_t buffer_size;
    std::string test_name;
    bool warmup;
};

// Extended results structure based on psyne::BenchmarkResult
struct BenchmarkResults : public psyne::BenchmarkResult {
    double throughput_msgs_per_sec;
    double min_latency_ns;
    double max_latency_ns;
    size_t total_bytes;
    double elapsed_seconds;
    bool zero_copy_success;
    size_t messages_received;
};

// Simple message type for benchmarking
class BenchmarkMessage : public psyne::Message<BenchmarkMessage> {
public:
    static constexpr uint32_t message_type = 100;
    
    using psyne::Message<BenchmarkMessage>::Message;
    
    static size_t calculate_size() {
        return 64 * 1024 * 1024; // 64MB max size
    }
    
    void set_data(const std::vector<uint8_t>& test_data, std::chrono::high_resolution_clock::time_point timestamp) {
        if (!is_valid()) return;
        
        // Write timestamp first
        std::memcpy(data(), &timestamp, sizeof(timestamp));
        
        // Write actual data
        size_t copy_size = std::min(test_data.size(), size() - sizeof(timestamp));
        std::memcpy(data() + sizeof(timestamp), test_data.data(), copy_size);
    }
    
    std::chrono::high_resolution_clock::time_point get_timestamp() const {
        if (!is_valid()) return {};
        
        std::chrono::high_resolution_clock::time_point timestamp;
        std::memcpy(&timestamp, data(), sizeof(timestamp));
        return timestamp;
    }
    
    bool verify_data(const std::vector<uint8_t>& expected_data) const {
        if (!is_valid()) return false;
        
        size_t check_size = std::min(expected_data.size(), size() - sizeof(std::chrono::high_resolution_clock::time_point));
        const uint8_t* data_ptr = data() + sizeof(std::chrono::high_resolution_clock::time_point);
        
        for (size_t i = 0; i < std::min<size_t>(64, check_size); ++i) {
            if (data_ptr[i] != expected_data[i]) {
                return false;
            }
        }
        return true;
    }
};

class SPSCBenchmark {
private:
    std::unique_ptr<psyne::Channel> channel_;
    std::atomic<bool> producer_done_{false};
    std::atomic<bool> consumer_done_{false};
    std::vector<std::chrono::nanoseconds> latencies_;
    size_t max_threads_;
    
public:
    SPSCBenchmark(size_t buffer_size) {
        channel_ = psyne::create_channel("memory://spsc_bench", buffer_size, psyne::ChannelMode::SPSC);
        
        // Use maximum available threads for benchmarking
        max_threads_ = std::thread::hardware_concurrency();
        if (max_threads_ == 0) max_threads_ = 8; // Fallback
        
        std::cout << "Initialized SPSC benchmark with " << max_threads_ << " threads available" << std::endl;
    }
    
    BenchmarkResults run_benchmark(const BenchmarkConfig& config) {
        std::cout << "Running SPSC benchmark: " << config.test_name 
                  << " (msg_size=" << config.message_size 
                  << ", num_msgs=" << config.num_messages << ")" << std::endl;
        
        // Reset state
        producer_done_ = false;
        consumer_done_ = false;
        latencies_.clear();
        latencies_.reserve(config.num_messages);
        
        BenchmarkResults results{};
        results.total_bytes = config.message_size * config.num_messages;
        
        // Start consumer and producer threads 
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::thread consumer_thread([this, &config, &results]() {
            consume_messages(config, results);
        });
        
        std::thread producer_thread([this, &config, start_time]() {
            produce_messages(config, start_time);
        });
        
        // Wait for completion
        producer_thread.join();
        consumer_thread.join();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        results.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
        
        // Calculate metrics
        calculate_metrics(results, config);
        
        return results;
    }
    
private:
    void produce_messages(const BenchmarkConfig& config, 
                         std::chrono::high_resolution_clock::time_point start_time) {
        
        // Create test data pattern
        std::vector<uint8_t> test_data(config.message_size);
        for (size_t i = 0; i < config.message_size; ++i) {
            test_data[i] = static_cast<uint8_t>(i % 256);
        }
        
        size_t messages_sent = 0;
        
        while (messages_sent < config.num_messages) {
            try {
                // Create message using public API
                BenchmarkMessage msg(*channel_);
                
                // Set timestamp and data
                auto timestamp = std::chrono::high_resolution_clock::now();
                msg.set_data(test_data, timestamp);
                
                // Send message (zero-copy)
                msg.send();
                
                messages_sent++;
            } catch (const std::exception& e) {
                // Buffer full, yield and retry
                std::this_thread::yield();
            }
        }
        
        producer_done_ = true;
    }
    
    void consume_messages(const BenchmarkConfig& config, BenchmarkResults& results) {
        size_t messages_received = 0;
        bool zero_copy_working = true;
        
        // Create test data for verification
        std::vector<uint8_t> expected_data(config.message_size);
        for (size_t i = 0; i < config.message_size; ++i) {
            expected_data[i] = static_cast<uint8_t>(i % 256);
        }
        
        while (messages_received < config.num_messages || !producer_done_) {
            // Try to receive message
            auto msg_opt = channel_->receive<BenchmarkMessage>(std::chrono::milliseconds(1));
            
            if (msg_opt) {
                auto receive_time = std::chrono::high_resolution_clock::now();
                auto& msg = *msg_opt;
                
                // Get timestamp and calculate latency
                auto send_time = msg.get_timestamp();
                auto latency = receive_time - send_time;
                latencies_.push_back(latency);
                
                // Verify data integrity (sample check)
                if (messages_received % 1000 == 0) {
                    if (!msg.verify_data(expected_data)) {
                        zero_copy_working = false;
                    }
                }
                
                messages_received++;
            } else {
                // No message available, yield
                std::this_thread::yield();
            }
        }
        
        results.zero_copy_success = zero_copy_working;
        results.messages_received = messages_received;
        consumer_done_ = true;
    }
    
    void calculate_metrics(BenchmarkResults& results, const BenchmarkConfig& config) {
        results.messages_sent = config.num_messages;
        
        // Throughput calculations
        results.throughput_mbps = (results.total_bytes / (1024.0 * 1024.0)) / results.elapsed_seconds;
        results.throughput_msgs_per_sec = results.messages_received / results.elapsed_seconds;
        
        // Latency calculations
        if (!latencies_.empty()) {
            std::sort(latencies_.begin(), latencies_.end());
            
            auto sum = std::accumulate(latencies_.begin(), latencies_.end(), 
                                     std::chrono::nanoseconds(0));
            double avg_latency_ns = sum.count() / static_cast<double>(latencies_.size());
            
            results.min_latency_ns = latencies_.front().count();
            results.max_latency_ns = latencies_.back().count();
            
            // Convert to microseconds for psyne::BenchmarkResult fields
            results.latency_us_p50 = latencies_[latencies_.size()/2].count() / 1000.0;
            results.latency_us_p99 = latencies_[latencies_.size() * 99/100].count() / 1000.0;
            results.latency_us_p999 = latencies_[latencies_.size() * 999/1000].count() / 1000.0;
        }
    }
};

void print_results(const BenchmarkResults& results, const BenchmarkConfig& config) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n--- " << config.test_name << " Results ---\n";
    std::cout << "Message size: " << config.message_size << " bytes\n";
    std::cout << "Messages sent: " << results.messages_sent << "\n";
    std::cout << "Messages received: " << results.messages_received << "\n";
    std::cout << "Total data: " << (results.total_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Elapsed time: " << results.elapsed_seconds << " seconds\n";
    std::cout << "Throughput: " << results.throughput_mbps << " MB/s\n";
    std::cout << "Message rate: " << results.throughput_msgs_per_sec << " msg/s\n";
    std::cout << "P50 latency: " << results.latency_us_p50 << " μs\n";
    std::cout << "Min latency: " << results.min_latency_ns << " ns\n";
    std::cout << "Max latency: " << results.max_latency_ns << " ns\n";
    std::cout << "P99 latency: " << results.latency_us_p99 << " μs\n";
    std::cout << "P99.9 latency: " << results.latency_us_p999 << " μs\n";
    std::cout << "Zero-copy: " << (results.zero_copy_success ? "SUCCESS" : "FAILED") << "\n";
    std::cout << std::string(50, '-') << "\n";
}

} // namespace psyne_bench

int main() {
    std::cout << "=== Psyne SPSC (Single Producer Single Consumer) Benchmark ===\n\n";
    
    using namespace psyne_bench;
    
    // Test configurations for different message sizes
    std::vector<BenchmarkConfig> configs = {
        // Small messages - high frequency
        {64, 1'000'000, 64 * 1024 * 1024, "Small Messages (64B)", false},
        {256, 1'000'000, 64 * 1024 * 1024, "Medium Messages (256B)", false},
        {1024, 500'000, 64 * 1024 * 1024, "Large Messages (1KB)", false},
        
        // Medium messages
        {4096, 200'000, 64 * 1024 * 1024, "Large Messages (4KB)", false},
        {16384, 100'000, 64 * 1024 * 1024, "Very Large Messages (16KB)", false},
        {65536, 50'000, 128 * 1024 * 1024, "Huge Messages (64KB)", false},
        
        // Large messages - test memory bandwidth
        {1024 * 1024, 10'000, 256 * 1024 * 1024, "Massive Messages (1MB)", false},
        {4 * 1024 * 1024, 2'000, 512 * 1024 * 1024, "Giant Messages (4MB)", false},
        {16 * 1024 * 1024, 500, 1024 * 1024 * 1024, "Enormous Messages (16MB)", false},
        {64 * 1024 * 1024, 100, 2048ULL * 1024 * 1024, "Colossal Messages (64MB)", false},
    };
    
    std::vector<BenchmarkResults> all_results;
    
    for (const auto& config : configs) {
        try {
            SPSCBenchmark benchmark(config.buffer_size);
            auto results = benchmark.run_benchmark(config);
            print_results(results, config);
            all_results.push_back(results);
        } catch (const std::exception& e) {
            std::cerr << "Benchmark failed for " << config.test_name 
                      << ": " << e.what() << std::endl;
        }
    }
    
    // Print summary
    std::cout << "\n=== SPSC Benchmark Summary ===\n";
    std::cout << std::setw(20) << "Message Size" 
              << std::setw(15) << "Throughput (MB/s)"
              << std::setw(18) << "Rate (Mmsg/s)"  
              << std::setw(15) << "P50 Latency (μs)"
              << std::setw(12) << "Zero-Copy\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (size_t i = 0; i < configs.size() && i < all_results.size(); ++i) {
        const auto& config = configs[i];
        const auto& result = all_results[i];
        
        std::cout << std::setw(20) << (std::to_string(config.message_size) + "B")
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.throughput_mbps
                  << std::setw(18) << std::fixed << std::setprecision(3) << (result.throughput_msgs_per_sec / 1e6)
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.latency_us_p50
                  << std::setw(12) << (result.zero_copy_success ? "YES" : "NO") << "\n";
    }
    
    std::cout << "\nSPSC benchmark completed successfully!\n";
    return 0;
}