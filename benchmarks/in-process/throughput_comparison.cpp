/**
 * @file throughput_comparison.cpp
 * @brief Comprehensive throughput comparison across all messaging patterns
 * 
 * Compares SPSC, MPSC, SPMC, and MPMC patterns side-by-side to understand
 * the performance characteristics and trade-offs of each approach.
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
#include <mutex>
#include <barrier>
#include <numeric>

namespace psyne_bench {

// Generic benchmark message
class ThroughputMessage : public psyne::Message<ThroughputMessage> {
public:
    static constexpr uint32_t message_type = 300;
    
    using psyne::Message<ThroughputMessage>::Message;
    
    static size_t calculate_size() {
        return 16 * 1024 * 1024; // 16MB max size
    }
    
    void set_payload(size_t producer_id, const std::vector<uint8_t>& data, 
                     std::chrono::high_resolution_clock::time_point timestamp) {
        if (!is_valid()) return;
        
        uint8_t* ptr = this->data();
        
        // Write timestamp
        std::memcpy(ptr, &timestamp, sizeof(timestamp));
        ptr += sizeof(timestamp);
        
        // Write producer ID  
        std::memcpy(ptr, &producer_id, sizeof(producer_id));
        ptr += sizeof(producer_id);
        
        // Write payload
        size_t remaining = size() - sizeof(timestamp) - sizeof(producer_id);
        size_t copy_size = std::min(data.size(), remaining);
        std::memcpy(ptr, data.data(), copy_size);
    }
    
    std::chrono::high_resolution_clock::time_point get_timestamp() const {
        if (!is_valid()) return {};
        
        std::chrono::high_resolution_clock::time_point timestamp;
        std::memcpy(&timestamp, data(), sizeof(timestamp));
        return timestamp;
    }
    
    size_t get_producer_id() const {
        if (!is_valid()) return SIZE_MAX;
        
        size_t producer_id;
        std::memcpy(&producer_id, data() + sizeof(std::chrono::high_resolution_clock::time_point), sizeof(producer_id));
        return producer_id;
    }
};

struct BenchmarkResult {
    std::string pattern_name;
    size_t message_size;
    size_t num_producers;
    size_t num_consumers;
    double throughput_mbps;
    double message_rate;
    double avg_latency_ns;
    double elapsed_seconds;
    size_t total_messages;
    bool success;
};

class PatternBenchmark {
private:
    std::unique_ptr<psyne::Channel> channel_;
    std::atomic<size_t> producers_done_{0};
    std::atomic<size_t> consumers_done_{0};
    std::atomic<size_t> total_sent_{0};
    std::atomic<size_t> total_received_{0};
    std::vector<std::chrono::nanoseconds> latencies_;
    std::mutex latency_mutex_;
    size_t max_threads_;
    
public:
    PatternBenchmark(const std::string& name, size_t buffer_size, psyne::ChannelMode mode) {
        channel_ = psyne::create_channel("memory://" + name, buffer_size, mode);
        
        // Use maximum available threads for benchmarking
        max_threads_ = std::thread::hardware_concurrency();
        if (max_threads_ == 0) max_threads_ = 8; // Fallback;
    }
    
    BenchmarkResult run_pattern(const std::string& pattern_name, size_t message_size, 
                               size_t num_producers, size_t num_consumers, 
                               size_t messages_per_producer, size_t buffer_size) {
        
        std::cout << "Testing " << pattern_name << " pattern: " 
                  << num_producers << "P/" << num_consumers << "C, "
                  << message_size << "B messages..." << std::flush;
        
        // Reset state
        producers_done_ = 0;
        consumers_done_ = 0;
        total_sent_ = 0;
        total_received_ = 0;
        latencies_.clear();
        
        BenchmarkResult result;
        result.pattern_name = pattern_name;
        result.message_size = message_size;
        result.num_producers = num_producers;
        result.num_consumers = num_consumers;
        result.total_messages = num_producers * messages_per_producer;
        
        try {
            // Synchronization barrier for simultaneous start
            std::barrier start_barrier(num_producers + num_consumers + 1);
            
            // Start consumer threads
            std::vector<std::thread> consumers;
            for (size_t i = 0; i < num_consumers; ++i) {
                consumers.emplace_back([this, &start_barrier, message_size, messages_per_producer, num_producers]() {
                    start_barrier.arrive_and_wait();
                    consume_messages(message_size, messages_per_producer * num_producers);
                });
            }
            
            // Start producer threads
            auto start_time = std::chrono::high_resolution_clock::now();
            std::vector<std::thread> producers;
            for (size_t i = 0; i < num_producers; ++i) {
                producers.emplace_back([this, &start_barrier, i, message_size, messages_per_producer]() {
                    start_barrier.arrive_and_wait();
                    produce_messages(i, message_size, messages_per_producer);
                });
            }
            
            // Start benchmark
            start_barrier.arrive_and_wait();
            
            // Wait for completion
            for (auto& producer : producers) {
                producer.join();
            }
            for (auto& consumer : consumers) {
                consumer.join();
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
            
            // Calculate metrics
            size_t total_bytes = total_received_.load() * message_size;
            result.throughput_mbps = (total_bytes / (1024.0 * 1024.0)) / result.elapsed_seconds;
            result.message_rate = total_received_.load() / result.elapsed_seconds;
            
            if (!latencies_.empty()) {
                auto sum = std::accumulate(latencies_.begin(), latencies_.end(), std::chrono::nanoseconds(0));
                result.avg_latency_ns = sum.count() / static_cast<double>(latencies_.size());
            }
            
            result.success = (total_received_.load() >= result.total_messages * 0.95); // 95% delivery rate
            
            std::cout << " " << std::fixed << std::setprecision(1) 
                      << result.throughput_mbps << " MB/s" << std::endl;
            
        } catch (const std::exception& e) {
            result.success = false;
            std::cout << " FAILED: " << e.what() << std::endl;
        }
        
        return result;
    }
    
private:
    void produce_messages(size_t producer_id, size_t message_size, size_t num_messages) {
        std::vector<uint8_t> test_data(message_size);
        for (size_t i = 0; i < message_size; ++i) {
            test_data[i] = static_cast<uint8_t>((producer_id + i) % 256);
        }
        
        size_t sent = 0;
        while (sent < num_messages) {
            try {
                ThroughputMessage msg(*channel_);
                auto timestamp = std::chrono::high_resolution_clock::now();
                msg.set_payload(producer_id, test_data, timestamp);
                msg.send();
                sent++;
                total_sent_.fetch_add(1);
            } catch (const std::exception&) {
                std::this_thread::yield();
            }
        }
        
        producers_done_.fetch_add(1);
    }
    
    void consume_messages(size_t message_size, size_t expected_messages) {
        size_t received = 0;
        
        while (received < expected_messages && 
               (producers_done_.load() < 1 || received < total_sent_.load())) {
            
            auto msg_opt = channel_->receive<ThroughputMessage>(std::chrono::milliseconds(10));
            
            if (msg_opt) {
                auto receive_time = std::chrono::high_resolution_clock::now();
                auto& msg = *msg_opt;
                
                auto send_time = msg.get_timestamp();
                auto latency = receive_time - send_time;
                
                {
                    std::lock_guard<std::mutex> lock(latency_mutex_);
                    latencies_.push_back(latency);
                }
                
                received++;
                total_received_.fetch_add(1);
            } else {
                std::this_thread::yield();
            }
        }
        
        consumers_done_.fetch_add(1);
    }
};

} // namespace psyne_bench

int main() {
    std::cout << "=== Psyne Throughput Comparison Benchmark ===\n\n";
    
    using namespace psyne_bench;
    
    // Test configurations
    struct TestConfig {
        std::string pattern;
        psyne::ChannelMode mode;
        size_t producers;
        size_t consumers;
    };
    
    std::vector<TestConfig> patterns = {
        {"SPSC", psyne::ChannelMode::SPSC, 1, 1},
        {"MPSC-2P", psyne::ChannelMode::MPSC, 2, 1},
        {"MPSC-4P", psyne::ChannelMode::MPSC, 4, 1},
        {"SPMC-2C", psyne::ChannelMode::SPMC, 1, 2},
        {"SPMC-4C", psyne::ChannelMode::SPMC, 1, 4},
        {"MPMC-2x2", psyne::ChannelMode::MPMC, 2, 2},
        {"MPMC-4x4", psyne::ChannelMode::MPMC, 4, 4},
    };
    
    std::vector<size_t> message_sizes = {1024, 16384, 1024*1024};
    std::vector<BenchmarkResult> all_results;
    
    for (size_t msg_size : message_sizes) {
        std::cout << "\n=== Message Size: " << msg_size << " bytes ===\n";
        
        for (const auto& pattern : patterns) {
            try {
                size_t buffer_size = std::max<size_t>(64 * 1024 * 1024, msg_size * 1000);
                size_t messages_per_producer = std::max<size_t>(1000, 10000000 / msg_size);
                
                PatternBenchmark benchmark(pattern.pattern + "_" + std::to_string(msg_size), 
                                         buffer_size, pattern.mode);
                
                auto result = benchmark.run_pattern(pattern.pattern, msg_size, 
                                                   pattern.producers, pattern.consumers,
                                                   messages_per_producer, buffer_size);
                all_results.push_back(result);
                
            } catch (const std::exception& e) {
                std::cerr << "Failed to test " << pattern.pattern << ": " << e.what() << std::endl;
            }
        }
    }
    
    // Print comprehensive summary
    std::cout << "\n=== Comprehensive Throughput Comparison ===\n";
    std::cout << std::setw(12) << "Pattern" 
              << std::setw(10) << "Msg Size"
              << std::setw(8) << "P/C"
              << std::setw(15) << "Throughput (MB/s)"
              << std::setw(18) << "Rate (Kmsg/s)"
              << std::setw(15) << "Avg Lat (ns)"
              << std::setw(10) << "Status\n";
    std::cout << std::string(90, '-') << "\n";
    
    for (const auto& result : all_results) {
        if (result.success) {
            std::string pc_ratio = std::to_string(result.num_producers) + "/" + std::to_string(result.num_consumers);
            std::string msg_size_str = (result.message_size >= 1024*1024) ? 
                std::to_string(result.message_size/(1024*1024)) + "MB" :
                (result.message_size >= 1024) ? 
                std::to_string(result.message_size/1024) + "KB" :
                std::to_string(result.message_size) + "B";
            
            std::cout << std::setw(12) << result.pattern_name
                      << std::setw(10) << msg_size_str
                      << std::setw(8) << pc_ratio
                      << std::setw(15) << std::fixed << std::setprecision(1) << result.throughput_mbps
                      << std::setw(18) << std::fixed << std::setprecision(1) << (result.message_rate / 1000.0)
                      << std::setw(15) << std::fixed << std::setprecision(0) << result.avg_latency_ns
                      << std::setw(10) << "OK" << "\n";
        }
    }
    
    // Performance insights
    std::cout << "\n=== Performance Insights ===\n";
    
    // Find best pattern for each message size
    for (size_t msg_size : message_sizes) {
        std::vector<BenchmarkResult> size_results;
        std::copy_if(all_results.begin(), all_results.end(), std::back_inserter(size_results),
                    [msg_size](const BenchmarkResult& r) { return r.message_size == msg_size && r.success; });
        
        if (!size_results.empty()) {
            auto best = std::max_element(size_results.begin(), size_results.end(),
                [](const BenchmarkResult& a, const BenchmarkResult& b) {
                    return a.throughput_mbps < b.throughput_mbps;
                });
            
            std::cout << "Best for " << msg_size << "B messages: " << best->pattern_name 
                      << " (" << std::fixed << std::setprecision(1) << best->throughput_mbps << " MB/s)\n";
        }
    }
    
    std::cout << "\nThroughput comparison completed successfully!\n";
    return 0;
}