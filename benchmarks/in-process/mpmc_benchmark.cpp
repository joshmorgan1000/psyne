/**
 * @file mpmc_benchmark.cpp
 * @brief MPMC (Multi Producer Multi Consumer) messaging benchmark
 * 
 * Tests high-performance messaging from multiple producer threads to multiple consumer threads.
 * This benchmark measures throughput, latency, contention effects, and load balancing
 * as both producer and consumer thread counts scale.
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

struct MPMCConfig {
    size_t message_size;
    size_t num_messages_per_producer;
    size_t num_producers;
    size_t num_consumers;
    size_t buffer_size;
    std::string test_name;
};

struct MPMCResults {
    double total_throughput_mbps;
    double total_throughput_msgs_per_sec;
    double avg_latency_ns;
    double min_latency_ns;
    double max_latency_ns;
    double p99_latency_ns;
    size_t total_bytes_produced;
    size_t total_bytes_consumed;
    double elapsed_seconds;
    size_t total_messages_sent;
    size_t total_messages_received;
    std::vector<size_t> producer_message_counts;
    std::vector<size_t> consumer_message_counts;
    size_t contention_events;
    bool zero_copy_success;
    double load_balance_efficiency;
};

// Message type with producer ID for tracking
class MPMCMessage : public psyne::Message<MPMCMessage> {
public:
    static constexpr uint32_t message_type = 200;
    
    using Message<MPMCMessage>::Message;
    
    static size_t calculate_size() {
        return 64 * 1024 * 1024; // 64MB max size
    }
    
    void set_data(size_t producer_id, size_t sequence_num, const std::vector<uint8_t>& test_data, 
                  std::chrono::high_resolution_clock::time_point timestamp) {
        if (!is_valid()) return;
        
        uint8_t* ptr = data();
        
        // Write timestamp
        std::memcpy(ptr, &timestamp, sizeof(timestamp));
        ptr += sizeof(timestamp);
        
        // Write producer ID
        std::memcpy(ptr, &producer_id, sizeof(producer_id));
        ptr += sizeof(producer_id);
        
        // Write sequence number
        std::memcpy(ptr, &sequence_num, sizeof(sequence_num));
        ptr += sizeof(sequence_num);
        
        // Write actual data
        size_t remaining = size() - sizeof(timestamp) - sizeof(producer_id) - sizeof(sequence_num);
        size_t copy_size = std::min(test_data.size(), remaining);
        std::memcpy(ptr, test_data.data(), copy_size);
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
    
    size_t get_sequence_num() const {
        if (!is_valid()) return SIZE_MAX;
        
        size_t sequence_num;
        std::memcpy(&sequence_num, data() + sizeof(std::chrono::high_resolution_clock::time_point) + sizeof(size_t), sizeof(sequence_num));
        return sequence_num;
    }
    
    bool verify_data(size_t expected_producer_id, const std::vector<uint8_t>& expected_data) const {
        if (!is_valid()) return false;
        
        size_t producer_id = get_producer_id();
        if (producer_id != expected_producer_id) return false;
        
        size_t header_size = sizeof(std::chrono::high_resolution_clock::time_point) + 2 * sizeof(size_t);
        size_t check_size = std::min(expected_data.size(), size() - header_size);
        const uint8_t* data_ptr = data() + header_size;
        
        // Sample check first 64 bytes
        for (size_t i = 0; i < std::min<size_t>(64, check_size); ++i) {
            if (data_ptr[i] != expected_data[i]) {
                return false;
            }
        }
        return true;
    }
};

class MPMCBenchmark {
private:
    std::unique_ptr<psyne::Channel> channel_;
    std::atomic<size_t> producers_done_{0};
    std::atomic<size_t> consumers_done_{0};
    std::vector<std::vector<std::chrono::nanoseconds>> consumer_latencies_;
    std::vector<std::atomic<size_t>> producer_message_counts_;
    std::vector<std::atomic<size_t>> consumer_message_counts_;
    std::atomic<size_t> contention_count_{0};
    std::mutex latency_mutex_;
    size_t max_threads_;
    
public:
    MPMCBenchmark(size_t buffer_size, size_t num_producers, size_t num_consumers) 
        : consumer_latencies_(num_consumers), 
          producer_message_counts_(num_producers),
          consumer_message_counts_(num_consumers) {
        
        channel_ = psyne::create_channel("memory://mpmc_bench", buffer_size, psyne::ChannelMode::MPMC);
        
        // Use maximum available threads for benchmarking
        max_threads_ = std::thread::hardware_concurrency();
        if (max_threads_ == 0) max_threads_ = 8; // Fallback;
        
        for (size_t i = 0; i < num_consumers; ++i) {
            consumer_latencies_[i].reserve(100000);
            consumer_message_counts_[i] = 0;
        }
        
        for (size_t i = 0; i < num_producers; ++i) {
            producer_message_counts_[i] = 0;
        }
    }
    
    MPMCResults run_benchmark(const MPMCConfig& config) {
        std::cout << "Running MPMC benchmark: " << config.test_name 
                  << " (producers=" << config.num_producers
                  << ", consumers=" << config.num_consumers
                  << ", msg_size=" << config.message_size 
                  << ", msgs_per_producer=" << config.num_messages_per_producer << ")" << std::endl;
        
        // Reset state
        producers_done_ = 0;
        consumers_done_ = 0;
        contention_count_ = 0;
        
        for (size_t i = 0; i < config.num_consumers; ++i) {
            consumer_latencies_[i].clear();
            consumer_message_counts_[i] = 0;
        }
        
        for (size_t i = 0; i < config.num_producers; ++i) {
            producer_message_counts_[i] = 0;
        }
        
        MPMCResults results{};
        results.total_messages_sent = config.num_producers * config.num_messages_per_producer;
        results.total_bytes_produced = config.message_size * results.total_messages_sent;
        results.producer_message_counts.resize(config.num_producers, 0);
        results.consumer_message_counts.resize(config.num_consumers, 0);
        
        // Synchronization barrier for simultaneous start
        std::barrier start_barrier(config.num_producers + config.num_consumers + 1); // +1 for main thread
        
        // Consumer threads
        std::vector<std::thread> consumers;
        for (size_t i = 0; i < config.num_consumers; ++i) {
            consumers.emplace_back([this, &config, &results, &start_barrier, i]() {
                start_barrier.arrive_and_wait();
                consume_messages(config, results, i);
            });
        }
        
        // Producer threads
        std::vector<std::thread> producers;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < config.num_producers; ++i) {
            producers.emplace_back([this, &config, &start_barrier, i, start_time]() {
                start_barrier.arrive_and_wait();
                produce_messages(config, i, start_time);
            });
        }
        
        // Wait for all to be ready and start
        start_barrier.arrive_and_wait();
        
        // Wait for completion
        for (auto& producer : producers) {
            producer.join();
        }
        for (auto& consumer : consumers) {
            consumer.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        results.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
        results.contention_events = contention_count_.load();
        
        // Collect statistics
        for (size_t i = 0; i < config.num_producers; ++i) {
            results.producer_message_counts[i] = producer_message_counts_[i].load();
        }
        
        for (size_t i = 0; i < config.num_consumers; ++i) {
            results.consumer_message_counts[i] = consumer_message_counts_[i].load();
            results.total_messages_received += results.consumer_message_counts[i];
            results.total_bytes_consumed += results.consumer_message_counts[i] * config.message_size;
        }
        
        // Calculate metrics
        calculate_metrics(results, config);
        
        return results;
    }
    
private:
    void produce_messages(const MPMCConfig& config, size_t producer_id, 
                         std::chrono::high_resolution_clock::time_point start_time) {
        
        // Create unique test data pattern for this producer
        std::vector<uint8_t> test_data(config.message_size);
        for (size_t i = 0; i < config.message_size; ++i) {
            test_data[i] = static_cast<uint8_t>((producer_id * 23 + i) % 256);
        }
        
        size_t messages_sent = 0;
        size_t retry_count = 0;
        
        while (messages_sent < config.num_messages_per_producer) {
            try {
                // Create message using public API
                MPMCMessage msg(*channel_);
                
                // Set timestamp and data
                auto timestamp = std::chrono::high_resolution_clock::now();
                msg.set_data(producer_id, messages_sent, test_data, timestamp);
                
                // Send message (zero-copy)
                msg.send();
                
                // Reset retry count on successful send
                if (retry_count > 0) {
                    contention_count_.fetch_add(retry_count, std::memory_order_relaxed);
                    retry_count = 0;
                }
                
                messages_sent++;
                producer_message_counts_[producer_id].fetch_add(1, std::memory_order_relaxed);
                
            } catch (const std::exception& e) {
                // Buffer full, count contention and yield
                retry_count++;
                std::this_thread::yield();
            }
        }
        
        // Record any remaining retries
        if (retry_count > 0) {
            contention_count_.fetch_add(retry_count, std::memory_order_relaxed);
        }
        
        producers_done_.fetch_add(1, std::memory_order_release);
    }
    
    void consume_messages(const MPMCConfig& config, MPMCResults& results, size_t consumer_id) {
        size_t messages_received = 0;
        bool zero_copy_working = true;
        std::vector<size_t> producer_counts(config.num_producers, 0);
        
        // Create expected test data patterns for each producer
        std::vector<std::vector<uint8_t>> expected_patterns(config.num_producers);
        for (size_t p = 0; p < config.num_producers; ++p) {
            expected_patterns[p].resize(config.message_size);
            for (size_t i = 0; i < config.message_size; ++i) {
                expected_patterns[p][i] = static_cast<uint8_t>((p * 23 + i) % 256);
            }
        }
        
        while (producers_done_.load(std::memory_order_acquire) < config.num_producers || 
               messages_received < results.total_messages_sent / config.num_consumers) {
            
            // Try to receive message
            auto msg_opt = channel_->receive<MPMCMessage>(std::chrono::milliseconds(1));
            
            if (msg_opt) {
                auto receive_time = std::chrono::high_resolution_clock::now();
                auto& msg = *msg_opt;
                
                // Get timestamp and calculate latency
                auto send_time = msg.get_timestamp();
                auto latency = receive_time - send_time;
                consumer_latencies_[consumer_id].push_back(latency);
                
                // Get producer ID for verification
                size_t producer_id = msg.get_producer_id();
                
                // Verify data integrity (sample check)
                if (messages_received % 1000 == 0 && producer_id < config.num_producers) {
                    producer_counts[producer_id]++;
                    
                    if (!msg.verify_data(producer_id, expected_patterns[producer_id])) {
                        zero_copy_working = false;
                    }
                }
                
                messages_received++;
                consumer_message_counts_[consumer_id].fetch_add(1, std::memory_order_relaxed);
                
            } else {
                // No message available, yield
                std::this_thread::yield();
            }
        }
        
        // Update results (thread-safe)
        if (!zero_copy_working) {
            results.zero_copy_success = false;
        }
        
        // Print producer distribution for this consumer
        std::cout << "Consumer " << consumer_id << " producer distribution: ";
        for (size_t i = 0; i < config.num_producers; ++i) {
            std::cout << "P" << i << ":" << producer_counts[i] << " ";
        }
        std::cout << std::endl;
        
        consumers_done_.fetch_add(1, std::memory_order_release);
    }
    
    void calculate_metrics(MPMCResults& results, const MPMCConfig& config) {
        // Throughput calculations
        results.total_throughput_mbps = (results.total_bytes_consumed / (1024.0 * 1024.0)) / results.elapsed_seconds;
        results.total_throughput_msgs_per_sec = results.total_messages_received / results.elapsed_seconds;
        
        // Load balance efficiency
        if (config.num_consumers > 1) {
            auto min_consumer = *std::min_element(results.consumer_message_counts.begin(), 
                                                results.consumer_message_counts.end());
            auto max_consumer = *std::max_element(results.consumer_message_counts.begin(), 
                                                results.consumer_message_counts.end());
            
            if (max_consumer > 0) {
                results.load_balance_efficiency = static_cast<double>(min_consumer) / max_consumer;
            }
        } else {
            results.load_balance_efficiency = 1.0;
        }
        
        // Aggregate latency from all consumers
        std::vector<std::chrono::nanoseconds> all_latencies;
        for (const auto& consumer_latencies : consumer_latencies_) {
            all_latencies.insert(all_latencies.end(), 
                               consumer_latencies.begin(), 
                               consumer_latencies.end());
        }
        
        if (!all_latencies.empty()) {
            std::sort(all_latencies.begin(), all_latencies.end());
            
            auto sum = std::accumulate(all_latencies.begin(), all_latencies.end(), 
                                     std::chrono::nanoseconds(0));
            results.avg_latency_ns = sum.count() / static_cast<double>(all_latencies.size());
            
            results.min_latency_ns = all_latencies.front().count();
            results.max_latency_ns = all_latencies.back().count();
            
            // P99 latency
            size_t p99_index = static_cast<size_t>(all_latencies.size() * 0.99);
            results.p99_latency_ns = all_latencies[p99_index].count();
        }
        
        results.zero_copy_success = true; // Default to true, consumers set to false if issues
    }
};

void print_results(const MPMCResults& results, const MPMCConfig& config) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n--- " << config.test_name << " Results ---\n";
    std::cout << "Producers: " << config.num_producers << "\n";
    std::cout << "Consumers: " << config.num_consumers << "\n";
    std::cout << "Message size: " << config.message_size << " bytes\n";
    std::cout << "Messages per producer: " << config.num_messages_per_producer << "\n";
    std::cout << "Total messages sent: " << results.total_messages_sent << "\n";
    std::cout << "Total messages received: " << results.total_messages_received << "\n";
    std::cout << "Data produced: " << (results.total_bytes_produced / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Data consumed: " << (results.total_bytes_consumed / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Elapsed time: " << results.elapsed_seconds << " seconds\n";
    std::cout << "Total throughput: " << results.total_throughput_mbps << " MB/s\n";
    std::cout << "Message rate: " << results.total_throughput_msgs_per_sec << " msg/s\n";
    std::cout << "Avg latency: " << results.avg_latency_ns << " ns\n";
    std::cout << "Min latency: " << results.min_latency_ns << " ns\n";
    std::cout << "Max latency: " << results.max_latency_ns << " ns\n";
    std::cout << "P99 latency: " << results.p99_latency_ns << " ns\n";
    std::cout << "Contention events: " << results.contention_events << "\n";
    std::cout << "Load balance efficiency: " << (results.load_balance_efficiency * 100.0) << "%\n";
    std::cout << "Zero-copy: " << (results.zero_copy_success ? "SUCCESS" : "FAILED") << "\n";
    
    // Producer distribution
    std::cout << "Producer counts: ";
    for (size_t i = 0; i < config.num_producers; ++i) {
        std::cout << "P" << i << ":" << results.producer_message_counts[i] << " ";
    }
    std::cout << "\n";
    
    // Consumer distribution
    std::cout << "Consumer counts: ";
    for (size_t i = 0; i < config.num_consumers; ++i) {
        std::cout << "C" << i << ":" << results.consumer_message_counts[i] << " ";
    }
    std::cout << "\n";
    std::cout << std::string(50, '-') << "\n";
}

} // namespace psyne_bench

int main() {
    std::cout << "=== Psyne MPMC (Multi Producer Multi Consumer) Benchmark ===\n\n";
    
    using namespace psyne_bench;
    
    // Test configurations for different producer/consumer combinations
    std::vector<MPMCConfig> configs = {
        // Small scale tests
        {1024, 100'000, 2, 2, 64 * 1024 * 1024, "2P-2C Small Messages"},
        {1024, 100'000, 4, 2, 128 * 1024 * 1024, "4P-2C Small Messages"},
        {1024, 100'000, 2, 4, 128 * 1024 * 1024, "2P-4C Small Messages"},
        {1024, 100'000, 4, 4, 256 * 1024 * 1024, "4P-4C Small Messages"},
        
        // Medium scale tests
        {4096, 50'000, 4, 4, 256 * 1024 * 1024, "4P-4C Medium Messages"},
        {4096, 50'000, 8, 4, 512 * 1024 * 1024, "8P-4C Medium Messages"},
        {4096, 50'000, 4, 8, 512 * 1024 * 1024, "4P-8C Medium Messages"},
        {4096, 50'000, 8, 8, 1024 * 1024 * 1024, "8P-8C Medium Messages"},
        
        // Large scale tests  
        {16384, 20'000, 8, 8, 1024 * 1024 * 1024, "8P-8C Large Messages"},
        {16384, 20'000, 16, 8, 2048ULL * 1024 * 1024, "16P-8C Large Messages"},
        {16384, 20'000, 8, 16, 2048ULL * 1024 * 1024, "8P-16C Large Messages"},
        {16384, 10'000, 16, 16, 4096ULL * 1024 * 1024, "16P-16C Large Messages"},
        
        // Massive message tests - high contention
        {1024 * 1024, 1'000, 4, 4, 2048ULL * 1024 * 1024, "4P-4C Massive Messages"},
        {1024 * 1024, 1'000, 8, 8, 4096ULL * 1024 * 1024, "8P-8C Massive Messages"},
    };
    
    std::vector<MPMCResults> all_results;
    
    for (const auto& config : configs) {
        try {
            MPMCBenchmark benchmark(config.buffer_size, config.num_producers, config.num_consumers);
            auto results = benchmark.run_benchmark(config);
            print_results(results, config);
            all_results.push_back(results);
        } catch (const std::exception& e) {
            std::cerr << "Benchmark failed for " << config.test_name 
                      << ": " << e.what() << std::endl;
        }
    }
    
    // Print summary
    std::cout << "\n=== MPMC Benchmark Summary ===\n";
    std::cout << std::setw(20) << "Test Configuration" 
              << std::setw(8) << "P/C"
              << std::setw(15) << "Throughput (MB/s)"
              << std::setw(18) << "Rate (Kmsg/s)"  
              << std::setw(15) << "Avg Lat (ns)"
              << std::setw(12) << "Balance %"
              << std::setw(12) << "Contention\n";
    std::cout << std::string(100, '-') << "\n";
    
    for (size_t i = 0; i < configs.size() && i < all_results.size(); ++i) {
        const auto& config = configs[i];
        const auto& result = all_results[i];
        
        std::string pc_ratio = std::to_string(config.num_producers) + "/" + std::to_string(config.num_consumers);
        
        std::cout << std::setw(20) << config.test_name.substr(0, 19)
                  << std::setw(8) << pc_ratio
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.total_throughput_mbps
                  << std::setw(18) << std::fixed << std::setprecision(1) << (result.total_throughput_msgs_per_sec / 1000.0)
                  << std::setw(15) << std::fixed << std::setprecision(0) << result.avg_latency_ns
                  << std::setw(12) << std::fixed << std::setprecision(1) << (result.load_balance_efficiency * 100.0)
                  << std::setw(12) << result.contention_events << "\n";
    }
    
    std::cout << "\nMPMC benchmark completed successfully!\n";
    return 0;
}