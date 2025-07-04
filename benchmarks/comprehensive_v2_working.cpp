/**
 * @file comprehensive_v2_working.cpp
 * @brief Comprehensive v2.0 benchmarks using the WORKING pattern approach
 *
 * Tests all combinations with 30-second runs for statistical significance:
 * - Patterns: SPSC, MPSC, SPMC, MPMC (using working inline implementations)
 * - Substrates: InProcess only (known working)
 * - Message sizes: Small (64B), Medium (1KB), Large (64KB)
 * - Duration: 30 seconds per test
 */

#include "../include/psyne/core/behaviors.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

// Test message types
struct SmallMessage {
    uint64_t id;
    uint64_t timestamp;
    uint32_t producer_id;
    uint32_t sequence;
    char padding[32]; // Total: 64 bytes
};

struct MediumMessage {
    uint64_t id;
    uint64_t timestamp;
    uint32_t producer_id;
    uint32_t sequence;
    char data[1000]; // Total: ~1KB
};

struct LargeMessage {
    uint64_t id;
    uint64_t timestamp;
    uint32_t producer_id;
    uint32_t sequence;
    char data[65500]; // Total: ~64KB
};

// Working substrate (same as tests)
class BenchmarkSubstrate : public psyne::behaviors::SubstrateBehavior {
public:
    void *allocate_memory_slab(size_t size_bytes) override {
        allocated_memory_ = std::aligned_alloc(64, size_bytes);
        if (!allocated_memory_)
            throw std::bad_alloc();
        slab_size_ = size_bytes;
        return allocated_memory_;
    }

    void deallocate_memory_slab(void *memory) override {
        if (memory && memory == allocated_memory_) {
            std::free(memory);
            allocated_memory_ = nullptr;
        }
    }

    void transport_send(void *data, size_t size) override {
        bytes_sent_ += size;
        packets_sent_++;
    }

    void transport_receive(void *buffer, size_t buffer_size) override {
        bytes_received_ += buffer_size;
        packets_received_++;
    }

    const char *substrate_name() const override {
        return "InProcess";
    }
    bool is_zero_copy() const override {
        return true;
    }
    bool is_cross_process() const override {
        return false;
    }

    // Statistics
    std::atomic<size_t> bytes_sent_{0};
    std::atomic<size_t> bytes_received_{0};
    std::atomic<size_t> packets_sent_{0};
    std::atomic<size_t> packets_received_{0};

private:
    void *allocated_memory_ = nullptr;
    size_t slab_size_ = 0;
};

// Working SPSC pattern (1 producer, 1 consumer)
class WorkingSPSC : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        if (!slab_memory_) {
            slab_memory_ = slab_memory;
            message_size_ = message_size;
        }

        size_t slot =
            write_pos_.fetch_add(1, std::memory_order_acq_rel) % max_messages_;
        return static_cast<char *>(slab_memory) + (slot * message_size);
    }

    void *coordinate_receive() override {
        size_t current_read = read_pos_.load(std::memory_order_relaxed);
        size_t current_write = write_pos_.load(std::memory_order_acquire);

        if (current_read >= current_write) {
            return nullptr;
        }

        size_t slot = current_read % max_messages_;
        read_pos_.fetch_add(1, std::memory_order_release);

        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }

    void producer_sync() override {}
    void consumer_sync() override {}

    const char *pattern_name() const override {
        return "SPSC";
    }
    bool needs_locks() const override {
        return false;
    }
    size_t max_producers() const override {
        return 1;
    }
    size_t max_consumers() const override {
        return 1;
    }

private:
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    size_t max_messages_ = 1024 * 1024; // 1M messages
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;
};

// Working MPSC pattern (multiple producers, 1 consumer)
class WorkingMPSC : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        if (!slab_memory_) {
            std::lock_guard<std::mutex> lock(slab_mutex_);
            if (!slab_memory_) {
                slab_memory_ = slab_memory;
                message_size_ = message_size;
            }
        }

        size_t slot =
            write_pos_.fetch_add(1, std::memory_order_acq_rel) % max_messages_;
        return static_cast<char *>(slab_memory) + (slot * message_size);
    }

    void *coordinate_receive() override {
        size_t current_read = read_pos_.load(std::memory_order_relaxed);
        size_t current_write = write_pos_.load(std::memory_order_acquire);

        if (current_read >= current_write) {
            return nullptr;
        }

        size_t slot = current_read % max_messages_;
        read_pos_.fetch_add(1, std::memory_order_release);

        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }

    void producer_sync() override {}
    void consumer_sync() override {}

    const char *pattern_name() const override {
        return "MPSC";
    }
    bool needs_locks() const override {
        return true;
    }
    size_t max_producers() const override {
        return SIZE_MAX;
    }
    size_t max_consumers() const override {
        return 1;
    }

private:
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    size_t max_messages_ = 1024 * 1024;
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;
    std::mutex slab_mutex_;
};

// Working SPMC pattern (1 producer, multiple consumers)
class WorkingSPMC : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        if (!slab_memory_) {
            slab_memory_ = slab_memory;
            message_size_ = message_size;
        }

        size_t slot =
            write_pos_.fetch_add(1, std::memory_order_acq_rel) % max_messages_;
        return static_cast<char *>(slab_memory) + (slot * message_size);
    }

    void *coordinate_receive() override {
        size_t current_read = read_pos_.fetch_add(1, std::memory_order_acq_rel);
        size_t current_write = write_pos_.load(std::memory_order_acquire);

        if (current_read >= current_write) {
            // Rollback optimistic read
            read_pos_.fetch_sub(1, std::memory_order_acq_rel);
            return nullptr;
        }

        size_t slot = current_read % max_messages_;
        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }

    void producer_sync() override {}
    void consumer_sync() override {}

    const char *pattern_name() const override {
        return "SPMC";
    }
    bool needs_locks() const override {
        return true;
    }
    size_t max_producers() const override {
        return 1;
    }
    size_t max_consumers() const override {
        return SIZE_MAX;
    }

private:
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    size_t max_messages_ = 1024 * 1024;
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;
};

// Working MPMC pattern (multiple producers, multiple consumers)
class WorkingMPMC : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        if (!slab_memory_) {
            std::lock_guard<std::mutex> lock(slab_mutex_);
            if (!slab_memory_) {
                slab_memory_ = slab_memory;
                message_size_ = message_size;
            }
        }

        // Check for full buffer
        size_t current_write = write_pos_.load(std::memory_order_relaxed);
        size_t current_read = read_pos_.load(std::memory_order_acquire);

        if (current_write - current_read >= max_messages_) {
            return nullptr; // Buffer full
        }

        size_t slot =
            write_pos_.fetch_add(1, std::memory_order_acq_rel) % max_messages_;
        return static_cast<char *>(slab_memory) + (slot * message_size);
    }

    void *coordinate_receive() override {
        size_t current_read = read_pos_.fetch_add(1, std::memory_order_acq_rel);
        size_t current_write = write_pos_.load(std::memory_order_acquire);

        if (current_read >= current_write) {
            // Rollback optimistic read
            read_pos_.fetch_sub(1, std::memory_order_acq_rel);
            return nullptr;
        }

        size_t slot = current_read % max_messages_;
        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }

    void producer_sync() override {}
    void consumer_sync() override {}

    const char *pattern_name() const override {
        return "MPMC";
    }
    bool needs_locks() const override {
        return true;
    }
    size_t max_producers() const override {
        return SIZE_MAX;
    }
    size_t max_consumers() const override {
        return SIZE_MAX;
    }

private:
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    size_t max_messages_ = 1024 * 1024;
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;
    std::mutex slab_mutex_;
};

// Benchmark configuration and results
struct BenchmarkConfig {
    std::string pattern_name;
    std::string message_size;
    size_t num_producers;
    size_t num_consumers;
    std::chrono::seconds duration{30};
    size_t message_size_bytes;
};

struct BenchmarkResults {
    std::string test_name;
    size_t total_messages_sent = 0;
    size_t total_messages_received = 0;
    double duration_seconds = 0.0;
    double throughput_msgs_per_sec = 0.0;
    double throughput_mbps = 0.0;
    double avg_latency_ns = 0.0;
    double p95_latency_ns = 0.0;
    double p99_latency_ns = 0.0;
    size_t bytes_transferred = 0;
    size_t message_size_bytes = 0;
    bool success = false;
};

class LatencyTracker {
public:
    void record(std::chrono::nanoseconds latency) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (samples_.size() < max_samples_) {
            samples_.push_back(latency.count());
        }
    }

    void calculate_stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (samples_.empty())
            return;

        std::sort(samples_.begin(), samples_.end());

        avg_latency_ = std::accumulate(samples_.begin(), samples_.end(), 0.0) /
                       samples_.size();

        if (samples_.size() >= 20) {
            p95_latency_ = samples_[samples_.size() * 95 / 100];
            p99_latency_ = samples_[samples_.size() * 99 / 100];
        }
    }

    double get_avg() const {
        return avg_latency_;
    }
    double get_p95() const {
        return p95_latency_;
    }
    double get_p99() const {
        return p99_latency_;
    }
    size_t get_sample_count() const {
        return samples_.size();
    }

private:
    std::vector<double> samples_;
    const size_t max_samples_ = 100000; // 100K samples
    double avg_latency_ = 0.0;
    double p95_latency_ = 0.0;
    double p99_latency_ = 0.0;
    std::mutex mutex_;
};

template <typename MessageType, typename PatternType>
BenchmarkResults run_benchmark(const BenchmarkConfig &config) {
    BenchmarkResults results;
    results.test_name =
        config.pattern_name + "_InProcess_" + config.message_size;
    results.message_size_bytes = config.message_size_bytes;

    std::cout << "\n=== " << results.test_name << " ===\n";
    std::cout << "Producers: " << config.num_producers
              << ", Consumers: " << config.num_consumers << "\n";
    std::cout << "Duration: " << config.duration.count()
              << "s, Message size: " << config.message_size_bytes << " bytes\n";

    try {
        using ChannelType =
            psyne::behaviors::ChannelBridge<MessageType, BenchmarkSubstrate,
                                            PatternType>;

        // Create channel with large slab for long-running test
        ChannelType channel(128 * 1024 * 1024); // 128MB slab

        std::atomic<bool> test_running{true};
        std::atomic<bool> warmup_complete{false};
        std::atomic<size_t> total_sent{0};
        std::atomic<size_t> total_received{0};
        std::atomic<size_t> producers_finished{0};

        LatencyTracker latency_tracker;

        auto start_time = std::chrono::steady_clock::now();
        auto warmup_duration = std::chrono::seconds(3); // 3s warmup

        // Producers
        std::vector<std::thread> producers;
        for (size_t p = 0; p < config.num_producers; ++p) {
            producers.emplace_back([&, producer_id = p + 1]() {
                size_t local_sent = 0;

                while (test_running.load()) {
                    try {
                        auto msg_start =
                            std::chrono::high_resolution_clock::now();

                        // Create message with timing info
                        auto msg = channel.create_message();
                        msg->id = local_sent + 1;
                        msg->timestamp = std::chrono::duration_cast<
                                             std::chrono::nanoseconds>(
                                             msg_start.time_since_epoch())
                                             .count();
                        msg->producer_id = producer_id;
                        msg->sequence = local_sent;

                        channel.send_message(msg);
                        local_sent++;

                        // Don't count warmup messages
                        if (warmup_complete.load()) {
                            total_sent.fetch_add(1);
                        }

                        // Brief pause for stability
                        if (local_sent % 10000 == 0) {
                            std::this_thread::sleep_for(
                                std::chrono::microseconds(1));
                        }

                    } catch (const std::exception &e) {
                        // Channel full or other error - brief pause
                        std::this_thread::sleep_for(
                            std::chrono::microseconds(10));
                    }
                }

                producers_finished.fetch_add(1);
            });
        }

        // Consumers
        std::vector<std::thread> consumers;
        for (size_t c = 0; c < config.num_consumers; ++c) {
            consumers.emplace_back([&, consumer_id = c + 1]() {
                size_t local_received = 0;

                while (test_running.load() ||
                       producers_finished.load() < config.num_producers) {
                    auto msg_opt = channel.try_receive();
                    if (msg_opt) {
                        auto receive_time =
                            std::chrono::high_resolution_clock::now();
                        auto &msg = *msg_opt;

                        // Calculate latency for non-warmup messages
                        if (warmup_complete.load() && msg->timestamp > 0) {
                            auto send_time =
                                std::chrono::nanoseconds(msg->timestamp);
                            auto latency =
                                receive_time.time_since_epoch() - send_time;
                            latency_tracker.record(latency);
                        }

                        local_received++;

                        // Don't count warmup messages
                        if (warmup_complete.load()) {
                            total_received.fetch_add(1);
                        }
                    } else {
                        // No message - brief pause
                        std::this_thread::sleep_for(
                            std::chrono::microseconds(1));
                    }
                }
            });
        }

        // Warmup phase
        std::this_thread::sleep_for(warmup_duration);
        warmup_complete.store(true);
        std::cout << "Warmup complete, starting measurement...\n";

        auto measurement_start = std::chrono::steady_clock::now();

        // Run for specified duration
        std::this_thread::sleep_for(config.duration);

        // Stop test
        test_running.store(false);
        auto measurement_end = std::chrono::steady_clock::now();

        // Wait for all threads
        for (auto &producer : producers) {
            producer.join();
        }
        for (auto &consumer : consumers) {
            consumer.join();
        }

        // Calculate results
        auto actual_duration =
            std::chrono::duration<double>(measurement_end - measurement_start);
        results.duration_seconds = actual_duration.count();
        results.total_messages_sent = total_sent.load();
        results.total_messages_received = total_received.load();
        results.bytes_transferred =
            results.total_messages_received * config.message_size_bytes;

        if (results.duration_seconds > 0) {
            results.throughput_msgs_per_sec =
                results.total_messages_received / results.duration_seconds;
            results.throughput_mbps =
                (results.bytes_transferred / (1024.0 * 1024.0)) /
                results.duration_seconds;
        }

        // Latency statistics
        latency_tracker.calculate_stats();
        results.avg_latency_ns = latency_tracker.get_avg();
        results.p95_latency_ns = latency_tracker.get_p95();
        results.p99_latency_ns = latency_tracker.get_p99();

        results.success =
            (results.total_messages_received > 0) &&
            (results.total_messages_received >=
             results.total_messages_sent * 0.90); // 90% receive rate

        // Print immediate results
        std::cout << "Messages sent:     " << results.total_messages_sent
                  << "\n";
        std::cout << "Messages received: " << results.total_messages_received
                  << "\n";
        std::cout << "Duration:          " << std::fixed << std::setprecision(2)
                  << results.duration_seconds << "s\n";
        std::cout << "Throughput:        " << std::fixed << std::setprecision(0)
                  << results.throughput_msgs_per_sec << " msgs/sec\n";
        std::cout << "Data rate:         " << std::fixed << std::setprecision(2)
                  << results.throughput_mbps << " MB/s\n";

        if (latency_tracker.get_sample_count() > 0) {
            std::cout << "Latency samples:   "
                      << latency_tracker.get_sample_count() << "\n";
            std::cout << "Avg latency:       " << std::fixed
                      << std::setprecision(1) << results.avg_latency_ns / 1000.0
                      << " μs\n";
            std::cout << "P95 latency:       " << std::fixed
                      << std::setprecision(1) << results.p95_latency_ns / 1000.0
                      << " μs\n";
            std::cout << "P99 latency:       " << std::fixed
                      << std::setprecision(1) << results.p99_latency_ns / 1000.0
                      << " μs\n";
        }

        std::cout << "Status:            "
                  << (results.success ? "✅ SUCCESS" : "❌ FAILED") << "\n";

    } catch (const std::exception &e) {
        std::cout << "❌ Benchmark failed: " << e.what() << "\n";
        results.success = false;
    }

    return results;
}

void save_results_to_csv(const std::vector<BenchmarkResults> &all_results) {
    std::ofstream csv("psyne_v2_comprehensive_results.csv");

    csv << "test_name,pattern,message_size,messages_sent,messages_received,"
        << "duration_s,throughput_msgs_per_sec,throughput_mbps,avg_latency_ns,"
        << "p95_latency_ns,p99_latency_ns,bytes_transferred,success\n";

    for (const auto &result : all_results) {
        csv << result.test_name << ","
            << result.test_name.substr(0, result.test_name.find('_')) << ","
            << result.test_name.substr(result.test_name.rfind('_') + 1) << ","
            << result.total_messages_sent << ","
            << result.total_messages_received << "," << std::fixed
            << std::setprecision(3) << result.duration_seconds << ","
            << std::fixed << std::setprecision(0)
            << result.throughput_msgs_per_sec << "," << std::fixed
            << std::setprecision(2) << result.throughput_mbps << ","
            << std::fixed << std::setprecision(1) << result.avg_latency_ns
            << "," << std::fixed << std::setprecision(1)
            << result.p95_latency_ns << "," << std::fixed
            << std::setprecision(1) << result.p99_latency_ns << ","
            << result.bytes_transferred << ","
            << (result.success ? "true" : "false") << "\n";
    }

    csv.close();
    std::cout << "\nResults saved to: psyne_v2_comprehensive_results.csv\n";
}

int main() {
    std::cout << "Psyne v2.0 Comprehensive Pattern × Message Size Benchmark\n";
    std::cout << "=========================================================\n";
    std::cout << "Testing ALL patterns with ALL message sizes\n";
    std::cout << "Duration: 30 seconds per test (+ 3s warmup)\n";
    std::cout << "Message sizes: 64B, 1KB, 64KB\n";
    std::cout << "Total estimated time: ~7 minutes\n\n";

    std::vector<BenchmarkResults> all_results;

    // SPSC Tests (1 producer, 1 consumer)
    std::cout << "\n🎯 SPSC (Single Producer, Single Consumer) Tests\n";
    std::cout << "================================================\n";

    all_results.push_back(run_benchmark<SmallMessage, WorkingSPSC>(
        {"SPSC", "Small", 1, 1, std::chrono::seconds(30),
         sizeof(SmallMessage)}));

    all_results.push_back(run_benchmark<MediumMessage, WorkingSPSC>(
        {"SPSC", "Medium", 1, 1, std::chrono::seconds(30),
         sizeof(MediumMessage)}));

    all_results.push_back(run_benchmark<LargeMessage, WorkingSPSC>(
        {"SPSC", "Large", 1, 1, std::chrono::seconds(30),
         sizeof(LargeMessage)}));

    // MPSC Tests (4 producers, 1 consumer)
    std::cout << "\n🔥 MPSC (Multi Producer, Single Consumer) Tests\n";
    std::cout << "===============================================\n";

    all_results.push_back(run_benchmark<SmallMessage, WorkingMPSC>(
        {"MPSC", "Small", 4, 1, std::chrono::seconds(30),
         sizeof(SmallMessage)}));

    all_results.push_back(run_benchmark<MediumMessage, WorkingMPSC>(
        {"MPSC", "Medium", 4, 1, std::chrono::seconds(30),
         sizeof(MediumMessage)}));

    all_results.push_back(run_benchmark<LargeMessage, WorkingMPSC>(
        {"MPSC", "Large", 4, 1, std::chrono::seconds(30),
         sizeof(LargeMessage)}));

    // SPMC Tests (1 producer, 4 consumers)
    std::cout << "\n📡 SPMC (Single Producer, Multi Consumer) Tests\n";
    std::cout << "===============================================\n";

    all_results.push_back(run_benchmark<SmallMessage, WorkingSPMC>(
        {"SPMC", "Small", 1, 4, std::chrono::seconds(30),
         sizeof(SmallMessage)}));

    all_results.push_back(run_benchmark<MediumMessage, WorkingSPMC>(
        {"SPMC", "Medium", 1, 4, std::chrono::seconds(30),
         sizeof(MediumMessage)}));

    all_results.push_back(run_benchmark<LargeMessage, WorkingSPMC>(
        {"SPMC", "Large", 1, 4, std::chrono::seconds(30),
         sizeof(LargeMessage)}));

    // MPMC Tests (4 producers, 4 consumers)
    std::cout << "\n⚡ MPMC (Multi Producer, Multi Consumer) Tests\n";
    std::cout << "=============================================\n";

    all_results.push_back(run_benchmark<SmallMessage, WorkingMPMC>(
        {"MPMC", "Small", 4, 4, std::chrono::seconds(30),
         sizeof(SmallMessage)}));

    all_results.push_back(run_benchmark<MediumMessage, WorkingMPMC>(
        {"MPMC", "Medium", 4, 4, std::chrono::seconds(30),
         sizeof(MediumMessage)}));

    all_results.push_back(run_benchmark<LargeMessage, WorkingMPMC>(
        {"MPMC", "Large", 4, 4, std::chrono::seconds(30),
         sizeof(LargeMessage)}));

    // Summary
    std::cout << "\n📊 COMPREHENSIVE BENCHMARK SUMMARY\n";
    std::cout << "==================================\n";

    size_t passed = 0;
    double max_throughput = 0.0;
    std::string best_config;

    for (const auto &result : all_results) {
        if (result.success) {
            passed++;
            if (result.throughput_msgs_per_sec > max_throughput) {
                max_throughput = result.throughput_msgs_per_sec;
                best_config = result.test_name;
            }
        }

        std::cout << std::left << std::setw(20) << result.test_name
                  << std::right << std::setw(15) << std::fixed
                  << std::setprecision(0) << result.throughput_msgs_per_sec
                  << " msgs/s" << std::setw(12) << std::fixed
                  << std::setprecision(1) << result.throughput_mbps << " MB/s";

        if (result.avg_latency_ns > 0) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(1)
                      << result.avg_latency_ns / 1000.0 << " μs";
        } else {
            std::cout << std::setw(12) << "-";
        }

        std::cout << "  " << (result.success ? "✅" : "❌") << "\n";
    }

    std::cout << "\nResults: " << passed << "/" << all_results.size()
              << " tests passed\n";
    std::cout << "Best performance: " << best_config << " (" << std::fixed
              << std::setprecision(0) << max_throughput << " msgs/s)\n";

    // Save to CSV
    save_results_to_csv(all_results);

    std::cout << "\n🚀 Comprehensive v2.0 benchmark complete!\n";
    std::cout << "📈 Pattern × Message Size performance matrix established.\n";
    std::cout << "📊 Data exported to CSV for analysis.\n";
    std::cout << "🖥️  Ready for Linux dedicated machine testing!\n";

    return 0;
}