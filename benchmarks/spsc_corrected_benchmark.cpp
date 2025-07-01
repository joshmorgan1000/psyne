/**
 * @file spsc_corrected_benchmark.cpp
 * @brief CORRECTED Performance benchmark for SPSC + InProcess substrate
 *
 * SPSC = Single Producer, Single Consumer (exactly 1 of each!)
 *
 * Measures:
 * - Single-threaded throughput
 * - Single producer -> Single consumer throughput
 * - Latency between producer and consumer threads
 */

#include "../include/psyne/core/behaviors.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

// Benchmark message
struct BenchmarkMessage {
    uint64_t id;
    uint64_t send_timestamp;
    uint64_t receive_timestamp;
    char payload[64];

    BenchmarkMessage() : id(0), send_timestamp(0), receive_timestamp(0) {
        std::memset(payload, 0, sizeof(payload));
    }

    BenchmarkMessage(uint64_t id) : id(id), receive_timestamp(0) {
        auto now = std::chrono::high_resolution_clock::now();
        send_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             now.time_since_epoch())
                             .count();
        std::snprintf(payload, sizeof(payload), "SPSC_Msg_%llu",
                      static_cast<unsigned long long>(id));
    }

    void mark_received() {
        auto now = std::chrono::high_resolution_clock::now();
        receive_timestamp =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch())
                .count();
    }

    uint64_t latency_ns() const {
        if (receive_timestamp >= send_timestamp) {
            return receive_timestamp - send_timestamp;
        }
        return 0; // Invalid
    }
};

// Simple InProcess substrate
class SPSCInProcessSubstrate : public psyne::behaviors::SubstrateBehavior {
public:
    void *allocate_memory_slab(size_t size_bytes) override {
        return std::aligned_alloc(64, size_bytes);
    }

    void deallocate_memory_slab(void *memory) override {
        std::free(memory);
    }

    void transport_send(void *data, size_t size) override {
        // In-process: no transport needed
    }

    void transport_receive(void *buffer, size_t buffer_size) override {
        // In-process: no transport needed
    }

    const char *substrate_name() const override {
        return "SPSCInProcess";
    }
    bool is_zero_copy() const override {
        return true;
    }
    bool is_cross_process() const override {
        return false;
    }
};

// Proper SPSC pattern
class TrueSPSCPattern : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        size_t current_slot =
            write_pos_.load(std::memory_order_relaxed) % max_messages_;

        // Store for receive
        slab_memory_ = slab_memory;
        message_size_ = message_size;

        // Increment after getting slot (single producer, so no race)
        write_pos_.store(write_pos_.load() + 1, std::memory_order_release);

        return static_cast<char *>(slab_memory) + (current_slot * message_size);
    }

    void *coordinate_receive() override {
        size_t current_read = read_pos_.load(std::memory_order_relaxed);
        size_t current_write = write_pos_.load(std::memory_order_acquire);

        if (current_read >= current_write) {
            return nullptr; // No messages available
        }

        size_t slot = current_read % max_messages_;

        // Increment after getting slot (single consumer, so no race)
        read_pos_.store(current_read + 1, std::memory_order_release);

        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }

    void producer_sync() override {}
    void consumer_sync() override {}

    const char *pattern_name() const override {
        return "TrueSPSC";
    }
    bool needs_locks() const override {
        return false;
    }
    size_t max_producers() const override {
        return 1;
    } // EXACTLY 1!
    size_t max_consumers() const override {
        return 1;
    } // EXACTLY 1!

    size_t messages_available() const {
        return write_pos_.load(std::memory_order_acquire) -
               read_pos_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    size_t max_messages_ = 1024 * 1024;
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;
};

struct SPSCBenchmarkResults {
    uint64_t total_messages;
    double duration_seconds;
    double throughput_msgs_per_sec;
    std::vector<uint64_t> latencies;
    double avg_latency_ns;
    double min_latency_ns;
    double max_latency_ns;
    double p95_latency_ns;
    double p99_latency_ns;
};

SPSCBenchmarkResults
run_single_threaded_spsc_benchmark(size_t num_messages = 1000000) {
    std::cout << "\n=== Single-Threaded SPSC Benchmark ===\n";
    std::cout << "Mode: One thread doing send->receive loop\n";
    std::cout << "Messages: " << num_messages << "\n";

    using ChannelType = psyne::behaviors::ChannelBridge<
        BenchmarkMessage, SPSCInProcessSubstrate, TrueSPSCPattern>;

    size_t slab_size = (num_messages + 1024) * sizeof(BenchmarkMessage);
    ChannelType channel(slab_size);

    std::vector<uint64_t> latencies;
    latencies.reserve(num_messages);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Single thread: send then immediately receive
    for (uint64_t i = 1; i <= num_messages; ++i) {
        // Send
        auto msg = channel.create_message(i);
        channel.send_message(msg);

        // Receive
        auto received_opt = channel.try_receive();
        if (received_opt) {
            auto &received = *received_opt;
            received->mark_received();
            uint64_t latency = received->latency_ns();
            if (latency > 0 &&
                latency < 1000000000ULL) { // Sanity check: < 1 second
                latencies.push_back(latency);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time);

    std::sort(latencies.begin(), latencies.end());

    SPSCBenchmarkResults results;
    results.total_messages = latencies.size();
    results.duration_seconds = duration.count() / 1e9;
    results.throughput_msgs_per_sec =
        results.total_messages / results.duration_seconds;
    results.latencies = std::move(latencies);
    results.avg_latency_ns =
        results.latencies.empty()
            ? 0
            : std::accumulate(results.latencies.begin(),
                              results.latencies.end(), 0.0) /
                  results.latencies.size();
    results.min_latency_ns =
        results.latencies.empty() ? 0 : results.latencies.front();
    results.max_latency_ns =
        results.latencies.empty() ? 0 : results.latencies.back();
    results.p95_latency_ns =
        results.latencies.empty()
            ? 0
            : results.latencies[results.latencies.size() * 95 / 100];
    results.p99_latency_ns =
        results.latencies.empty()
            ? 0
            : results.latencies[results.latencies.size() * 99 / 100];

    return results;
}

SPSCBenchmarkResults run_true_spsc_benchmark(size_t num_messages = 1000000) {
    std::cout << "\n=== True SPSC Benchmark ===\n";
    std::cout << "Mode: 1 Producer Thread -> 1 Consumer Thread\n";
    std::cout << "Messages: " << num_messages << "\n";

    using ChannelType = psyne::behaviors::ChannelBridge<
        BenchmarkMessage, SPSCInProcessSubstrate, TrueSPSCPattern>;

    size_t slab_size = (num_messages + 1024) * sizeof(BenchmarkMessage);
    ChannelType channel(slab_size);

    std::vector<uint64_t> latencies;
    std::atomic<bool> producer_done{false};
    std::atomic<uint64_t> messages_sent{0};
    std::atomic<uint64_t> messages_received{0};

    auto start_time = std::chrono::high_resolution_clock::now();

    // Single producer thread
    std::thread producer([&]() {
        for (uint64_t i = 1; i <= num_messages; ++i) {
            auto msg = channel.create_message(i);
            channel.send_message(msg);
            messages_sent.fetch_add(1, std::memory_order_relaxed);
        }
        producer_done.store(true, std::memory_order_release);
    });

    // Single consumer thread
    std::thread consumer([&]() {
        std::vector<uint64_t> local_latencies;
        local_latencies.reserve(num_messages);

        while (!producer_done.load(std::memory_order_acquire) ||
               messages_received.load() < num_messages) {
            auto msg_opt = channel.try_receive();
            if (msg_opt) {
                auto &msg = *msg_opt;
                msg->mark_received();

                uint64_t latency = msg->latency_ns();
                if (latency > 0 && latency < 1000000000ULL) { // Sanity check
                    local_latencies.push_back(latency);
                }

                messages_received.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Move local latencies to shared (thread-safe since consumer is done)
        latencies = std::move(local_latencies);
    });

    producer.join();
    consumer.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time);

    std::sort(latencies.begin(), latencies.end());

    SPSCBenchmarkResults results;
    results.total_messages = messages_received.load();
    results.duration_seconds = duration.count() / 1e9;
    results.throughput_msgs_per_sec =
        results.total_messages / results.duration_seconds;
    results.latencies = std::move(latencies);
    results.avg_latency_ns =
        results.latencies.empty()
            ? 0
            : std::accumulate(results.latencies.begin(),
                              results.latencies.end(), 0.0) /
                  results.latencies.size();
    results.min_latency_ns =
        results.latencies.empty() ? 0 : results.latencies.front();
    results.max_latency_ns =
        results.latencies.empty() ? 0 : results.latencies.back();
    results.p95_latency_ns =
        results.latencies.empty()
            ? 0
            : results.latencies[results.latencies.size() * 95 / 100];
    results.p99_latency_ns =
        results.latencies.empty()
            ? 0
            : results.latencies[results.latencies.size() * 99 / 100];

    return results;
}

void print_spsc_results(const std::string &test_name,
                        const SPSCBenchmarkResults &results) {
    std::cout << "\n" << test_name << " Results:\n";
    std::cout << "================================\n";
    std::cout << "Total Messages:     " << results.total_messages << "\n";
    std::cout << "Duration:           " << std::fixed << std::setprecision(3)
              << results.duration_seconds << " seconds\n";
    std::cout << "Throughput:         " << std::fixed << std::setprecision(0)
              << results.throughput_msgs_per_sec << " msgs/sec\n";
    std::cout << "Data Rate:          " << std::fixed << std::setprecision(2)
              << (results.total_messages * sizeof(BenchmarkMessage) /
                  results.duration_seconds) /
                     (1024 * 1024)
              << " MB/sec\n";

    if (!results.latencies.empty()) {
        std::cout << "\nLatency Statistics:\n";
        std::cout << "Average:            " << std::fixed
                  << std::setprecision(1) << results.avg_latency_ns << " ns\n";
        std::cout << "Minimum:            " << std::fixed
                  << std::setprecision(1) << results.min_latency_ns << " ns\n";
        std::cout << "Maximum:            " << std::fixed
                  << std::setprecision(1) << results.max_latency_ns << " ns\n";
        std::cout << "95th percentile:    " << std::fixed
                  << std::setprecision(1) << results.p95_latency_ns << " ns\n";
        std::cout << "99th percentile:    " << std::fixed
                  << std::setprecision(1) << results.p99_latency_ns << " ns\n";
    }
}

int main() {
    std::cout << "CORRECTED SPSC + InProcess Performance Benchmark\n";
    std::cout << "================================================\n";
    std::cout
        << "SPSC = Single Producer, Single Consumer (exactly 1 of each!)\n";
    std::cout << "Message size: " << sizeof(BenchmarkMessage) << " bytes\n";

    try {
        // Single-threaded benchmark (baseline)
        auto single_threaded = run_single_threaded_spsc_benchmark(1000000);
        print_spsc_results("Single-Threaded SPSC", single_threaded);

        // True SPSC benchmark (1 producer thread -> 1 consumer thread)
        auto true_spsc = run_true_spsc_benchmark(1000000);
        print_spsc_results("True SPSC (1P->1C)", true_spsc);

        std::cout << "\n=== SPSC Summary ===\n";
        std::cout << "Single-threaded:    " << std::fixed
                  << std::setprecision(0)
                  << single_threaded.throughput_msgs_per_sec << " msgs/sec\n";
        std::cout << "Cross-thread SPSC:  " << std::fixed
                  << std::setprecision(0) << true_spsc.throughput_msgs_per_sec
                  << " msgs/sec\n";
        std::cout << "Cross-thread ratio: " << std::fixed
                  << std::setprecision(2)
                  << (true_spsc.throughput_msgs_per_sec /
                      single_threaded.throughput_msgs_per_sec)
                  << "x\n";

        std::cout << "\nArchitecture:       Physical substrate + Abstract "
                     "message lens\n";
        std::cout << "Producers:          1 (exactly)\n";
        std::cout << "Consumers:          1 (exactly)\n";
        std::cout << "Zero-copy:          ✅ True\n";
        std::cout << "Lock-free:          ✅ True\n";

        std::cout << "\n🎯 TRUE SPSC benchmark complete!\n";
        std::cout << "Ready for MPSC (Multi-Producer, Single-Consumer) "
                     "implementation!\n";

    } catch (const std::exception &e) {
        std::cerr << "❌ Benchmark failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}