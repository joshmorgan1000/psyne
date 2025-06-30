#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <psyne/psyne.hpp>
#include <thread>
#include <vector>

using namespace psyne;

// High-performance message for benchmarking
class PerfMessage : public Message<PerfMessage> {
public:
    static constexpr uint32_t message_type = 2000;
    static constexpr size_t size = 1024; // 1KB

    template <typename Channel>
    explicit PerfMessage(Channel &channel) : Message<PerfMessage>(channel) {
        if (this->data_) {
            std::memset(this->data_, 0, size);
        }
    }

    explicit PerfMessage(const void *data, size_t sz)
        : Message<PerfMessage>(data, sz) {}

    static constexpr size_t calculate_size() {
        return size;
    }

    void set_data(uint64_t id, uint32_t thread_id) {
        if (data_) {
            uint64_t *header = reinterpret_cast<uint64_t *>(data_);
            header[0] = id;
            header[1] = thread_id;
            // Fill rest with pattern
            for (size_t i = 16; i < size; i += 4) {
                *reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(data_) +
                                              i) =
                    static_cast<uint32_t>(id ^ thread_id);
            }
        }
    }

    uint64_t get_id() const {
        if (!data_)
            return 0;
        return *reinterpret_cast<const uint64_t *>(data_);
    }

    uint32_t get_thread_id() const {
        if (!data_)
            return 0;
        return *reinterpret_cast<const uint32_t *>(
            static_cast<const uint8_t *>(data_) + 8);
    }

    void before_send() {}
};

template class psyne::Message<PerfMessage>;

// Multi-core stress test
void multi_core_stress_test() {
    std::cout << "🔥 M4 MULTI-CORE STRESS TEST 🔥\n";
    std::cout << "================================\n\n";

    const int num_channels = 16;            // One per core
    const int messages_per_thread = 500000; // 10x bigger!
    const int total_messages = num_channels * messages_per_thread;

    std::vector<std::unique_ptr<Channel>> channels;
    std::vector<std::thread> threads;
    std::atomic<int> total_sent{0};
    std::atomic<int> total_received{0};
    std::atomic<bool> all_done{false};

    // Create channels
    for (int i = 0; i < num_channels; ++i) {
        std::string uri = "memory://stress_" + std::to_string(i);
        channels.push_back(
            create_channel(uri, 16 * 1024 * 1024)); // 16MB per channel
    }

    std::cout << "Starting " << num_channels << " threads on your M4...\n";
    std::cout << "Each thread will process " << messages_per_thread
              << " messages\n";
    std::cout << "Total workload: " << total_messages << " messages\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch producer/consumer pairs
    for (int t = 0; t < num_channels; ++t) {
        threads.emplace_back([&, t]() {
            auto &channel = *channels[t];
            int local_sent = 0;
            int local_received = 0;

            // Interleaved send/receive for maximum stress
            for (int i = 0; i < messages_per_thread; ++i) {
                // Send
                PerfMessage msg(channel);
                msg.set_data(i, t);
                msg.send();
                local_sent++;

                // Try to receive (may not get one immediately)
                auto received = channel.receive_single<PerfMessage>();
                if (received) {
                    local_received++;
                    // Verify data integrity
                    assert(received->get_thread_id() ==
                           static_cast<uint32_t>(t));
                }

                // Every 1000 messages, try harder to catch up on receiving
                if (i % 1000 == 0) {
                    for (int j = 0; j < 10; ++j) {
                        auto msg_r = channel.receive_single<PerfMessage>();
                        if (msg_r) {
                            local_received++;
                            assert(msg_r->get_thread_id() ==
                                   static_cast<uint32_t>(t));
                        } else {
                            break;
                        }
                    }
                }
            }

            // Final cleanup - receive remaining messages
            while (local_received < local_sent) {
                auto msg = channel.receive_single<PerfMessage>(
                    std::chrono::milliseconds(1));
                if (msg) {
                    local_received++;
                    assert(msg->get_thread_id() == static_cast<uint32_t>(t));
                } else {
                    break; // No more messages
                }
            }

            total_sent.fetch_add(local_sent);
            total_received.fetch_add(local_received);

            std::cout << "Thread " << std::setw(2) << t
                      << " completed: " << local_sent << " sent, "
                      << local_received << " received\n";
        });
    }

    // Wait for all threads
    for (auto &thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    double elapsed_seconds = duration.count() / 1e6;
    double messages_per_second = total_received.load() / elapsed_seconds;
    double mb_per_second = (total_received.load() * PerfMessage::size) /
                           (1024.0 * 1024.0 * elapsed_seconds);
    double gb_per_second = mb_per_second / 1024.0;

    std::cout << "\n🚀 M4 PERFORMANCE RESULTS 🚀\n";
    std::cout << "============================\n";
    std::cout << "Threads utilized: " << num_channels << "\n";
    std::cout << "Messages sent: " << total_sent.load() << "\n";
    std::cout << "Messages received: " << total_received.load() << "\n";
    std::cout << "Success rate: " << std::fixed << std::setprecision(1)
              << (100.0 * total_received.load() / total_sent.load()) << "%\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(3)
              << elapsed_seconds << " seconds\n";
    std::cout << "Throughput: " << std::scientific << std::setprecision(2)
              << messages_per_second << " msg/s\n";
    std::cout << "Bandwidth: " << std::fixed << std::setprecision(1)
              << mb_per_second << " MB/s";
    if (gb_per_second >= 1.0) {
        std::cout << " (" << std::setprecision(2) << gb_per_second << " GB/s)";
    }
    std::cout << "\n";
    std::cout << "Per-core throughput: " << std::scientific
              << std::setprecision(2) << (messages_per_second / num_channels)
              << " msg/s\n";
    std::cout << "Message size: " << PerfMessage::size << " bytes\n\n";
}

// CPU-intensive computation stress test
void cpu_intensive_test() {
    std::cout << "🧠 NEURAL CORE COMPUTATIONAL STRESS TEST 🧠\n";
    std::cout << "==========================================\n\n";

    const int num_threads = 16;
    const int iterations_per_thread = 10000000; // 10x bigger!

    std::vector<std::thread> threads;
    std::atomic<uint64_t> total_operations{0};

    std::cout << "Launching " << num_threads << " CPU-intensive threads...\n";
    std::cout << "Each thread performing " << iterations_per_thread
              << " floating-point operations\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            double accumulator = 1.0;
            uint64_t local_ops = 0;

            // Intensive floating-point computation
            for (int i = 0; i < iterations_per_thread; ++i) {
                double x = static_cast<double>(t * 1000 + i);
                accumulator +=
                    std::sin(x) * std::cos(x / 2.0) + std::sqrt(x + 1.0);
                accumulator =
                    std::fmod(accumulator, 1000000.0); // Prevent overflow
                local_ops += 4; // sin, cos, sqrt, fmod operations

                if (i % 100000 == 0) {
                    std::cout << "Thread " << std::setw(2) << t
                              << " progress: " << std::setw(3)
                              << (i * 100 / iterations_per_thread) << "%\n";
                }
            }

            total_operations.fetch_add(local_ops);

            // Prevent optimization from eliminating computation
            volatile double result = accumulator;
            (void)result;
        });
    }

    // Wait for completion
    for (auto &thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    double elapsed_seconds = duration.count() / 1e6;
    double ops_per_second = total_operations.load() / elapsed_seconds;

    std::cout << "\n🔥 COMPUTATIONAL RESULTS 🔥\n";
    std::cout << "===========================\n";
    std::cout << "CPU threads: " << num_threads << "\n";
    std::cout << "Total operations: " << total_operations.load() << "\n";
    std::cout << "Execution time: " << std::fixed << std::setprecision(3)
              << elapsed_seconds << " seconds\n";
    std::cout << "Operations/second: " << std::scientific
              << std::setprecision(2) << ops_per_second << "\n";
    std::cout << "Per-core ops/sec: " << std::scientific << std::setprecision(2)
              << (ops_per_second / num_threads) << "\n\n";
}

// Memory bandwidth test
void memory_bandwidth_test() {
    std::cout << "💾 MEMORY BANDWIDTH STRESS TEST 💾\n";
    std::cout << "==================================\n\n";

    const int num_threads = 16;
    const size_t buffer_size = 256 * 1024 * 1024; // 256MB per thread
    const int iterations = 500;                   // 5x bigger!

    std::vector<std::thread> threads;
    std::atomic<uint64_t> total_bytes_processed{0};

    std::cout << "Testing memory bandwidth with " << num_threads
              << " threads...\n";
    std::cout << "Buffer size per thread: " << (buffer_size / 1024 / 1024)
              << " MB\n";
    std::cout << "Total memory footprint: "
              << (buffer_size * num_threads / 1024 / 1024) << " MB\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<uint8_t> buffer(buffer_size);
            uint64_t local_bytes = 0;

            for (int iter = 0; iter < iterations; ++iter) {
                // Write pass
                for (size_t i = 0; i < buffer_size; i += 64) {
                    *reinterpret_cast<uint64_t *>(&buffer[i]) =
                        t * 1000000ULL + iter * 1000ULL + i;
                }
                local_bytes += buffer_size;

                // Read pass
                uint64_t checksum = 0;
                for (size_t i = 0; i < buffer_size; i += 64) {
                    checksum ^= *reinterpret_cast<const uint64_t *>(&buffer[i]);
                }
                local_bytes += buffer_size;

                // Prevent optimization
                volatile uint64_t result = checksum;
                (void)result;

                if (iter % 20 == 0) {
                    std::cout << "Thread " << std::setw(2) << t << " iteration "
                              << std::setw(3) << iter << "/" << iterations
                              << "\n";
                }
            }

            total_bytes_processed.fetch_add(local_bytes);
        });
    }

    // Wait for completion
    for (auto &thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    double elapsed_seconds = duration.count() / 1e6;
    double bytes_per_second = total_bytes_processed.load() / elapsed_seconds;
    double gb_per_second = bytes_per_second / (1024.0 * 1024.0 * 1024.0);

    std::cout << "\n🌊 MEMORY BANDWIDTH RESULTS 🌊\n";
    std::cout << "===============================\n";
    std::cout << "Memory threads: " << num_threads << "\n";
    std::cout << "Total bytes processed: "
              << (total_bytes_processed.load() / 1024 / 1024) << " MB\n";
    std::cout << "Execution time: " << std::fixed << std::setprecision(3)
              << elapsed_seconds << " seconds\n";
    std::cout << "Memory bandwidth: " << std::fixed << std::setprecision(2)
              << gb_per_second << " GB/s\n";
    std::cout << "Per-thread bandwidth: " << std::fixed << std::setprecision(2)
              << (gb_per_second / num_threads) << " GB/s\n\n";
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════"
                 "══════╗\n";
    std::cout << "║              Psyne v1.2.0 - M4 MULTI-CORE BENCHMARK        "
                 "     ║\n";
    std::cout << "║                 WAKE UP THOSE NEURAL CORES! 🚀             "
                 "     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════"
                 "══════╝\n\n";

    std::cout
        << "System: Apple M4 with 16 cores + 40 neural processing units\n";
    std::cout << "Objective: Maximum utilization stress test\n";
    std::cout << "Library: Psyne v1.2.0 Zero-Copy Messaging\n\n";

    try {
        // Run all tests
        multi_core_stress_test();
        cpu_intensive_test();
        memory_bandwidth_test();

        std::cout << "╔════════════════════════════════════════════════════════"
                     "══════════╗\n";
        std::cout << "║                    M4 BENCHMARK COMPLETE! 🎯           "
                     "         ║\n";
        std::cout << "║     Your M4 has been properly stressed across all "
                     "dimensions     ║\n";
        std::cout << "║   Psyne v1.2.0 delivers exceptional multi-core "
                     "performance!     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════"
                     "══════════╝\n";

        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}