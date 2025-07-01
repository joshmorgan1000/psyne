/**
 * @file messaging_patterns_benchmark.cpp
 * @brief Comprehensive benchmarks for various messaging patterns
 *
 * Tests different messaging patterns to saturate hardware:
 * 1. Throughput test - maximum messages per second
 * 2. Latency test - round-trip time measurements
 * 3. Bandwidth test - large message transfers
 * 4. Burst test - sudden traffic spikes
 * 5. Concurrent test - multiple producers/consumers
 */

#include "../include/psyne/core/behaviors.hpp"
#include "../include/psyne/global/logger.hpp"
#include "../include/psyne/global/threadpool.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <random>
#include <thread>
#include <vector>

using namespace psyne;
using namespace std::chrono;

// Global thread pool for benchmark execution
static std::unique_ptr<PsynePool> benchmark_pool;

// Initialize benchmark pool
void init_benchmark_pool(const BenchmarkConfig &config) {
    if (!benchmark_pool) {
        // Configure thread pool based on benchmark requirements
        PsynePool::ThreadAffinity affinity;
        affinity.use_affinity = config.cpu_affinity;

        size_t num_threads = config.num_producers + config.num_consumers;
        if (num_threads == 0)
            num_threads = std::thread::hardware_concurrency();

        benchmark_pool = std::make_unique<PsynePool>(
            std::min(num_threads,
                     std::thread::hardware_concurrency()), // min threads
            std::max(num_threads,
                     std::thread::hardware_concurrency()), // max threads
            5000, // thread evict wait (5 seconds)
            true, // enable work stealing
            affinity);

        log_info("Initialized benchmark thread pool with ", num_threads,
                 " threads");
    }
}

// Test configuration
struct BenchmarkConfig {
    std::string transport;   // "ipc", "tcp", "spsc"
    size_t message_size;     // Size of each message
    size_t num_messages;     // Total messages to send
    size_t num_producers;    // Number of producer threads
    size_t num_consumers;    // Number of consumer threads
    size_t burst_size;       // Messages per burst
    bool zero_copy;          // Use zero-copy optimization
    bool cpu_affinity;       // Pin threads to CPUs
    std::string server_host; // For TCP mode
    uint16_t server_port;    // For TCP mode
};

// Statistics tracking
struct BenchmarkStats {
    std::atomic<uint64_t> messages_sent{0};
    std::atomic<uint64_t> messages_received{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> total_latency_ns{0};
    std::atomic<uint64_t> min_latency_ns{UINT64_MAX};
    std::atomic<uint64_t> max_latency_ns{0};
    high_resolution_clock::time_point start_time;
    high_resolution_clock::time_point end_time;
};

// Message structure for benchmarking
struct BenchmarkMessage {
    uint64_t sequence;
    uint64_t timestamp_ns;
    uint32_t producer_id;
    uint32_t checksum;
    uint8_t payload[]; // Variable size payload
};

// Checksum calculation for data integrity
uint32_t calculate_checksum(const uint8_t *data, size_t size) {
    uint32_t checksum = 0;
    for (size_t i = 0; i < size; i += 4) {
        checksum ^= *reinterpret_cast<const uint32_t *>(data + i);
    }
    return checksum;
}

// Set CPU affinity for thread
void set_cpu_affinity(int cpu_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#elif __APPLE__
    // macOS doesn't have pthread_setaffinity_np, use thread QoS instead
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

// Create channel based on configuration
template <typename T>
std::shared_ptr<Channel<T>>
create_benchmark_channel(const BenchmarkConfig &config, bool is_producer) {
    ChannelConfig chan_config;
    chan_config.size_mb = 64; // 64MB for benchmarking
    chan_config.blocking = true;

    if (config.transport == "ipc") {
        chan_config.name = "benchmark_ipc_channel";
        chan_config.mode = ChannelMode::SPSC;
        chan_config.transport = ChannelTransport::IPC;
        chan_config.is_producer = is_producer;
    } else if (config.transport == "tcp") {
        chan_config.name = "benchmark_tcp_channel";
        chan_config.mode = ChannelMode::SPSC;
        chan_config.transport = ChannelTransport::TCP;
        chan_config.is_server = is_producer;
        chan_config.remote_host = config.server_host;
        chan_config.remote_port = config.server_port;
    } else { // spsc
        chan_config.name = "benchmark_spsc_channel";
        chan_config.mode = ChannelMode::SPSC;
        chan_config.transport = ChannelTransport::IN_PROCESS;
    }

    return Channel<T>::create(chan_config);
}

// Throughput test - maximum messages per second
void throughput_producer(std::shared_ptr<Channel<BenchmarkMessage>> channel,
                         const BenchmarkConfig &config, BenchmarkStats &stats,
                         uint32_t producer_id) {
    if (config.cpu_affinity) {
        set_cpu_affinity(producer_id % std::thread::hardware_concurrency());
    }

    size_t total_size = sizeof(BenchmarkMessage) + config.message_size;
    uint64_t messages_per_producer = config.num_messages / config.num_producers;

    // Wait for start signal
    std::this_thread::sleep_for(milliseconds(100));

    auto start = high_resolution_clock::now();

    for (uint64_t i = 0; i < messages_per_producer; ++i) {
        auto msg = channel->allocate(total_size);

        msg->sequence = i;
        msg->timestamp_ns = duration_cast<nanoseconds>(
                                high_resolution_clock::now().time_since_epoch())
                                .count();
        msg->producer_id = producer_id;

        // Fill payload with pattern
        for (size_t j = 0; j < config.message_size; ++j) {
            msg->payload[j] = (uint8_t)((i + j) & 0xFF);
        }

        msg->checksum = calculate_checksum(msg->payload, config.message_size);

        msg.send();

        stats.messages_sent++;
        stats.bytes_sent += total_size;
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    log_info("Producer ", producer_id, " sent ", messages_per_producer,
             " messages in ", duration, " us (",
             (messages_per_producer * 1000000.0 / duration), " msg/s)");
}

void throughput_consumer(std::shared_ptr<Channel<BenchmarkMessage>> channel,
                         const BenchmarkConfig &config, BenchmarkStats &stats,
                         uint32_t consumer_id) {
    if (config.cpu_affinity) {
        set_cpu_affinity((config.num_producers + consumer_id) %
                         std::thread::hardware_concurrency());
    }

    uint64_t messages_per_consumer = config.num_messages / config.num_consumers;
    uint64_t received = 0;

    while (received < messages_per_consumer) {
        auto msg = channel->receive();

        // Verify checksum
        uint32_t expected_checksum =
            calculate_checksum(msg->payload, config.message_size);
        if (msg->checksum != expected_checksum) {
            log_error("Checksum mismatch! Expected: ", expected_checksum,
                      ", Got: ", msg->checksum);
        }

        // Calculate latency
        uint64_t now = duration_cast<nanoseconds>(
                           high_resolution_clock::now().time_since_epoch())
                           .count();
        uint64_t latency = now - msg->timestamp_ns;

        stats.total_latency_ns += latency;

        // Update min/max latency
        uint64_t current_min = stats.min_latency_ns.load();
        while (
            latency < current_min &&
            !stats.min_latency_ns.compare_exchange_weak(current_min, latency))
            ;

        uint64_t current_max = stats.max_latency_ns.load();
        while (
            latency > current_max &&
            !stats.max_latency_ns.compare_exchange_weak(current_max, latency))
            ;

        stats.messages_received++;
        stats.bytes_received += sizeof(BenchmarkMessage) + config.message_size;
        received++;
    }
}

// Bandwidth test - large message transfers
void bandwidth_test(const BenchmarkConfig &config) {
    log_info("=== Bandwidth Test ===");
    log_info("Message size: ", config.message_size / 1024, " KB");
    log_info("Total messages: ", config.num_messages);

    BenchmarkStats stats;
    auto channel = create_benchmark_channel<BenchmarkMessage>(config, true);

    // For TCP, wait for connection
    if (config.transport == "tcp") {
        if (auto tcp_chan = std::dynamic_pointer_cast<TCPChannel>(channel)) {
            tcp_chan->wait_for_connection();
        }
    }

    stats.start_time = high_resolution_clock::now();

    // Use PsynePool for bandwidth test threads
    init_benchmark_pool(config);

    // Producer task
    auto producer_future =
        benchmark_pool->enqueuePriority(PsynePool::HIGH_PRIORITY, [&]() {
            size_t total_size = sizeof(BenchmarkMessage) + config.message_size;

            for (uint64_t i = 0; i < config.num_messages; ++i) {
                auto msg = channel->allocate(total_size);

                msg->sequence = i;
                msg->timestamp_ns =
                    duration_cast<nanoseconds>(
                        high_resolution_clock::now().time_since_epoch())
                        .count();
                msg->producer_id = 0;

                // Fill with random data for bandwidth test
                std::mt19937 rng(i);
                for (size_t j = 0; j < config.message_size; j += 4) {
                    *reinterpret_cast<uint32_t *>(&msg->payload[j]) = rng();
                }

                msg->checksum =
                    calculate_checksum(msg->payload, config.message_size);

                msg.send();
                stats.messages_sent++;
                stats.bytes_sent += total_size;
            }
        });

    // Consumer task
    auto consumer_future =
        benchmark_pool->enqueuePriority(PsynePool::HIGH_PRIORITY, [&]() {
            for (uint64_t i = 0; i < config.num_messages; ++i) {
                auto msg = channel->receive();

                // Verify data integrity
                uint32_t expected_checksum =
                    calculate_checksum(msg->payload, config.message_size);
                if (msg->checksum != expected_checksum) {
                    log_error("Data corruption detected!");
                }

                stats.messages_received++;
                stats.bytes_received +=
                    sizeof(BenchmarkMessage) + config.message_size;
            }
        });

    producer_future.wait();
    consumer_future.wait();

    stats.end_time = high_resolution_clock::now();

    // Print results
    auto duration =
        duration_cast<milliseconds>(stats.end_time - stats.start_time).count();
    double throughput_mbps =
        (stats.bytes_sent.load() / 1024.0 / 1024.0) / (duration / 1000.0);

    log_info("Duration: ", duration, " ms");
    log_info("Total data transferred: ", stats.bytes_sent.load() / 1024 / 1024,
             " MB");
    log_info("Throughput: ", std::fixed, std::setprecision(2), throughput_mbps,
             " MB/s");
}

// Burst test - sudden traffic spikes
void burst_test(const BenchmarkConfig &config) {
    log_info("=== Burst Test ===");
    log_info("Burst size: ", config.burst_size, " messages");
    log_info("Message size: ", config.message_size, " bytes");

    BenchmarkStats stats;
    auto channel = create_benchmark_channel<BenchmarkMessage>(config, true);

    size_t total_size = sizeof(BenchmarkMessage) + config.message_size;
    size_t num_bursts = config.num_messages / config.burst_size;

    // Use PsynePool for burst test threads
    init_benchmark_pool(config);

    // Producer - send in bursts using thread pool
    auto producer_future =
        benchmark_pool->enqueuePriority(PsynePool::HIGH_PRIORITY, [&]() {
            for (size_t burst = 0; burst < num_bursts; ++burst) {
                auto burst_start = high_resolution_clock::now();

                // Send burst
                for (size_t i = 0; i < config.burst_size; ++i) {
                    auto msg = channel->allocate(total_size);

                    msg->sequence = burst * config.burst_size + i;
                    msg->timestamp_ns =
                        duration_cast<nanoseconds>(
                            high_resolution_clock::now().time_since_epoch())
                            .count();
                    msg->producer_id = 0;

                    // Simple payload
                    std::memset(msg->payload, (uint8_t)i, config.message_size);
                    msg->checksum = i; // Simple checksum for burst test

                    msg.send();
                }

                auto burst_end = high_resolution_clock::now();
                auto burst_duration =
                    duration_cast<microseconds>(burst_end - burst_start)
                        .count();

                log_info("Burst ", burst, ": ", config.burst_size,
                         " messages in ", burst_duration, " us");

                // Wait between bursts
                std::this_thread::sleep_for(milliseconds(100));
            }
        });

    // Consumer using thread pool
    auto consumer_future =
        benchmark_pool->enqueuePriority(PsynePool::HIGH_PRIORITY, [&]() {
            for (size_t i = 0; i < config.num_messages; ++i) {
                auto msg = channel->receive();

                uint64_t now =
                    duration_cast<nanoseconds>(
                        high_resolution_clock::now().time_since_epoch())
                        .count();
                uint64_t latency = now - msg->timestamp_ns;

                if (i % config.burst_size == 0) {
                    log_info("Burst ", (i / config.burst_size),
                             " first message latency: ", latency / 1000, " us");
                }
            }
        });

    producer_future.wait();
    consumer_future.wait();
}

// Multi-threaded stress test
void stress_test(const BenchmarkConfig &config) {
    log_info("=== Multi-threaded Stress Test ===");
    log_info("Producers: ", config.num_producers);
    log_info("Consumers: ", config.num_consumers);
    log_info("Messages per producer: ",
             config.num_messages / config.num_producers);

    BenchmarkStats stats;
    stats.start_time = high_resolution_clock::now();

    // Use PsynePool for efficient thread management
    init_benchmark_pool(config);

    std::vector<std::future<void>> producer_futures;
    std::vector<std::future<void>> consumer_futures;

    // Launch producers using thread pool
    for (size_t i = 0; i < config.num_producers; ++i) {
        auto future = benchmark_pool->enqueuePriority(
            PsynePool::HIGH_PRIORITY, [&config, &stats, i]() {
                auto channel =
                    create_benchmark_channel<BenchmarkMessage>(config, true);
                throughput_producer(channel, config, stats, i);
            });
        producer_futures.push_back(std::move(future));
    }

    // Launch consumers using thread pool
    for (size_t i = 0; i < config.num_consumers; ++i) {
        auto future = benchmark_pool->enqueuePriority(
            PsynePool::HIGH_PRIORITY, [&config, &stats, i]() {
                auto channel =
                    create_benchmark_channel<BenchmarkMessage>(config, false);
                throughput_consumer(channel, config, stats, i);
            });
        consumer_futures.push_back(std::move(future));
    }

    // Wait for completion
    for (auto &future : producer_futures) {
        future.wait();
    }
    for (auto &future : consumer_futures) {
        future.wait();
    }

    stats.end_time = high_resolution_clock::now();

    // Print aggregate results
    auto duration =
        duration_cast<seconds>(stats.end_time - stats.start_time).count();
    double msg_per_sec = stats.messages_sent.load() / (double)duration;
    double mb_per_sec = (stats.bytes_sent.load() / 1024.0 / 1024.0) / duration;

    log_info("Aggregate Results:");
    log_info("Total duration: ", duration, " seconds");
    log_info("Messages sent: ", stats.messages_sent.load());
    log_info("Messages received: ", stats.messages_received.load());
    log_info("Throughput: ", std::fixed, std::setprecision(2), msg_per_sec,
             " msg/s");
    log_info("Bandwidth: ", mb_per_sec, " MB/s");
    log_info("Average latency: ",
             stats.total_latency_ns.load() / stats.messages_received.load() /
                 1000,
             " us");
    log_info("Min latency: ", stats.min_latency_ns.load() / 1000, " us");
    log_info("Max latency: ", stats.max_latency_ns.load() / 1000, " us");
}

void print_usage(const char *program) {
    log_info("Usage: ", program, " [options]");
    log_info("Options:");
    log_info("  --transport <ipc|tcp|spsc>  Transport type (default: ipc)");
    log_info("  --pattern <throughput|bandwidth|burst|stress>  Test pattern");
    log_info("  --size <bytes>               Message size (default: 64)");
    log_info(
        "  --count <num>                Number of messages (default: 1000000)");
    log_info("  --producers <num>            Number of producers (default: 1)");
    log_info("  --consumers <num>            Number of consumers (default: 1)");
    log_info("  --burst <size>               Burst size (default: 1000)");
    log_info("  --affinity                   Enable CPU affinity");
    log_info(
        "  --host <hostname>            TCP server host (default: localhost)");
    log_info("  --port <port>                TCP server port (default: 9999)");
}

int main(int argc, char *argv[]) {
    BenchmarkConfig config;
    config.transport = "ipc";
    config.message_size = 64;
    config.num_messages = 1000000;
    config.num_producers = 1;
    config.num_consumers = 1;
    config.burst_size = 1000;
    config.zero_copy = true;
    config.cpu_affinity = false;
    config.server_host = "localhost";
    config.server_port = 9999;

    std::string test_pattern = "throughput";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--transport" && i + 1 < argc) {
            config.transport = argv[++i];
        } else if (arg == "--pattern" && i + 1 < argc) {
            test_pattern = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            config.message_size = std::stoul(argv[++i]);
        } else if (arg == "--count" && i + 1 < argc) {
            config.num_messages = std::stoul(argv[++i]);
        } else if (arg == "--producers" && i + 1 < argc) {
            config.num_producers = std::stoul(argv[++i]);
        } else if (arg == "--consumers" && i + 1 < argc) {
            config.num_consumers = std::stoul(argv[++i]);
        } else if (arg == "--burst" && i + 1 < argc) {
            config.burst_size = std::stoul(argv[++i]);
        } else if (arg == "--affinity") {
            config.cpu_affinity = true;
        } else if (arg == "--host" && i + 1 < argc) {
            config.server_host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.server_port = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    log_info("Psyne Messaging Patterns Benchmark");
    log_info("=================================");
    log_info("Transport: ", config.transport);
    log_info("Pattern: ", test_pattern);
    log_info("CPU cores: ", std::thread::hardware_concurrency());

    // Initialize thread pool based on configuration
    init_benchmark_pool(config);

    try {
        if (test_pattern == "throughput") {
            BenchmarkStats stats;
            auto channel =
                create_benchmark_channel<BenchmarkMessage>(config, true);

            // Use PsynePool for better thread management
            auto producer_future = benchmark_pool->enqueuePriority(
                PsynePool::HIGH_PRIORITY,
                [&]() { throughput_producer(channel, config, stats, 0); });

            auto consumer_future = benchmark_pool->enqueuePriority(
                PsynePool::HIGH_PRIORITY,
                [&]() { throughput_consumer(channel, config, stats, 0); });

            producer_future.wait();
            consumer_future.wait();

            log_info("Results:");
            log_info("Messages: ", stats.messages_sent.load());
            log_info("Avg latency: ",
                     stats.total_latency_ns.load() /
                         stats.messages_received.load() / 1000,
                     " us");

        } else if (test_pattern == "bandwidth") {
            // Test with various message sizes
            std::vector<size_t> sizes = {1024, 16384, 65536,
                                         1048576}; // 1KB to 1MB
            for (size_t size : sizes) {
                config.message_size = size;
                config.num_messages =
                    std::min(10000UL, 10UL * 1024 * 1024 * 1024 / size);
                bandwidth_test(config);
            }

        } else if (test_pattern == "burst") {
            burst_test(config);

        } else if (test_pattern == "stress") {
            stress_test(config);

        } else {
            log_error("Unknown test pattern: ", test_pattern);
            return 1;
        }

    } catch (const std::exception &e) {
        log_error("Error: ", e.what());
        return 1;
    }

    // Clean up thread pool
    if (benchmark_pool) {
        benchmark_pool->drain(); // Wait for all tasks to complete
        log_info("Benchmark completed successfully");
    }

    return 0;
}