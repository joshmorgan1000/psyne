#include "test_fixtures.hpp"
#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <thread>

using namespace psyne;
using namespace psyne::test;

/**
 * @brief Memory profiling benchmark for Psyne channels
 */
class MemoryProfilingBenchmark : public PerformanceTestFixture {
protected:
    void SetUp() override {
        PerformanceTestFixture::SetUp();

        // Clear any existing profiling data
        memory_samples_.clear();
        baseline_memory_ = get_memory_usage();
    }

    void TearDown() override {
        // Generate memory usage report
        generate_memory_report();
        PerformanceTestFixture::TearDown();
    }

    struct MemorySample {
        std::chrono::steady_clock::time_point timestamp;
        size_t rss_bytes;      // Resident Set Size
        size_t vms_bytes;      // Virtual Memory Size
        size_t heap_bytes;     // Heap usage (if available)
        std::string operation; // What operation was being performed
    };

    std::vector<MemorySample> memory_samples_;
    size_t baseline_memory_ = 0;

    /**
     * @brief Get current memory usage
     */
    MemorySample get_memory_sample(const std::string &operation = "") {
        MemorySample sample;
        sample.timestamp = std::chrono::steady_clock::now();
        sample.operation = operation;

#ifdef __APPLE__
        // macOS memory usage via task_info
        task_basic_info info;
        mach_msg_type_number_t size = sizeof(info);
        kern_return_t kerr = task_info(mach_task_self(), TASK_BASIC_INFO,
                                       (task_info_t)&info, &size);
        if (kerr == KERN_SUCCESS) {
            sample.rss_bytes = info.resident_size;
            sample.vms_bytes = info.virtual_size;
        }

        // Get heap info from malloc
        struct mstats heap_stats = mstats();
        sample.heap_bytes = heap_stats.bytes_used;

#elif defined(__linux__)
        // Linux memory usage via /proc/self/status
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                sample.rss_bytes = std::stoul(line.substr(7)) * 1024;
            } else if (line.substr(0, 6) == "VmSize:") {
                sample.vms_bytes = std::stoul(line.substr(7)) * 1024;
            }
        }

        // Get heap info from mallinfo
        struct mallinfo heap_info = mallinfo();
        sample.heap_bytes = heap_info.uordblks;
#else
        // Fallback for other platforms
        sample.rss_bytes = 0;
        sample.vms_bytes = 0;
        sample.heap_bytes = 0;
#endif

        return sample;
    }

    size_t get_memory_usage() {
        return get_memory_sample().rss_bytes;
    }

    void record_memory_sample(const std::string &operation) {
        memory_samples_.push_back(get_memory_sample(operation));
    }

    void generate_memory_report() {
        if (memory_samples_.empty())
            return;

        std::cout << "\n=== MEMORY PROFILING REPORT ===\n";
        std::cout << std::fixed << std::setprecision(2);

        auto start_time = memory_samples_[0].timestamp;

        std::cout << "Time(s)\tRSS(MB)\tVMS(MB)\tHeap(MB)\tOperation\n";
        std::cout << "------\t-------\t-------\t--------\t---------\n";

        for (const auto &sample : memory_samples_) {
            auto elapsed =
                std::chrono::duration<double>(sample.timestamp - start_time)
                    .count();

            std::cout << elapsed << "\t" << (sample.rss_bytes / 1024.0 / 1024.0)
                      << "\t" << (sample.vms_bytes / 1024.0 / 1024.0) << "\t"
                      << (sample.heap_bytes / 1024.0 / 1024.0) << "\t"
                      << sample.operation << "\n";
        }

        // Calculate peak usage
        auto max_rss =
            std::max_element(memory_samples_.begin(), memory_samples_.end(),
                             [](const MemorySample &a, const MemorySample &b) {
                                 return a.rss_bytes < b.rss_bytes;
                             });

        auto max_heap =
            std::max_element(memory_samples_.begin(), memory_samples_.end(),
                             [](const MemorySample &a, const MemorySample &b) {
                                 return a.heap_bytes < b.heap_bytes;
                             });

        std::cout << "\n=== PEAK USAGE ===\n";
        std::cout << "Peak RSS:  " << (max_rss->rss_bytes / 1024.0 / 1024.0)
                  << " MB\n";
        std::cout << "Peak Heap: " << (max_heap->heap_bytes / 1024.0 / 1024.0)
                  << " MB\n";
        std::cout << "Baseline:  " << (baseline_memory_ / 1024.0 / 1024.0)
                  << " MB\n";
        std::cout << "Growth:    "
                  << ((max_rss->rss_bytes - baseline_memory_) / 1024.0 / 1024.0)
                  << " MB\n";
    }

private:
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#include <malloc/malloc.h>
#elif defined(__linux__)
#include <fstream>
#include <malloc.h>
#endif
};

/**
 * @brief Test memory usage during channel creation and destruction
 */
TEST_F(MemoryProfilingBenchmark, ChannelLifecycleMemory) {
    record_memory_sample("baseline");

    const int num_channels = 100;
    std::vector<std::unique_ptr<Channel>> channels;

    // Create channels and measure memory growth
    for (int i = 0; i < num_channels; ++i) {
        channels.push_back(create_test_channel(
            "memory_test_" + std::to_string(i), 1024 * 1024));

        if (i % 10 == 0) {
            record_memory_sample("created_" + std::to_string(i) + "_channels");
        }
    }

    record_memory_sample("all_channels_created");

    // Clear half the channels
    channels.erase(channels.begin(), channels.begin() + num_channels / 2);
    record_memory_sample("half_channels_destroyed");

    // Clear remaining channels
    channels.clear();
    record_memory_sample("all_channels_destroyed");

    // Force garbage collection
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    record_memory_sample("after_gc_wait");
}

/**
 * @brief Test memory usage during high-volume messaging
 */
TEST_F(MemoryProfilingBenchmark, HighVolumeMessagingMemory) {
    auto channel = create_test_channel("high_volume_test",
                                       64 * 1024 * 1024); // 64MB buffer
    record_memory_sample("channel_created");

    const int num_messages = 10000;
    const size_t message_size = 4096; // 4KB messages

    // Send burst of messages
    for (int i = 0; i < num_messages; ++i) {
        ByteVector msg(*channel);
        msg.resize(message_size);

        // Fill with test data
        for (size_t j = 0; j < message_size; ++j) {
            msg[j] = static_cast<uint8_t>((i + j) % 256);
        }

        msg.send();

        if (i % 1000 == 0) {
            record_memory_sample("sent_" + std::to_string(i) + "_messages");
        }
    }

    record_memory_sample("all_messages_sent");

    // Receive all messages
    int received_count = 0;
    while (auto received = channel->receive_single<ByteVector>()) {
        received_count++;

        if (received_count % 1000 == 0) {
            record_memory_sample("received_" + std::to_string(received_count) +
                                 "_messages");
        }
    }

    record_memory_sample("all_messages_received");
    EXPECT_EQ(received_count, num_messages);
}

/**
 * @brief Test memory usage with different buffer sizes
 */
TEST_F(MemoryProfilingBenchmark, BufferSizeScaling) {
    record_memory_sample("baseline");

    std::vector<size_t> buffer_sizes = {
        1024,             // 1KB
        64 * 1024,        // 64KB
        1024 * 1024,      // 1MB
        16 * 1024 * 1024, // 16MB
        64 * 1024 * 1024  // 64MB
    };

    std::vector<std::unique_ptr<Channel>> channels;

    for (size_t buffer_size : buffer_sizes) {
        auto channel = create_test_channel(
            "buffer_test_" + std::to_string(buffer_size), buffer_size);
        channels.push_back(std::move(channel));

        record_memory_sample("buffer_" + std::to_string(buffer_size / 1024) +
                             "KB");
    }

    // Test messaging with largest buffer
    auto &large_channel = channels.back();
    const int num_large_messages = 100;
    const size_t large_message_size = 1024 * 1024; // 1MB messages

    for (int i = 0; i < num_large_messages; ++i) {
        ByteVector msg(*large_channel);
        msg.resize(large_message_size);
        msg.send();

        if (i % 20 == 0) {
            record_memory_sample("large_msg_" + std::to_string(i));
        }
    }

    record_memory_sample("large_messages_sent");

    // Cleanup
    channels.clear();
    record_memory_sample("cleanup_complete");
}

/**
 * @brief Test memory usage in multi-threaded scenario
 */
TEST_F(MemoryProfilingBenchmark, MultiThreadedMemoryUsage) {
    record_memory_sample("baseline");

    const int num_threads = 8;
    const int messages_per_thread = 1000;

    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<Channel>> channels;

    // Create channels for each thread
    for (int i = 0; i < num_threads; ++i) {
        channels.push_back(
            create_test_channel("thread_" + std::to_string(i), 1024 * 1024));
    }

    record_memory_sample("channels_created");

    // Launch threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            auto &channel = channels[t];

            for (int i = 0; i < messages_per_thread; ++i) {
                FloatVector msg(*channel);
                msg.resize(100); // 100 floats

                for (int j = 0; j < 100; ++j) {
                    msg[j] = static_cast<float>(t * 1000 + i * 100 + j);
                }

                msg.send();

                // Immediately receive to prevent queue buildup
                auto received = channel->receive_single<FloatVector>();
                EXPECT_TRUE(received.has_value());
            }
        });
    }

    record_memory_sample("threads_started");

    // Wait for completion
    for (auto &thread : threads) {
        thread.join();
    }

    record_memory_sample("threads_completed");

    // Cleanup
    channels.clear();
    record_memory_sample("cleanup_complete");
}