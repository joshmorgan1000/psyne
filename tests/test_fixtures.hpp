#pragma once

#include <psyne/psyne.hpp>
#include <memory>
#include <string>
#include <chrono>

namespace psyne {
namespace test {

/**
 * @brief Base test fixture for Psyne tests
 * 
 * Provides common setup and teardown functionality for all tests
 */
class PsyneTestFixture {
public:
    PsyneTestFixture() {
        setup();
    }
    
    virtual ~PsyneTestFixture() {
        teardown();
    }
    
protected:
    virtual void setup() {
        // Common setup for all tests
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    virtual void teardown() {
        // Common cleanup for all tests
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        
        // Optional: Log test duration
        (void)duration; // Suppress unused variable warning
    }
    
    /**
     * @brief Create a test channel with default settings
     */
    std::unique_ptr<Channel> create_test_channel(
        const std::string& name = "test",
        size_t buffer_size = 1024 * 1024,
        ChannelMode mode = ChannelMode::SPSC,
        ChannelType type = ChannelType::SingleType
    ) {
        std::string uri = "memory://" + name + "_" + std::to_string(++channel_counter_);
        return Channel::create(uri, buffer_size, mode, type);
    }
    
    /**
     * @brief Create a test IPC channel
     */
    std::unique_ptr<Channel> create_test_ipc_channel(
        const std::string& name = "ipc_test",
        size_t buffer_size = 1024 * 1024
    ) {
        std::string uri = "ipc://" + name + "_" + std::to_string(++channel_counter_);
        return Channel::create(uri, buffer_size, ChannelMode::SPSC, ChannelType::SingleType);
    }
    
    /**
     * @brief Verify channel creation was successful
     */
    void assert_channel_valid(const std::unique_ptr<Channel>& channel) {
        assert(channel != nullptr);
        assert(!channel->uri().empty());
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    static int channel_counter_;
};

// Static member definition
int PsyneTestFixture::channel_counter_ = 0;

/**
 * @brief Test fixture for messaging tests
 * 
 * Provides pre-configured channels and message types for messaging tests
 */
class MessagingTestFixture : public PsyneTestFixture {
public:
    MessagingTestFixture() {
        setup_messaging();
    }
    
protected:
    void setup_messaging() {
        // Create default test channels
        memory_channel_ = create_test_channel("messaging_test");
        assert_channel_valid(memory_channel_);
        
        try {
            ipc_channel_ = create_test_ipc_channel("messaging_ipc_test");
            assert_channel_valid(ipc_channel_);
        } catch (const std::exception&) {
            // IPC might not be available in all test environments
            ipc_channel_ = nullptr;
        }
    }
    
    /**
     * @brief Send and receive a test message
     */
    template<typename MessageType>
    bool test_message_roundtrip(Channel& channel) {
        // Send message
        MessageType msg(channel);
        msg.send();
        
        // Receive message
        auto received = channel.receive_single<MessageType>();
        return received.has_value();
    }
    
protected:
    std::unique_ptr<Channel> memory_channel_;
    std::unique_ptr<Channel> ipc_channel_;
};

/**
 * @brief Test fixture for performance tests
 * 
 * Provides timing utilities and performance measurement helpers
 */
class PerformanceTestFixture : public PsyneTestFixture {
public:
    PerformanceTestFixture() {
        setup_performance();
    }
    
protected:
    void setup_performance() {
        // Create high-performance channel
        perf_channel_ = create_test_channel("perf_test", 64 * 1024 * 1024); // 64MB
        assert_channel_valid(perf_channel_);
    }
    
    /**
     * @brief Measure execution time of a function
     */
    template<typename Func>
    std::chrono::microseconds measure_execution_time(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    
    /**
     * @brief Benchmark message throughput
     */
    template<typename MessageType>
    double benchmark_throughput(Channel& channel, int num_messages) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_messages; ++i) {
            MessageType msg(channel);
            msg.send();
            
            auto received = channel.receive_single<MessageType>();
            assert(received.has_value());
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return static_cast<double>(num_messages) / (duration.count() / 1e6);
    }
    
protected:
    std::unique_ptr<Channel> perf_channel_;
};

/**
 * @brief Test fixture for multi-threaded tests
 * 
 * Provides utilities for concurrent testing
 */
class ConcurrencyTestFixture : public PsyneTestFixture {
public:
    ConcurrencyTestFixture() {
        setup_concurrency();
    }
    
protected:
    void setup_concurrency() {
        // Create channels for multi-threaded testing
        for (int i = 0; i < 4; ++i) {
            auto channel = create_test_channel("concurrent_test_" + std::to_string(i));
            assert_channel_valid(channel);
            concurrent_channels_.push_back(std::move(channel));
        }
    }
    
    /**
     * @brief Run a function concurrently on multiple threads
     */
    template<typename Func>
    void run_concurrent(Func&& func, int num_threads = 4) {
        std::vector<std::thread> threads;
        
        for (int i = 0; i < num_threads && i < static_cast<int>(concurrent_channels_.size()); ++i) {
            threads.emplace_back([&func, &channel = *concurrent_channels_[i], i]() {
                func(channel, i);
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
protected:
    std::vector<std::unique_ptr<Channel>> concurrent_channels_;
};

} // namespace test
} // namespace psyne