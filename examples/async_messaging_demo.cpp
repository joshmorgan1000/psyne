/**
 * @file async_messaging_demo.cpp
 * @brief Demonstration of async/await messaging with Psyne
 *
 * This example shows how to use boost.asio coroutines with Psyne channels
 * for async message handling with configurable thread pools.
 */

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/io_context.hpp>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

// Enable async support
#define PSYNE_ASYNC_SUPPORT
#include "../src/async/async_channel.hpp"
#include "../src/utils/pthread.hpp"

using namespace psyne;
using namespace psyne::async;

// Simple message type
struct SensorData : public Message<SensorData> {
    static constexpr uint32_t message_type = 100;

    float temperature;
    float humidity;
    uint64_t timestamp;

    using Message<SensorData>::Message;

    static size_t calculate_size() {
        return sizeof(float) * 2 + sizeof(uint64_t);
    }

    void initialize() {
        temperature = 0.0f;
        humidity = 0.0f;
        timestamp = 0;
    }
};

// Async producer coroutine
boost::asio::awaitable<void> async_producer(AsyncChannel<Channel> &channel) {
    for (int i = 0; i < 10; ++i) {
        // Create and send sensor data
        auto msg = channel.reserve<SensorData>();
        if (msg) {
            msg->temperature = 20.0f + (i * 0.5f);
            msg->humidity = 50.0f + (i * 2.0f);
            msg->timestamp =
                std::chrono::system_clock::now().time_since_epoch().count();
            msg->send();

            std::cout << "Sent sensor data #" << i
                      << ": temp=" << msg->temperature << "°C" << std::endl;
        }

        // Async delay
        boost::asio::steady_timer timer(
            co_await boost::asio::this_coro::executor);
        timer.expires_after(std::chrono::milliseconds(100));
        co_await timer.async_wait(boost::asio::use_awaitable);
    }
}

// Async consumer coroutine
boost::asio::awaitable<void> async_consumer(AsyncChannel<Channel> &channel) {
    int count = 0;
    while (count < 10) {
        // Await message with timeout
        auto msg = co_await channel.async_receive<SensorData>(
            std::chrono::milliseconds(1000));

        if (msg) {
            std::cout << "Received sensor data: temp=" << msg->temperature
                      << "°C, humidity=" << msg->humidity << "%" << std::endl;
            count++;
        } else {
            std::cout << "Timeout waiting for message" << std::endl;
        }
    }
}

// Handler-based example with thread pool
void handler_based_example() {
    std::cout << "\n=== Handler-based Example with Thread Pool ==="
              << std::endl;

    // Create thread pool for handlers
    PsynePool handler_pool(4, 8); // 4-8 threads

    // Create async channel
    auto base_channel = create_channel("memory://handler_demo", 64 * 1024);
    AsyncChannel<Channel> channel(base_channel->uri(),
                                  base_channel->buffer_size_);

    // Register async handler with thread pool
    auto handler_token = channel.register_async_handler<SensorData>(
        [](SensorData &&msg) {
            // This runs in the thread pool
            std::cout << "[Thread " << std::this_thread::get_id()
                      << "] Processing: temp=" << msg.temperature << "°C"
                      << std::endl;

            // Simulate processing
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        },
        AsyncHandlerConfig{.max_concurrent_handlers = 4,
                           .use_thread_pool = true,
                           .thread_pool = &handler_pool});

    // Start async processing
    channel.start_async();

    // Send messages
    std::thread producer([&channel]() {
        for (int i = 0; i < 20; ++i) {
            auto msg = channel.reserve<SensorData>();
            if (msg) {
                msg->temperature = 15.0f + i;
                msg->humidity = 60.0f;
                msg->timestamp = i;
                msg->send();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    });

    producer.join();
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Stop handler
    *handler_token = false;
    channel.stop_async();
}

// Coroutine example
void coroutine_example() {
    std::cout << "\n=== Coroutine Example ===" << std::endl;

    // Shared io_context for both channels
    auto io_ctx = std::make_shared<boost::asio::io_context>();

    // Create async channels
    auto base_channel = create_channel("memory://coro_demo", 64 * 1024);
    AsyncChannel<Channel> channel1(base_channel->uri(),
                                   base_channel->buffer_size_, io_ctx);
    AsyncChannel<Channel> channel2(base_channel->uri(),
                                   base_channel->buffer_size_, io_ctx);

    // Spawn coroutines
    boost::asio::co_spawn(*io_ctx, async_producer(channel1),
                          boost::asio::detached);
    boost::asio::co_spawn(*io_ctx, async_consumer(channel2),
                          boost::asio::detached);

    // Run event loop
    io_ctx->run();
}

int main() {
    std::cout << "=== Psyne Async Messaging Demo ===" << std::endl;
    std::cout
        << "Demonstrating boost.asio coroutines and thread pool handlers\n"
        << std::endl;

    try {
        // Run examples
        coroutine_example();
        handler_based_example();

        std::cout << "\nDemo completed successfully!" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}