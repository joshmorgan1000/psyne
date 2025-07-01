/**
 * @file data_parallel_broadcast.cpp
 * @brief Example of using SPMC channel for data parallel broadcast
 *
 * Demonstrates a single producer broadcasting work items to multiple
 * consumers for parallel processing. Common in data analysis pipelines.
 */

#include "logger.hpp"
#include "psyne/channel/spmc_channel.hpp"
#include "psyne/core/tensor_message.hpp"

#include <chrono>
#include <thread>
#include <vector>

using namespace psyne;

// Data chunk for parallel processing
struct DataChunk {
    uint32_t chunk_id;
    uint32_t total_chunks;
    float data[1024];  // 4KB of data per chunk
    uint64_t timestamp;
};

/**
 * @brief Producer that generates data chunks
 */
void data_producer(SPMCChannel *channel, size_t num_chunks) {
    thread_context = "Producer";
    
    log_info("Starting data production, ", num_chunks, " chunks");
    
    // auto progress = log_progress("Producing data chunks");
    
    for (size_t i = 0; i < num_chunks; ++i) {
        // Allocate chunk
        auto msg_opt = channel->try_allocate(sizeof(DataChunk));
        if (!msg_opt) {
            log_error("Failed to allocate chunk ", i);
            continue;
        }
        
        auto &msg = *msg_opt;
        auto *chunk = msg.data_as<DataChunk>();
        
        // Fill chunk with data
        chunk->chunk_id = i;
        chunk->total_chunks = num_chunks;
        chunk->timestamp = get_timestamp_ns();
        
        // Simulate data generation
        for (size_t j = 0; j < 1024; ++j) {
            chunk->data[j] = static_cast<float>(i * 1024 + j) * 0.1f;
        }
        
        // Broadcast to all consumers
        channel->commit(msg);
        
        // Update progress
        // progress(static_cast<float>(i + 1) / num_chunks);
        
        // Simulate data generation rate
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    log_info("Completed producing ", num_chunks, " chunks");
}

/**
 * @brief Consumer that processes data chunks
 */
void data_consumer(SPMCChannel::ConsumerHandle *consumer, uint32_t consumer_id) {
    thread_context = "Consumer" + std::to_string(consumer_id);
    
    log_info("Starting data processing");
    
    size_t chunks_processed = 0;
    double sum = 0.0;
    
    while (true) {
        // Receive chunk
        auto msg = consumer->receive();
        auto *chunk = msg.data_as<DataChunk>();
        
        log_info("Received chunk ", chunk->chunk_id, " of ", chunk->total_chunks);
        
        // Process data (compute sum as example)
        for (size_t i = 0; i < 1024; ++i) {
            sum += chunk->data[i];
        }
        
        chunks_processed++;
        
        if (chunks_processed % 5 == 0) {
            log_info("Processed ", chunks_processed, " chunks, sum=", sum);
        }
        
        // Check if this was the last chunk
        if (chunk->chunk_id == chunk->total_chunks - 1) {
            // Release the last message
            consumer->release(msg);
            break;
        }
        
        // Release message for producer to reuse buffer
        consumer->release(msg);
    }
    
    log_info("Completed processing ", chunks_processed, " chunks, final sum=", sum);
}

/**
 * @brief Async consumer using coroutines
 */
boost::asio::awaitable<void> async_data_consumer(
    boost::asio::io_context &io_context,
    SPMCChannel::ConsumerHandle *consumer, 
    uint32_t consumer_id) {
    
    thread_context = "AsyncConsumer" + std::to_string(consumer_id);
    
    log_info("Starting async data processing");
    
    size_t chunks_processed = 0;
    double sum = 0.0;
    
    while (true) {
        // Async receive chunk
        auto msg = co_await consumer->async_receive();
        auto *chunk = msg.data_as<DataChunk>();
        
        // Process data
        for (size_t i = 0; i < 1024; ++i) {
            sum += chunk->data[i];
        }
        
        chunks_processed++;
        
        if (chunks_processed % 100 == 0) {
            log_debug("Async processed ", chunks_processed, " chunks, sum=", sum);
        }
        
        // Check if this was the last chunk
        bool is_last = (chunk->chunk_id == chunk->total_chunks - 1);
        
        // Release message
        consumer->release(msg);
        
        if (is_last) break;
        
        // Yield to other coroutines
        co_await boost::asio::post(io_context, boost::asio::use_awaitable);
    }
    
    log_info("Async completed processing ", chunks_processed, " chunks, final sum=", sum);
}

int main() {
    thread_context = "Main";
    
    log_info("=== SPMC Data Parallel Broadcast Demo ===");
    
    const size_t num_consumers = 2;
    const size_t num_async_consumers = 0;  // Disable async for now
    const size_t num_chunks = 10;
    
    // Create io_context for async operations
    boost::asio::io_context io_context;
    
    // Create SPMC channel for data broadcast
    ChannelConfig config{
        .size_mb = 32,  // 32MB buffer
        .mode = ChannelMode::SPMC,
        .use_huge_pages = true,
        .blocking = true,
        .name = "broadcast_channel"
    };
    
    SPMCChannel channel(config, &io_context);
    
    log_info("Created SPMC channel with ", config.size_mb, "MB capacity");
    log_info("Starting ", num_consumers, " sync + ", num_async_consumers, " async consumers");
    
    // Register consumers and start worker threads
    std::vector<std::thread> consumer_threads;
    std::vector<std::unique_ptr<SPMCChannel::ConsumerHandle>> consumers;
    
    // Start synchronous consumers
    for (uint32_t i = 0; i < num_consumers; ++i) {
        auto consumer = channel.register_consumer();
        if (!consumer) {
            log_error("Failed to register consumer ", i);
            return 1;
        }
        consumer_threads.emplace_back(data_consumer, consumer.get(), i);
        consumers.push_back(std::move(consumer));
    }
    
    // Start async consumers
    std::vector<std::unique_ptr<SPMCChannel::ConsumerHandle>> async_consumers;
    for (uint32_t i = 0; i < num_async_consumers; ++i) {
        auto consumer = channel.register_consumer();
        if (!consumer) {
            log_error("Failed to register async consumer ", i);
            return 1;
        }
        
        // Launch coroutine
        boost::asio::co_spawn(io_context,
            async_data_consumer(io_context, consumer.get(), num_consumers + i),
            boost::asio::detached);
            
        async_consumers.push_back(std::move(consumer));
    }
    
    // Start io_context in separate thread
    std::thread io_thread([&io_context]() {
        thread_context = "IOContext";
        io_context.run();
    });
    
    // Start producer
    std::thread producer_thread(data_producer, &channel, num_chunks);
    
    // Wait for producer to complete
    producer_thread.join();
    
    // Wait for sync consumers
    for (auto &thread : consumer_threads) {
        thread.join();
    }
    
    // Stop io_context and wait for async consumers
    io_context.stop();
    io_thread.join();
    
    // Print statistics
    auto stats = channel.get_stats();
    
    log_info("\n=== Channel Statistics ===");
    log_info("Messages sent: ", stats.messages_sent);
    log_info("Messages received: ", stats.messages_received);
    log_info("Total consumers: ", num_consumers + num_async_consumers);
    log_info("Messages per consumer: ", stats.messages_received / 
             (num_consumers + num_async_consumers));
    log_info("Bytes transferred: ", stats.bytes_sent / 1024.0 / 1024.0, " MB");
    log_info("Average latency: ", stats.avg_latency_ns, " ns");
    log_info("Min latency: ", stats.min_latency_ns, " ns");
    log_info("Max latency: ", stats.max_latency_ns, " ns");
    
    return 0;
}