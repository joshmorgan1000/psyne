/**
 * @file layer_communication.cpp
 * @brief Example of using Psyne channels for neural network layer communication
 *
 * Demonstrates zero-copy tensor passing between neural network layers
 * with sub-microsecond latency.
 */

#include "logger.hpp"
#include "psyne/channel/channel.hpp"
#include "psyne/core/tensor_message.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

using namespace psyne;

/**
 * @brief Simulated neural network layer that produces embeddings
 */
void embedding_layer(
    std::shared_ptr<Channel<Embedding256Message>> output_channel,
    size_t num_batches) {
    thread_context = "EmbeddingLayer";

    log_info("Starting embedding layer with ", num_batches, " batches");

    for (size_t batch = 0; batch < num_batches; ++batch) {
        // Allocate message directly in channel memory (zero-copy)
        auto msg = output_channel->allocate();

        // Set metadata
        msg->batch_idx = batch;
        msg->layer_id = 1;
        msg.set_type(static_cast<uint16_t>(TensorMessageType::FLOAT32_VECTOR));

        // Fill embedding data (in real scenario, this would be from lookup
        // table)
        auto eigen_vec = msg->as_eigen();
        eigen_vec.setRandom(); // Simulate embedding lookup
        eigen_vec.normalize(); // Normalize embeddings

        // Send to next layer (just updates pointers, no copy)
        msg.send();

        if (batch % 100 == 0) {
            log_debug("Embedding layer processed batch ", batch);
        }
    }

    log_info("Embedding layer completed ", num_batches, " batches");
}

/**
 * @brief Simulated transformer layer that processes embeddings
 */
void transformer_layer(
    std::shared_ptr<Channel<Embedding256Message>> input_channel,
    std::shared_ptr<Channel<Embedding256Message>> output_channel,
    size_t num_batches) {
    thread_context = "TransformerLayer";

    log_info("Starting transformer layer with ", num_batches, " batches");

    // Simulate transformer weights (use dynamic allocation for large matrices)
    Eigen::MatrixXf attention_weights(256, 256);
    attention_weights.setRandom();

    for (size_t batch = 0; batch < num_batches; ++batch) {
        // Receive embedding from previous layer
        auto input_msg = input_channel->receive();

        // Allocate output message
        auto output_msg = output_channel->allocate();

        // Process through transformer (simplified)
        auto input_eigen = input_msg->as_eigen();
        auto output_eigen = output_msg->as_eigen();

        // Self-attention (simplified as matrix multiply)
        output_eigen = attention_weights * input_eigen;
        output_eigen.normalize();

        // Copy metadata
        output_msg->batch_idx = input_msg->batch_idx;
        output_msg->layer_id = 2;
        output_msg.set_type(
            static_cast<uint16_t>(TensorMessageType::ACTIVATION));

        // Send to next layer
        output_msg.send();

        // Input message automatically released when it goes out of scope
    }

    log_info("Transformer layer completed ", num_batches, " batches");
}

/**
 * @brief Final layer that consumes processed embeddings
 */
void output_layer(std::shared_ptr<Channel<Embedding256Message>> input_channel,
                  size_t num_batches) {
    thread_context = "OutputLayer";

    log_info("Starting output layer with ", num_batches, " batches");

    uint64_t total_latency_ns = 0;
    uint64_t min_latency_ns = UINT64_MAX;
    uint64_t max_latency_ns = 0;

    for (size_t batch = 0; batch < num_batches; ++batch) {
        // Receive processed embedding
        auto msg = input_channel->receive();

        // Calculate end-to-end latency
        uint64_t latency = get_timestamp_ns() - msg.header()->timestamp_ns;
        total_latency_ns += latency;
        min_latency_ns = std::min(min_latency_ns, latency);
        max_latency_ns = std::max(max_latency_ns, latency);

        // Simulate final processing (e.g., classification)
        auto eigen_vec = msg->as_eigen();
        float prediction = eigen_vec.sum(); // Simplified prediction
    }

    // Report latency statistics
    double avg_latency_ns = static_cast<double>(total_latency_ns) / num_batches;
    log_info("Output layer completed ", num_batches, " batches");
    log_info("End-to-end latency - Avg: ", std::fixed, std::setprecision(0),
             avg_latency_ns, "ns, Min: ", min_latency_ns,
             "ns, Max: ", max_latency_ns, "ns");
}

int main() {
    // Initialize logging
    thread_context = "Main";

    log_info("=== Psyne Neural Network Layer Communication Demo ===");

    // Create channels between layers
    ChannelConfig config{.size_mb = 16, // 16MB per channel
                         .mode = ChannelMode::SPSC,
                         .use_huge_pages = true,
                         .gpu_enabled = false, // CPU-only for this demo
                         .blocking = true,
                         .name = ""};

    config.name = "embedding_to_transformer";
    auto channel1 = Channel<Embedding256Message>::create(config);

    config.name = "transformer_to_output";
    auto channel2 = Channel<Embedding256Message>::create(config);

    const size_t num_batches = 10000;

    log_info("Processing ", num_batches, " batches through 3-layer network");
    log_info("Message size: ", sizeof(Embedding256Message), " bytes");

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch layers in separate threads
    std::thread embedding_thread(embedding_layer, channel1, num_batches);
    std::thread transformer_thread(transformer_layer, channel1, channel2,
                                   num_batches);
    std::thread output_thread(output_layer, channel2, num_batches);

    // Wait for completion
    embedding_thread.join();
    transformer_thread.join();
    output_thread.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Print statistics
    log_info("\n=== Performance Summary ===");
    log_info("Total time: ", duration.count(), "ms");
    log_info("Throughput: ", std::fixed, std::setprecision(2),
             (num_batches * 1000.0) / duration.count(), " messages/sec");

    auto stats1 = channel1->get_stats();
    auto stats2 = channel2->get_stats();

    log_info("\n=== Channel Statistics ===");
    log_info("Channel 1 (embedding->transformer):");
    log_info("  Messages: ", stats1.messages_sent, " sent, ",
             stats1.messages_received, " received");
    log_info("  Data: ", stats1.bytes_sent / 1024.0 / 1024.0, " MB sent, ",
             stats1.bytes_received / 1024.0 / 1024.0, " MB received");
    log_info("  Avg latency: ", std::fixed, std::setprecision(0),
             stats1.avg_latency_ns, " ns");

    log_info("\nChannel 2 (transformer->output):");
    log_info("  Messages: ", stats2.messages_sent, " sent, ",
             stats2.messages_received, " received");
    log_info("  Data: ", stats2.bytes_sent / 1024.0 / 1024.0, " MB sent, ",
             stats2.bytes_received / 1024.0 / 1024.0, " MB received");
    log_info("  Avg latency: ", std::fixed, std::setprecision(0),
             stats2.avg_latency_ns, " ns");

    return 0;
}