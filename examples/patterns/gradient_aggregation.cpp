/**
 * @file gradient_aggregation.cpp
 * @brief Example of using MPSC channel for gradient aggregation
 *
 * Demonstrates multiple workers computing gradients and sending them
 * to a single parameter server for aggregation.
 */

#include "logger.hpp"
#include "psyne/channel/channel.hpp"
#include "psyne/core/tensor_message.hpp"

#include <chrono>
#include <random>
#include <thread>
#include <vector>

using namespace psyne;

// Gradient message for a layer with 512 parameters
using LayerGradient = GradientMessage<512>;

/**
 * @brief Worker thread that computes gradients
 */
void gradient_worker(uint32_t worker_id,
                     std::shared_ptr<Channel<LayerGradient>> channel,
                     std::unique_ptr<MPSCProducer> producer,
                     size_t num_iterations) {
    thread_context = "Worker" + std::to_string(worker_id);

    std::mt19937 rng(worker_id);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    log_info("Starting gradient computation");

    for (size_t iter = 0; iter < num_iterations; ++iter) {
        // Simulate gradient computation time
        std::this_thread::sleep_for(std::chrono::microseconds(100));

        // Allocate gradient message
        auto msg_opt = producer->try_allocate(sizeof(LayerGradient));
        if (!msg_opt) {
            log_error("Failed to allocate gradient message");
            continue;
        }

        auto &msg = *msg_opt;
        auto *grad = msg.data_as<LayerGradient>();

        // Fill with simulated gradients
        for (size_t i = 0; i < 512; ++i) {
            grad->gradients[i] = dist(rng);
            grad->momentum[i] = 0.0f; // Initialize momentum
        }

        grad->layer_id = 1;
        grad->parameter_id = worker_id;
        grad->iteration = iter;
        grad->learning_rate = 0.001f;

        // Send to parameter server
        producer->commit(msg);

        if (iter % 100 == 0) {
            log_debug("Computed and sent gradient for iteration ", iter);
        }
    }

    log_info("Completed ", num_iterations, " gradient computations");
}

/**
 * @brief Parameter server that aggregates gradients
 */
void parameter_server(std::shared_ptr<Channel<LayerGradient>> channel,
                      size_t num_workers, size_t num_iterations) {
    thread_context = "ParamServer";

    log_info("Starting parameter server for ", num_workers, " workers");

    // Accumulated gradients
    std::vector<float> accumulated_gradients(512, 0.0f);
    std::vector<float> parameters(512, 0.0f);

    size_t gradients_received = 0;
    size_t gradients_per_batch = num_workers;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (gradients_received < num_workers * num_iterations) {
        auto msg = channel->receive();
        auto *grad = msg.data();

        // Accumulate gradients
        for (size_t i = 0; i < 512; ++i) {
            accumulated_gradients[i] += grad->gradients[i];
        }

        gradients_received++;

        // Apply updates when we have gradients from all workers
        if (gradients_received % gradients_per_batch == 0) {
            // Average gradients
            for (size_t i = 0; i < 512; ++i) {
                accumulated_gradients[i] /= num_workers;

                // Apply gradient update (simple SGD)
                parameters[i] -= grad->learning_rate * accumulated_gradients[i];

                // Reset accumulator
                accumulated_gradients[i] = 0.0f;
            }

            size_t batch_num = gradients_received / gradients_per_batch;
            if (batch_num % 10 == 0) {
                log_info("Applied gradient update for batch ", batch_num);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    log_info("Processed ", gradients_received, " gradients in ",
             duration.count(), "ms");
    log_info("Throughput: ", (gradients_received * 1000.0) / duration.count(),
             " gradients/sec");
}

int main() {
    thread_context = "Main";

    log_info("=== MPSC Gradient Aggregation Demo ===");

    const size_t num_workers = 4;
    const size_t num_iterations = 1000;

    // Create MPSC channel for gradient aggregation
    ChannelConfig config{.size_mb = 64, // 64MB total
                         .mode = ChannelMode::MPSC,
                         .use_huge_pages = true,
                         .blocking = true,
                         .name = "gradient_channel"};

    auto channel = Channel<LayerGradient>::create(config);

    log_info("Created MPSC channel with ", config.size_mb, "MB capacity");
    log_info("Starting ", num_workers, " workers, ", num_iterations,
             " iterations each");

    // Start parameter server
    std::thread server_thread(parameter_server, channel, num_workers,
                              num_iterations);

    // Start workers
    std::vector<std::thread> worker_threads;
    std::vector<std::unique_ptr<MPSCProducer>> producers;

    for (uint32_t i = 0; i < num_workers; ++i) {
        auto producer = channel->register_mpsc_producer();
        worker_threads.emplace_back(gradient_worker, i, channel,
                                    std::move(producer), num_iterations);
    }

    // Wait for completion
    for (auto &thread : worker_threads) {
        thread.join();
    }
    server_thread.join();

    // Print statistics
    auto stats = channel->get_stats();

    log_info("\n=== Channel Statistics ===");
    log_info("Messages sent: ", stats.messages_sent);
    log_info("Messages received: ", stats.messages_received);
    log_info("Bytes transferred: ", stats.bytes_sent / 1024.0 / 1024.0, " MB");
    log_info("Average latency: ", stats.avg_latency_ns, " ns");
    log_info("Min latency: ", stats.min_latency_ns, " ns");
    log_info("Max latency: ", stats.max_latency_ns, " ns");
    log_info("Send failures: ", stats.send_failures);
    log_info("Receive failures: ", stats.receive_failures);

    return 0;
}