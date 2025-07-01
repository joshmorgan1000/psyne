/**
 * @file ipc_example.cpp
 * @brief Example of inter-process communication using Psyne IPC channels
 *
 * This example demonstrates zero-copy IPC between two processes.
 * Shows the v2.0 substrate + pattern + message architecture.
 */

#include "../../include/psyne/core/behaviors.hpp"
#include "../../include/psyne/channel/substrate/ipc.hpp"
#include "../../include/psyne/channel/pattern/spsc.hpp"
#include <iostream>
#include <chrono>
#include <cstring>
#include <thread>
#include <atomic>

/**
 * @brief Simple message structure for testing
 */
struct TestMessage {
    uint64_t sequence;
    uint64_t timestamp;
    char data[256];
};

void run_producer() {
    log_info("Starting IPC producer...");

    // Create channel configuration for IPC
    ChannelConfig config;
    config.name = "test_ipc_channel";
    config.size_mb = 16;
    config.mode = ChannelMode::SPSC;
    config.transport = ChannelTransport::IPC;
    config.is_producer = true;
    config.blocking = true;

    // Create the channel
    auto channel = Channel<TestMessage>::create(config);

    log_info("Producer channel created, sending messages...");

    // Send messages
    for (uint64_t i = 0; i < 1000; ++i) {
        auto msg = channel->allocate();

        msg->sequence = i;
        msg->timestamp = std::chrono::high_resolution_clock::now()
                             .time_since_epoch()
                             .count();
        snprintf(msg->data, sizeof(msg->data), "Message %lu from producer", i);

        msg.send();

        if (i % 100 == 0) {
            log_info("Sent ", i, " messages");
        }

        // Small delay to simulate work
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // Print statistics
    auto stats = channel->get_stats();
    log_info("Producer Statistics:");
    log_info("  Messages sent: ", stats.messages_sent);
    log_info("  Bytes sent: ", stats.bytes_sent);
    log_info("  Send failures: ", stats.send_failures);
    log_info("  Average latency: ", stats.avg_latency_ns, " ns");
}

void run_consumer() {
    log_info("Starting IPC consumer...");

    // Create channel configuration for IPC
    ChannelConfig config;
    config.name = "test_ipc_channel";
    config.size_mb = 16;
    config.mode = ChannelMode::SPSC;
    config.transport = ChannelTransport::IPC;
    config.is_producer = false;
    config.blocking = true;

    // Create the channel
    auto channel = Channel<TestMessage>::create(config);

    log_info("Consumer channel created, receiving messages...");

    // Receive messages
    uint64_t expected_sequence = 0;
    uint64_t total_latency = 0;
    uint64_t message_count = 0;

    while (expected_sequence < 1000) {
        auto msg = channel->receive();

        // Verify sequence
        if (msg->sequence != expected_sequence) {
            log_error("ERROR: Expected sequence ", expected_sequence,
                      " but got ", msg->sequence);
        }

        // Calculate end-to-end latency
        uint64_t now = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
        uint64_t latency = now - msg->timestamp;
        total_latency += latency;
        message_count++;

        if (expected_sequence % 100 == 0) {
            log_info("Received ", expected_sequence,
                      " messages, data: ", msg->data);
        }

        expected_sequence++;
    }

    // Print statistics
    auto stats = channel->get_stats();
    log_info("Consumer Statistics:");
    log_info("  Messages received: ", stats.messages_received);
    log_info("  Bytes received: ", stats.bytes_received);
    log_info("  Receive failures: ", stats.receive_failures);
    log_info("  Average channel latency: ", stats.avg_latency_ns, " ns");
    log_info("  Average end-to-end latency: ",
              (total_latency / message_count), " ns");
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        log_error("Usage: ", argv[0], " [producer|consumer]");
        return 1;
    }

    std::string mode = argv[1];

    try {
        if (mode == "producer") {
            run_producer();
        } else if (mode == "consumer") {
            run_consumer();
        } else {
            log_error("Invalid mode. Use 'producer' or 'consumer'");
            return 1;
        }
    } catch (const std::exception &e) {
        log_error("Error: ", e.what());
        return 1;
    }

    return 0;
}