/**
 * @file tcp_example.cpp
 * @brief Example of TCP network communication using Psyne channels
 *
 * This example demonstrates zero-copy TCP communication between processes
 * potentially on different machines. Run with "server" or "client" argument.
 */

#include "psyne/psyne.hpp"
#include "logger.hpp"
#include <chrono>
#include <cstring>
#include <thread>

using namespace psyne;

/**
 * @brief Test message for network transfer
 */
struct NetworkMessage {
    uint64_t sequence;
    uint64_t timestamp;
    double values[100]; // Some data to transfer
    char description[128];
};

void run_server() {
    log_info("Starting TCP server on port 9999...");

    // Create channel configuration for TCP server
    ChannelConfig config;
    config.name = "tcp_server_channel";
    config.size_mb = 8;
    config.mode = ChannelMode::SPSC;
    config.transport = ChannelTransport::TCP;
    config.is_server = true;
    config.remote_port = 9999;
    config.blocking = true;

    // Create the channel
    auto channel = Channel<NetworkMessage>::create(config);

    // TCP channel will automatically wait for connection
    log_info("Waiting for client connection...");
    // First receive will block until connected
    log_info("Ready to receive messages...");

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

        // Calculate network latency
        uint64_t now = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
        uint64_t latency = now - msg->timestamp;
        total_latency += latency;
        message_count++;

        // Verify data integrity
        double expected_sum = 0;
        for (int i = 0; i < 100; ++i) {
            expected_sum += i * msg->sequence;
        }

        double actual_sum = 0;
        for (int i = 0; i < 100; ++i) {
            actual_sum += msg->values[i];
        }

        if (std::abs(actual_sum - expected_sum) > 0.001) {
            log_error("ERROR: Data corruption detected! Expected sum ",
                      expected_sum, " but got ", actual_sum);
        }

        if (expected_sequence % 100 == 0) {
            log_info("Received ", expected_sequence,
                      " messages, desc: ", msg->description,
                      ", latency: ", (latency / 1000), " us");
        }

        expected_sequence++;
    }

    // Print statistics
    auto stats = channel->get_stats();
    log_info("Server Statistics:");
    log_info("  Messages received: ", stats.messages_received);
    log_info("  Bytes received: ", stats.bytes_received);
    log_info("  Receive failures: ", stats.receive_failures);
    log_info("  Average channel latency: ", stats.avg_latency_ns, " ns");
    log_info("  Average network latency: ",
              (total_latency / message_count / 1000), " us");
}

void run_client(const std::string &server_host) {
    log_info("Starting TCP client connecting to ", server_host, ":9999...");

    // Create channel configuration for TCP client
    ChannelConfig config;
    config.name = "tcp_client_channel";
    config.size_mb = 8;
    config.mode = ChannelMode::SPSC;
    config.transport = ChannelTransport::TCP;
    config.is_server = false;
    config.remote_host = server_host;
    config.remote_port = 9999;
    config.blocking = true;

    // Create the channel
    auto channel = Channel<NetworkMessage>::create(config);

    // TCP channel will automatically connect
    log_info("Connecting to server...");
    // First allocate will block until connected
    log_info("Ready to send messages...");

    // Send messages
    for (uint64_t i = 0; i < 1000; ++i) {
        auto msg = channel->allocate();

        msg->sequence = i;
        msg->timestamp = std::chrono::high_resolution_clock::now()
                             .time_since_epoch()
                             .count();

        // Fill data with predictable pattern
        for (int j = 0; j < 100; ++j) {
            msg->values[j] = j * i;
        }

        snprintf(msg->description, sizeof(msg->description),
                 "Message %lu from client", i);

        msg.send();

        if (i % 100 == 0) {
            log_info("Sent ", i, " messages");
        }

        // Small delay to simulate work and avoid overwhelming network
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // Give time for last messages to be received
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Print statistics
    auto stats = channel->get_stats();
    log_info("Client Statistics:");
    log_info("  Messages sent: ", stats.messages_sent);
    log_info("  Bytes sent: ", stats.bytes_sent);
    log_info("  Send failures: ", stats.send_failures);
    log_info("  Average latency: ", stats.avg_latency_ns, " ns");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        log_error("Usage: ", argv[0], " [server|client] [server_host]");
        log_error("  server - Start as server listening on port 9999");
        log_error("  client [host] - Start as client connecting to host ",
                     "(default: localhost)");
        return 1;
    }

    std::string mode = argv[1];

    try {
        if (mode == "server") {
            run_server();
        } else if (mode == "client") {
            std::string host = "localhost";
            if (argc > 2) {
                host = argv[2];
            }
            run_client(host);
        } else {
            log_error("Invalid mode. Use 'server' or 'client'");
            return 1;
        }
    } catch (const std::exception &e) {
        log_error("Error: ", e.what());
        return 1;
    }

    return 0;
}