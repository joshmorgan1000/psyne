#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

using namespace psyne;

int main() {
    try {
        // Create a memory channel with metrics enabled
        auto channel =
            create_channel("memory://metrics_test", 1024 * 1024,
                           ChannelMode::SPSC, ChannelType::SingleType, true);

        std::cout << "Created channel with metrics enabled" << std::endl;

        // Check if metrics are available
        if (channel->has_metrics()) {
            std::cout << "Channel has metrics support" << std::endl;
        }

        std::cout << "Sending messages..." << std::endl;

        // Send some messages
        for (int i = 0; i < 100; ++i) {
            FloatVector msg(*channel);
            msg.resize(10);
            for (size_t j = 0; j < msg.size(); ++j) {
                msg[j] = static_cast<float>(i * 10 + j);
            }
            channel->send(msg);
        }

        std::cout << "Receiving messages..." << std::endl;

        // Receive messages
        int received_count = 0;
        while (received_count < 100) {
            auto msg = channel->receive<FloatVector>();
            if (msg) {
                received_count++;
            }
        }

        std::cout << "Messages processed successfully" << std::endl;

        // Check metrics
        if (channel->has_metrics()) {
            auto metrics = channel->get_metrics();
            std::cout << "\nChannel metrics:" << std::endl;
            std::cout << "  Messages sent: " << metrics.messages_sent
                      << std::endl;
            std::cout << "  Bytes sent: " << metrics.bytes_sent << std::endl;
            std::cout << "  Messages received: " << metrics.messages_received
                      << std::endl;
            std::cout << "  Bytes received: " << metrics.bytes_received
                      << std::endl;
            std::cout << "  Send blocks: " << metrics.send_blocks << std::endl;
            std::cout << "  Receive blocks: " << metrics.receive_blocks
                      << std::endl;
        }

        // Test reset functionality
        std::cout << "\nTesting metrics reset..." << std::endl;
        channel->reset_metrics();

        if (channel->has_metrics()) {
            auto reset_metrics = channel->get_metrics();
            std::cout << "After reset:" << std::endl;
            std::cout << "  Messages sent: " << reset_metrics.messages_sent
                      << std::endl;
            std::cout << "  Messages received: "
                      << reset_metrics.messages_received << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}