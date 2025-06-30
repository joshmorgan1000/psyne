#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

using namespace psyne;

int main() {
    std::cout << "IPC Consumer Example\n";
    std::cout << "===================\n\n";

    try {
        // Open existing IPC channel
        auto channel =
            create_channel("ipc://demo_channel",
                           1024 * 1024, // Must match producer
                           ChannelMode::SPSC, ChannelType::SingleType);

        std::cout << "Connected to IPC channel: " << channel->uri() << "\n";
        std::cout << "Waiting for messages...\n\n";

        // Receive messages
        int count = 0;
        while (count < 10) {
            auto msg =
                channel->receive<FloatVector>(std::chrono::milliseconds(1000));

            if (msg) {
                std::cout << "Received message: ";
                for (float val : *msg) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
                count++;
            } else {
                std::cout << "." << std::flush;
            }
        }

        std::cout << "\nReceived " << count << " messages\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}