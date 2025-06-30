#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

using namespace psyne;

int main() {
    std::cout << "TCP Client Example\n";
    std::cout << "==================\n\n";

    try {
        // Create client channel - connect to localhost:9999
        auto channel =
            create_channel("tcp://localhost:9999",
                           1024 * 1024, // 1MB buffer
                           ChannelMode::SPSC, ChannelType::SingleType);

        std::cout << "Connected to server at localhost:9999\n";

        // Start receiver thread
        std::atomic<bool> running(true);
        std::thread receiver([&channel, &running]() {
            while (running) {
                auto msg = channel->receive<FloatVector>(
                    std::chrono::milliseconds(500));
                if (msg) {
                    std::cout << "\nReceived reply: ";
                    for (float val : *msg) {
                        std::cout << val << " ";
                    }
                    std::cout << std::endl;
                }
            }
        });

        // Send messages
        for (int i = 0; i < 10; ++i) {
            FloatVector msg(*channel);
            msg.resize(5);

            // Fill with test data
            for (size_t j = 0; j < 5; ++j) {
                msg[j] = static_cast<float>(i * 10 + j);
            }

            std::cout << "\nSending message " << i << ": ";
            for (float val : msg) {
                std::cout << val << " ";
            }
            std::cout << std::endl;

            channel->send(msg);

            // Wait a bit between messages
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        // Wait for final replies
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Stop receiver
        running = false;
        receiver.join();

        std::cout << "\nClient done.\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}