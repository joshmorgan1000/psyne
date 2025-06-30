#include <atomic>
#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <string>
#include <thread>

using namespace psyne;

void run_server() {
    std::cout << "=== TCP Server Mode ===\n";
    std::cout << "Starting server on port 9999...\n\n";

    try {
        // Create server channel - empty host means listen on all interfaces
        auto channel =
            create_channel("tcp://:9999",
                           1024 * 1024, // 1MB buffer
                           ChannelMode::SPSC, ChannelType::SingleType);

        std::cout << "Server listening on port 9999...\n";
        std::cout << "Waiting for messages (press Ctrl+C to stop)...\n\n";

        // Process messages and echo them back
        int count = 0;
        while (true) {
            auto msg =
                channel->receive<FloatVector>(std::chrono::milliseconds(1000));

            if (msg) {
                std::cout << "Received message " << count++ << ": ";
                for (float val : *msg) {
                    std::cout << val << " ";
                }
                std::cout << "\n";

                // Echo the message back to client
                FloatVector reply(*channel);
                reply.resize(msg->size());
                for (size_t i = 0; i < msg->size(); ++i) {
                    reply[i] = (*msg)[i] * 2.0f; // Double the values
                }
                reply.send();
                std::cout << "Echoed back (doubled values)\n\n";
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
}

void run_client() {
    std::cout << "=== TCP Client Mode ===\n";
    std::cout << "Connecting to server at localhost:9999...\n\n";

    try {
        // Create client channel - connect to localhost:9999
        auto channel =
            create_channel("tcp://localhost:9999",
                           1024 * 1024, // 1MB buffer
                           ChannelMode::SPSC, ChannelType::SingleType);

        std::cout << "Connected to server!\n";

        // Start receiver thread for replies
        std::atomic<bool> running(true);
        std::thread receiver([&channel, &running]() {
            while (running) {
                auto msg = channel->receive<FloatVector>(
                    std::chrono::milliseconds(500));
                if (msg) {
                    std::cout << "Received reply: ";
                    for (float val : *msg) {
                        std::cout << val << " ";
                    }
                    std::cout << "\n\n";
                }
            }
        });

        // Send test messages
        for (int i = 0; i < 5; ++i) {
            FloatVector msg(*channel);
            msg.resize(4);

            // Fill with test data
            for (size_t j = 0; j < 4; ++j) {
                msg[j] = static_cast<float>(i * 10 + j);
            }

            std::cout << "Sending: ";
            for (float val : msg) {
                std::cout << val << " ";
            }
            std::cout << "\n";

            msg.send();

            // Wait between messages
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        std::cout << "Client finished sending messages.\n";

        // Wait a bit for final replies
        std::this_thread::sleep_for(std::chrono::seconds(2));
        running = false;
        receiver.join();

    } catch (const std::exception &e) {
        std::cerr << "Client error: " << e.what() << std::endl;
    }
}

int main(int argc, char *argv[]) {
    std::cout << "Psyne TCP Demo\n";
    std::cout << "==============\n\n";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [server|client]\n";
        std::cout << "\nExamples:\n";
        std::cout << "  " << argv[0] << " server    # Run as TCP server\n";
        std::cout << "  " << argv[0] << " client    # Run as TCP client\n";
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "server") {
        run_server();
    } else if (mode == "client") {
        run_client();
    } else {
        std::cerr << "Invalid mode. Use 'server' or 'client'\n";
        return 1;
    }

    return 0;
}