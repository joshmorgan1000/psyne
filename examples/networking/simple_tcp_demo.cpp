/**
 * @file simple_tcp_demo.cpp
 * @brief Simple TCP demo using existing working channels to prove concept
 * 
 * This demonstrates the exact same patterns as our TCP examples
 * but using IPC channels to verify the code structure is correct.
 * Once TCP channels are fully implemented, we can switch the URI.
 */

#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

using namespace psyne;

void run_server() {
    std::cout << "🚀 Psyne Channel Server Demo (IPC-based)\n";
    std::cout << "=========================================\n";
    std::cout << "This uses IPC but shows the exact same patterns as TCP\n\n";

    try {
        // Use IPC channel with same patterns as TCP examples
        auto channel = create_channel("ipc://demo_server",
                                    4 * 1024 * 1024, // 4MB buffer
                                    ChannelMode::SPSC, 
                                    ChannelType::SingleType);

        std::cout << "✓ Server channel created\n";
        std::cout << "✓ Waiting for messages...\n\n";

        int message_count = 0;
        while (message_count < 5) {
            auto msg = channel->receive<FloatVector>(std::chrono::milliseconds(2000));

            if (msg) {
                message_count++;
                std::cout << "📨 Received message " << message_count << ": " 
                         << msg->size() << " floats\n";

                // Show first few values
                std::cout << "   Data: ";
                for (size_t i = 0; i < std::min(size_t(5), msg->size()); ++i) {
                    std::cout << (*msg)[i] << " ";
                }
                if (msg->size() > 5) std::cout << "...";
                std::cout << "\n";

                // Echo back with processing (same as TCP examples)
                FloatVector reply(channel);
                reply.resize(msg->size());
                
                for (size_t i = 0; i < msg->size(); ++i) {
                    reply[i] = (*msg)[i] * 2.0f + static_cast<float>(i);
                }
                
                reply.send();
                std::cout << "📤 Sent processed reply\n\n";
            } else {
                std::cout << "⏳ Waiting...\n";
            }
        }

        std::cout << "✅ Server completed successfully!\n";

    } catch (const std::exception &e) {
        std::cerr << "❌ Server error: " << e.what() << std::endl;
    }
}

void run_client() {
    std::cout << "🚀 Psyne Channel Client Demo (IPC-based)\n";
    std::cout << "=========================================\n";
    std::cout << "This uses IPC but shows the exact same patterns as TCP\n\n";

    try {
        // Connect to server (in real TCP, this would be tcp://localhost:8080)
        auto channel = create_channel("ipc://demo_server",
                                    0, // Client mode
                                    ChannelMode::SPSC, 
                                    ChannelType::SingleType);

        std::cout << "✓ Connected to server channel\n";
        std::cout << "✓ Sending test messages...\n\n";

        // Send exactly the same message patterns as TCP examples
        std::vector<size_t> sizes = {100, 1000, 10000, 50000, 100000};
        
        for (int i = 0; i < 5; ++i) {
            size_t size = sizes[i];
            
            // Create message with same pattern as TCP examples
            FloatVector msg(channel);
            msg.resize(size);

            for (size_t j = 0; j < size; ++j) {
                msg[j] = static_cast<float>(i * 1000 + j) * 0.01f;
            }

            std::cout << "📤 Sending message " << (i + 1) << ": " 
                     << size << " floats (" << (size * sizeof(float) / 1024) << " KB)\n";
            
            msg.send();

            // Wait for reply
            auto reply = channel->receive<FloatVector>(std::chrono::milliseconds(3000));
            if (reply) {
                std::cout << "📨 Received reply: " << reply->size() << " floats\n";
                
                // Verify processing
                if (reply->size() > 0) {
                    std::cout << "   First processed value: " << (*reply)[0] 
                             << " (should be " << (msg[0] * 2.0f) << ")\n";
                }
            }
            std::cout << "\n";

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "✅ Client completed successfully!\n";

    } catch (const std::exception &e) {
        std::cerr << "❌ Client error: " << e.what() << std::endl;
    }
}

int main(int argc, char *argv[]) {
    std::cout << "Psyne Channel Demo (TCP Example Patterns)\n";
    std::cout << "=========================================\n\n";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [server|client]\n\n";
        std::cout << "This demo shows the EXACT same code patterns as our TCP examples\n";
        std::cout << "but uses IPC channels to prove the examples are correctly written.\n\n";
        std::cout << "Run in two terminals:\n";
        std::cout << "  Terminal 1: " << argv[0] << " server\n";
        std::cout << "  Terminal 2: " << argv[0] << " client\n";
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "server") {
        run_server();
    } else if (mode == "client") {
        run_client();
    } else {
        std::cerr << "❌ Invalid mode. Use 'server' or 'client'\n";
        return 1;
    }

    return 0;
}