#include <atomic>
#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>
#include <vector>

using namespace psyne;

void test_multicast_basic() {
    std::cout << "=== Basic UDP Multicast Demo ===" << std::endl;

    const std::string multicast_addr = "239.255.0.1";
    const uint16_t port = 12345;

    try {
        // Create publisher
        std::cout << "Creating multicast publisher..." << std::endl;
        auto publisher = multicast::create_publisher(multicast_addr, port);

        // Create subscriber
        std::cout << "Creating multicast subscriber..." << std::endl;
        auto subscriber = multicast::create_subscriber(multicast_addr, port);

        // Give subscriber time to join the group
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::cout << "Publishing messages..." << std::endl;

        // Send test messages
        for (int i = 0; i < 10; ++i) {
            FloatVector msg(*publisher);
            msg.resize(5);
            for (size_t j = 0; j < msg.size(); ++j) {
                msg[j] = static_cast<float>(i * 10 + j);
            }

            size_t msg_size = msg.size();
            publisher->send(msg);
            std::cout << "Sent message " << (i + 1) << " with " << msg_size
                      << " floats" << std::endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "\nReceiving messages..." << std::endl;

        // Receive messages
        int received = 0;
        auto start_time = std::chrono::steady_clock::now();

        while (received < 10) {
            auto msg = subscriber->receive<FloatVector>();
            if (msg) {
                received++;
                std::cout << "Received message " << received << " with "
                          << msg->size() << " floats: ";
                for (size_t i = 0; i < msg->size() && i < 3; ++i) {
                    std::cout << (*msg)[i] << " ";
                }
                if (msg->size() > 3)
                    std::cout << "...";
                std::cout << std::endl;
            }

            // Timeout after 5 seconds
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed > std::chrono::seconds(5)) {
                std::cout << "Timeout waiting for messages" << std::endl;
                break;
            }
        }

        std::cout << "Successfully received " << received
                  << " out of 10 messages" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error in basic multicast test: " << e.what() << std::endl;
    }
}

void test_multicast_with_compression() {
    std::cout << "\n=== UDP Multicast with Compression Demo ===" << std::endl;

    const std::string multicast_addr = "239.255.0.2";
    const uint16_t port = 12346;

    try {
        // Configure compression
        compression::CompressionConfig comp_config;
        comp_config.type = compression::CompressionType::LZ4;
        comp_config.min_size_threshold = 100; // Compress messages > 100 bytes

        // Create publisher with compression
        std::cout << "Creating compressed multicast publisher..." << std::endl;
        auto publisher = multicast::create_publisher(multicast_addr, port,
                                                     1024 * 1024, comp_config);

        // Create subscriber
        std::cout << "Creating multicast subscriber..." << std::endl;
        auto subscriber = multicast::create_subscriber(multicast_addr, port);

        // Wait for setup
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::cout << "Publishing large messages with compression..."
                  << std::endl;

        // Send larger messages that will benefit from compression
        for (int i = 0; i < 5; ++i) {
            FloatVector msg(*publisher);
            msg.resize(100); // 400 bytes, should trigger compression

            // Fill with repeating pattern (good for compression)
            for (size_t j = 0; j < msg.size(); ++j) {
                msg[j] = static_cast<float>(j % 10) * 0.1f; // Repeating pattern
            }

            size_t msg_size = msg.size();
            publisher->send(msg);
            std::cout << "Sent compressed message " << (i + 1) << " with "
                      << msg_size << " floats" << std::endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        std::cout << "\nReceiving compressed messages..." << std::endl;

        // Receive messages
        int received = 0;
        auto start_time = std::chrono::steady_clock::now();

        while (received < 5) {
            auto msg = subscriber->receive<FloatVector>();
            if (msg) {
                received++;
                std::cout << "Received compressed message " << received
                          << " with " << msg->size() << " floats" << std::endl;

                // Verify pattern
                bool pattern_ok = true;
                for (size_t i = 0; i < msg->size() && i < 10; ++i) {
                    float expected = static_cast<float>(i % 10) * 0.1f;
                    if (std::abs((*msg)[i] - expected) > 1e-6f) {
                        pattern_ok = false;
                        break;
                    }
                }

                if (pattern_ok) {
                    std::cout << "  ✓ Data integrity verified" << std::endl;
                } else {
                    std::cout << "  ✗ Data corruption detected!" << std::endl;
                }
            }

            // Timeout after 5 seconds
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed > std::chrono::seconds(5)) {
                std::cout << "Timeout waiting for compressed messages"
                          << std::endl;
                break;
            }
        }

        std::cout << "Successfully received " << received
                  << " out of 5 compressed messages" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error in compression multicast test: " << e.what()
                  << std::endl;
    }
}

void test_multiple_subscribers() {
    std::cout << "\n=== Multiple Subscribers Demo ===" << std::endl;

    const std::string multicast_addr = "239.255.0.3";
    const uint16_t port = 12347;

    try {
        // Create publisher
        std::cout << "Creating multicast publisher..." << std::endl;
        auto publisher = multicast::create_publisher(multicast_addr, port);

        // Create multiple subscribers
        std::cout << "Creating 3 multicast subscribers..." << std::endl;
        std::vector<std::unique_ptr<Channel>> subscribers;
        std::vector<std::atomic<int>> receive_counts(3);

        for (int i = 0; i < 3; ++i) {
            subscribers.push_back(
                multicast::create_subscriber(multicast_addr, port));
            receive_counts[i] = 0;
        }

        // Wait for all subscribers to join
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Start receiver threads
        std::vector<std::thread> receiver_threads;
        std::atomic<bool> stop_receiving{false};

        for (int i = 0; i < 3; ++i) {
            receiver_threads.emplace_back([&, i]() {
                while (!stop_receiving) {
                    auto msg = subscribers[i]->receive<FloatVector>();
                    if (msg) {
                        receive_counts[i]++;
                        std::cout << "Subscriber " << (i + 1)
                                  << " received message " << receive_counts[i]
                                  << " (size: " << msg->size() << ")"
                                  << std::endl;
                    }
                }
            });
        }

        std::cout << "Publishing to multiple subscribers..." << std::endl;

        // Send messages
        for (int i = 0; i < 8; ++i) {
            FloatVector msg(*publisher);
            msg.resize(10);
            for (size_t j = 0; j < msg.size(); ++j) {
                msg[j] = static_cast<float>(i * 100 + j);
            }

            publisher->send(msg);
            std::cout << "Published message " << (i + 1) << std::endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }

        // Wait for all messages to be received
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Stop receivers
        stop_receiving = true;
        for (auto &thread : receiver_threads) {
            thread.join();
        }

        // Show final counts
        std::cout << "\nFinal receive counts:" << std::endl;
        for (int i = 0; i < 3; ++i) {
            std::cout << "  Subscriber " << (i + 1) << ": " << receive_counts[i]
                      << " messages" << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "Error in multiple subscribers test: " << e.what()
                  << std::endl;
    }
}

int main() {
    try {
        std::cout << "Psyne UDP Multicast Demo" << std::endl;
        std::cout << "========================" << std::endl;
        std::cout
            << "Note: This demo uses local multicast addresses (239.255.x.x)"
            << std::endl;
        std::cout << "Make sure your system supports multicast networking."
                  << std::endl
                  << std::endl;

        test_multicast_basic();
        test_multicast_with_compression();
        test_multiple_subscribers();

        std::cout << "\n✅ UDP Multicast demos completed!" << std::endl;
        std::cout << "\nMulticast features demonstrated:" << std::endl;
        std::cout << "  • One-to-many broadcasting" << std::endl;
        std::cout << "  • Message compression support" << std::endl;
        std::cout << "  • Multiple subscriber support" << std::endl;
        std::cout << "  • Automatic multicast group management" << std::endl;
        std::cout << "  • Sequence numbering and validation" << std::endl;
        std::cout << "  • High-performance UDP transport" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}