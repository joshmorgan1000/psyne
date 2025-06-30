#include <atomic>
#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <string>
#include <thread>
#include <vector>

using namespace psyne;

void run_producer() {
    std::cout << "=== IPC Producer Mode ===\n";
    std::cout << "Starting producer...\n\n";

    try {
        // Create an IPC channel - this will create shared memory
        auto channel =
            create_channel("ipc://demo_channel",
                           1024 * 1024, // 1MB buffer
                           ChannelMode::SPSC, ChannelType::MultiType);

        std::cout << "Created IPC channel: " << channel->uri() << "\n";
        std::cout << "Sending various message types...\n\n";

        // Send different types of messages
        for (int i = 0; i < 10; ++i) {
            if (i % 3 == 0) {
                // Send FloatVector
                FloatVector msg(*channel);
                msg.resize(5);
                for (size_t j = 0; j < 5; ++j) {
                    msg[j] = static_cast<float>(i * 10 + j);
                }

                std::cout << "Sending FloatVector " << i << ": ";
                for (float val : msg) {
                    std::cout << val << " ";
                }
                std::cout << "\n";
                msg.send();

            } else if (i % 3 == 1) {
                // Send ByteVector
                ByteVector msg(*channel);
                msg.resize(8);
                for (size_t j = 0; j < 8; ++j) {
                    msg[j] = static_cast<uint8_t>(i * 20 + j);
                }

                std::cout << "Sending ByteVector " << i << ": ";
                for (uint8_t val : msg) {
                    std::cout << static_cast<int>(val) << " ";
                }
                std::cout << "\n";
                msg.send();

            } else {
                // Send DoubleMatrix
                DoubleMatrix msg(*channel);
                msg.set_dimensions(2, 2);
                msg.at(0, 0) = i * 1.1;
                msg.at(0, 1) = i * 1.2;
                msg.at(1, 0) = i * 1.3;
                msg.at(1, 1) = i * 1.4;

                std::cout << "Sending DoubleMatrix " << i << ":\n";
                std::cout << "  [" << msg.at(0, 0) << ", " << msg.at(0, 1)
                          << "]\n";
                std::cout << "  [" << msg.at(1, 0) << ", " << msg.at(1, 1)
                          << "]\n";
                msg.send();
            }

            // Wait between messages
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        std::cout << "\nProducer finished sending messages.\n";

    } catch (const std::exception &e) {
        std::cerr << "Producer error: " << e.what() << std::endl;
    }
}

void run_consumer() {
    std::cout << "=== IPC Consumer Mode ===\n";
    std::cout << "Starting consumer...\n\n";

    try {
        // Connect to existing IPC channel
        auto channel =
            create_channel("ipc://demo_channel",
                           1024 * 1024, // 1MB buffer
                           ChannelMode::SPSC, ChannelType::MultiType);

        std::cout << "Connected to IPC channel: " << channel->uri() << "\n";
        std::cout << "Waiting for messages (timeout after 5 seconds of "
                     "inactivity)...\n\n";

        int received_count = 0;
        auto last_message_time = std::chrono::steady_clock::now();

        while (true) {
            // Try to receive different message types
            bool received_any = false;

            // Try FloatVector
            auto float_msg = channel->receive_single<FloatVector>(
                std::chrono::milliseconds(10));
            if (float_msg) {
                std::cout << "Received FloatVector: ";
                for (float val : *float_msg) {
                    std::cout << val << " ";
                }
                std::cout << "\n";
                received_any = true;
                received_count++;
            }

            // Try ByteVector
            auto byte_msg = channel->receive_single<ByteVector>(
                std::chrono::milliseconds(10));
            if (byte_msg) {
                std::cout << "Received ByteVector: ";
                for (uint8_t val : *byte_msg) {
                    std::cout << static_cast<int>(val) << " ";
                }
                std::cout << "\n";
                received_any = true;
                received_count++;
            }

            // Try DoubleMatrix
            auto matrix_msg = channel->receive_single<DoubleMatrix>(
                std::chrono::milliseconds(10));
            if (matrix_msg) {
                std::cout << "Received DoubleMatrix:\n";
                for (size_t i = 0; i < matrix_msg->rows(); ++i) {
                    std::cout << "  [";
                    for (size_t j = 0; j < matrix_msg->cols(); ++j) {
                        std::cout << matrix_msg->at(i, j);
                        if (j < matrix_msg->cols() - 1)
                            std::cout << ", ";
                    }
                    std::cout << "]\n";
                }
                received_any = true;
                received_count++;
            }

            if (received_any) {
                last_message_time = std::chrono::steady_clock::now();
            }

            // Check for timeout
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - last_message_time);
            if (elapsed.count() > 5) {
                std::cout << "\nTimeout reached. No messages received for 5 "
                             "seconds.\n";
                break;
            }

            // Small delay to prevent busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::cout << "Consumer finished. Total messages received: "
                  << received_count << "\n";

    } catch (const std::exception &e) {
        std::cerr << "Consumer error: " << e.what() << std::endl;
    }
}

void run_test() {
    std::cout << "=== IPC Test Mode ===\n";
    std::cout << "Running combined producer/consumer test...\n\n";

    try {
        auto channel =
            create_channel("ipc://test_channel", 1024 * 1024, ChannelMode::SPSC,
                           ChannelType::SingleType);

        std::atomic<int> messages_sent{0};
        std::atomic<int> messages_received{0};
        std::atomic<bool> producer_done{false};

        // Producer thread
        std::thread producer([&]() {
            for (int i = 0; i < 20; ++i) {
                FloatVector msg(*channel);
                msg.resize(4);
                for (size_t j = 0; j < 4; ++j) {
                    msg[j] = static_cast<float>(i * 100 + j);
                }
                msg.send();
                messages_sent++;

                if (i % 5 == 0) {
                    std::cout << "Producer sent " << (i + 1) << " messages\n";
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            producer_done = true;
            std::cout << "Producer finished. Sent " << messages_sent.load()
                      << " messages\n";
        });

        // Consumer thread
        std::thread consumer([&]() {
            while (!producer_done || messages_received < messages_sent) {
                auto msg = channel->receive_single<FloatVector>(
                    std::chrono::milliseconds(100));
                if (msg) {
                    messages_received++;
                    if (messages_received % 5 == 0) {
                        std::cout << "Consumer received "
                                  << messages_received.load() << " messages\n";
                    }
                }
            }
            std::cout << "Consumer finished. Received "
                      << messages_received.load() << " messages\n";
        });

        producer.join();
        consumer.join();

        std::cout << "\nTest Results:\n";
        std::cout << "  Messages sent: " << messages_sent.load() << "\n";
        std::cout << "  Messages received: " << messages_received.load()
                  << "\n";
        std::cout << "  Success rate: "
                  << (100.0 * messages_received.load() / messages_sent.load())
                  << "%\n";

    } catch (const std::exception &e) {
        std::cerr << "Test error: " << e.what() << std::endl;
    }
}

int main(int argc, char *argv[]) {
    std::cout << "Psyne IPC Demo\n";
    std::cout << "==============\n\n";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [producer|consumer|test]\n";
        std::cout << "\nModes:\n";
        std::cout << "  producer  # Send messages via IPC\n";
        std::cout << "  consumer  # Receive messages via IPC\n";
        std::cout << "  test      # Run combined producer/consumer test\n";
        std::cout << "\nFor producer/consumer mode:\n";
        std::cout << "  1. Run producer in one terminal\n";
        std::cout << "  2. Run consumer in another terminal\n";
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "producer") {
        run_producer();
    } else if (mode == "consumer") {
        run_consumer();
    } else if (mode == "test") {
        run_test();
    } else {
        std::cerr << "Invalid mode. Use 'producer', 'consumer', or 'test'\n";
        return 1;
    }

    return 0;
}