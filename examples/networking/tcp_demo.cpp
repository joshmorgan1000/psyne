/**
 * @file tcp_demo.cpp
 * @brief Comprehensive TCP networking demo showcasing zero-copy messaging
 * 
 * This example demonstrates:
 * - TCP server and client modes
 * - Zero-copy FloatVector messaging
 * - Bidirectional communication
 * - Error handling and connection management
 * - Performance metrics
 */

#include <atomic>
#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <string>
#include <thread>
#include <vector>

using namespace psyne;

void run_server() {
    std::cout << "=== Psyne TCP Server ===\n";
    std::cout << "Starting high-performance server on port 8080...\n\n";

    try {
        // Create server channel - empty host means listen on all interfaces
        auto channel = Channel::create("tcp://:8080",
                                     4 * 1024 * 1024, // 4MB buffer for high throughput
                                     ChannelMode::SPSC, 
                                     ChannelType::SingleType);

        std::cout << "✓ Server listening on port 8080\n";
        std::cout << "✓ Zero-copy ring buffer: 4MB capacity\n";
        std::cout << "✓ Waiting for client connections...\n\n";

        // Statistics tracking
        int messages_received = 0;
        size_t total_bytes = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (messages_received < 20) {  // Process 20 messages then exit
            auto msg = channel->receive<FloatVector>(std::chrono::milliseconds(2000));

            if (msg) {
                messages_received++;
                total_bytes += msg->size() * sizeof(float);
                
                std::cout << "📨 Message " << messages_received << ": " 
                         << msg->size() << " floats (";
                
                // Show first few values
                for (size_t i = 0; i < std::min(size_t(5), msg->size()); ++i) {
                    std::cout << (*msg)[i];
                    if (i < std::min(size_t(4), msg->size() - 1)) std::cout << ", ";
                }
                if (msg->size() > 5) std::cout << "...";
                std::cout << ")\n";

                // Echo back with processing (zero-copy)
                FloatVector reply(channel);
                reply.resize(msg->size());
                
                // Apply some processing - multiply by 2 and add index
                for (size_t i = 0; i < msg->size(); ++i) {
                    reply[i] = (*msg)[i] * 2.0f + static_cast<float>(i);
                }
                
                reply.send();
                std::cout << "📤 Echoed processed data back to client\n\n";
            } else {
                std::cout << "⏳ Waiting for messages...\n";
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\n🎯 Server Statistics:\n";
        std::cout << "   Messages processed: " << messages_received << "\n";
        std::cout << "   Total data: " << total_bytes << " bytes\n";
        std::cout << "   Duration: " << duration.count() << " ms\n";
        std::cout << "   Throughput: " << (total_bytes * 1000.0 / duration.count() / 1024 / 1024) 
                 << " MB/s\n";

    } catch (const std::exception &e) {
        std::cerr << "❌ Server error: " << e.what() << std::endl;
    }
}

void run_client() {
    std::cout << "=== Psyne TCP Client ===\n";
    std::cout << "Connecting to server at localhost:8080...\n\n";

    try {
        // Create client channel - connect to localhost:8080
        auto channel = Channel::create("tcp://localhost:8080",
                                     4 * 1024 * 1024, // 4MB buffer
                                     ChannelMode::SPSC, 
                                     ChannelType::SingleType);

        std::cout << "✓ Connected to server!\n";
        std::cout << "✓ Zero-copy ring buffer: 4MB capacity\n\n";

        // Start receiver thread for server replies
        std::atomic<bool> running(true);
        std::atomic<int> replies_received(0);
        
        std::thread receiver([&]() {
            while (running && replies_received < 10) {
                auto reply = channel->receive<FloatVector>(std::chrono::milliseconds(1000));
                if (reply) {
                    replies_received++;
                    std::cout << "📨 Reply " << replies_received.load() << ": " 
                             << reply->size() << " floats (";
                    
                    // Show first few processed values
                    for (size_t i = 0; i < std::min(size_t(5), reply->size()); ++i) {
                        std::cout << (*reply)[i];
                        if (i < std::min(size_t(4), reply->size() - 1)) std::cout << ", ";
                    }
                    if (reply->size() > 5) std::cout << "...";
                    std::cout << ")\n";
                }
            }
        });

        // Send test messages with varying sizes
        std::vector<size_t> message_sizes = {10, 100, 1000, 10000, 100000};
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int round = 0; round < 2; ++round) {
            for (size_t size : message_sizes) {
                // Create zero-copy message directly in ring buffer
                FloatVector msg(channel);
                msg.resize(size);

                // Fill with test data pattern
                for (size_t i = 0; i < size; ++i) {
                    msg[i] = static_cast<float>(round * 1000 + i) * 0.1f;
                }

                std::cout << "📤 Sending: " << size << " floats (round " 
                         << (round + 1) << ")\n";
                
                msg.send();  // Zero-copy network transmission

                // Small delay between messages
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        std::cout << "\n⏳ Waiting for remaining replies...\n";
        
        // Wait for all replies
        while (replies_received < 10 && running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        running = false;
        receiver.join();

        std::cout << "\n🎯 Client Statistics:\n";
        std::cout << "   Messages sent: 10\n";
        std::cout << "   Replies received: " << replies_received.load() << "\n";
        std::cout << "   Total duration: " << duration.count() << " ms\n";
        std::cout << "   Average round-trip: " << (duration.count() / 10.0) << " ms\n";

    } catch (const std::exception &e) {
        std::cerr << "❌ Client error: " << e.what() << std::endl;
    }
}

void run_performance_test() {
    std::cout << "=== Psyne TCP Performance Test ===\n";
    std::cout << "High-throughput zero-copy messaging benchmark\n\n";

    try {
        auto channel = Channel::create("tcp://localhost:8080",
                                     16 * 1024 * 1024, // 16MB buffer for maximum throughput
                                     ChannelMode::SPSC, 
                                     ChannelType::SingleType);

        std::cout << "✓ Connected for performance testing\n";
        std::cout << "✓ Using 16MB zero-copy ring buffer\n\n";

        const size_t message_size = 100000;  // 100K floats = ~400KB per message
        const int num_messages = 50;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_messages; ++i) {
            FloatVector msg(channel);
            msg.resize(message_size);
            
            // Fill with computational pattern
            for (size_t j = 0; j < message_size; ++j) {
                msg[j] = std::sin(static_cast<float>(i * message_size + j) * 0.001f);
            }
            
            msg.send();
            
            if (i % 10 == 0) {
                std::cout << "📤 Sent " << (i + 1) << " / " << num_messages 
                         << " messages (" << (message_size * sizeof(float) / 1024) 
                         << " KB each)\n";
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        size_t total_bytes = num_messages * message_size * sizeof(float);
        double throughput_mbps = (total_bytes * 1000000.0) / (duration.count() * 1024 * 1024);
        
        std::cout << "\n🚀 Performance Results:\n";
        std::cout << "   Messages: " << num_messages << " x " << message_size << " floats\n";
        std::cout << "   Total data: " << (total_bytes / 1024 / 1024) << " MB\n";
        std::cout << "   Duration: " << duration.count() << " μs\n";
        std::cout << "   Throughput: " << throughput_mbps << " MB/s\n";
        std::cout << "   Message rate: " << (num_messages * 1000000.0 / duration.count()) << " msg/s\n";

    } catch (const std::exception &e) {
        std::cerr << "❌ Performance test error: " << e.what() << std::endl;
    }
}

int main(int argc, char *argv[]) {
    std::cout << "Psyne Zero-Copy TCP Networking Demo\n";
    std::cout << "====================================\n\n";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [server|client|perf]\n\n";
        std::cout << "Examples:\n";
        std::cout << "  " << argv[0] << " server    # Run TCP server (processes 20 messages)\n";
        std::cout << "  " << argv[0] << " client    # Run TCP client (sends 10 messages)\n";
        std::cout << "  " << argv[0] << " perf      # Performance benchmark (50 large messages)\n\n";
        std::cout << "Features demonstrated:\n";
        std::cout << "  ✓ Zero-copy messaging with FloatVector\n";
        std::cout << "  ✓ Direct ring buffer access\n";
        std::cout << "  ✓ High-performance networking\n";
        std::cout << "  ✓ Bidirectional communication\n";
        std::cout << "  ✓ Performance metrics\n";
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "server") {
        run_server();
    } else if (mode == "client") {
        run_client();
    } else if (mode == "perf") {
        run_performance_test();
    } else {
        std::cerr << "❌ Invalid mode. Use 'server', 'client', or 'perf'\n";
        return 1;
    }

    return 0;
}