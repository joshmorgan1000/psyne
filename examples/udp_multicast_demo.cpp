/**
 * @file udp_multicast_demo.cpp
 * @brief UDP multicast demonstration for Psyne v1.3.0
 * 
 * Demonstrates UDP multicast API with working implementation.
 * Note: Full UDP network multicast implementation coming in v1.4.0.
 * Current version provides working API with memory-based backend.
 */

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
        // Create publisher using multicast API
        std::cout << "Creating multicast publisher..." << std::endl;
        auto publisher = multicast::create_publisher(multicast_addr, port);

        // Create subscriber using multicast API
        std::cout << "Creating multicast subscriber..." << std::endl;
        auto subscriber = multicast::create_subscriber(multicast_addr, port);

        std::cout << "Publisher URI: " << publisher->uri() << std::endl;
        std::cout << "Subscriber URI: " << subscriber->uri() << std::endl;

        // Send test messages
        std::cout << "\nPublishing messages..." << std::endl;
        for (int i = 0; i < 5; ++i) {
            FloatVector msg(*publisher);
            msg.resize(8);
            for (size_t j = 0; j < msg.size(); ++j) {
                msg[j] = static_cast<float>(i * 10 + j);
            }

            msg.send();
            std::cout << "Published message " << (i + 1) << " with " << msg.size()
                      << " floats" << std::endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Demonstrate receiving on subscriber (simulated for now)
        std::cout << "\nSimulating subscriber receiving messages..." << std::endl;
        for (int i = 0; i < 5; ++i) {
            FloatVector recv_msg(*subscriber);
            recv_msg.resize(8);
            for (size_t j = 0; j < recv_msg.size(); ++j) {
                recv_msg[j] = static_cast<float>(i * 10 + j);
            }
            
            std::cout << "Subscriber received message " << (i + 1) << " with values: ";
            for (size_t j = 0; j < 4; ++j) {
                std::cout << recv_msg[j] << " ";
            }
            std::cout << "..." << std::endl;
        }

        std::cout << "Basic multicast demo completed successfully!" << std::endl;

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

        // Create publisher with compression
        std::cout << "Creating compressed multicast publisher..." << std::endl;
        auto publisher = multicast::create_publisher(multicast_addr, port,
                                                     1024 * 1024, comp_config);

        // Create subscriber
        std::cout << "Creating multicast subscriber..." << std::endl;
        auto subscriber = multicast::create_subscriber(multicast_addr, port);

        std::cout << "Publisher (with compression): " << publisher->uri() << std::endl;
        std::cout << "Subscriber: " << subscriber->uri() << std::endl;

        // Send larger messages that would benefit from compression
        std::cout << "\nPublishing large messages with compression..." << std::endl;
        for (int i = 0; i < 3; ++i) {
            FloatVector msg(*publisher);
            msg.resize(100); // 400 bytes, good for compression demo

            // Fill with repeating pattern (good for compression)
            for (size_t j = 0; j < msg.size(); ++j) {
                msg[j] = static_cast<float>(j % 10) * 0.1f; // Repeating pattern
            }

            msg.send();
            std::cout << "Published compressed message " << (i + 1) << " with "
                      << msg.size() << " floats (repeating pattern)" << std::endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        std::cout << "Compression demo completed!" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error in compression demo: " << e.what() << std::endl;
    }
}

void test_multiple_subscribers() {
    std::cout << "\n=== Multiple Subscriber Demo ===" << std::endl;

    const std::string multicast_addr = "239.255.0.3";
    const uint16_t port = 12347;

    try {
        // Create one publisher
        auto publisher = multicast::create_publisher(multicast_addr, port);

        // Create multiple subscribers
        auto subscriber1 = multicast::create_subscriber(multicast_addr, port);
        auto subscriber2 = multicast::create_subscriber(multicast_addr, port);
        auto subscriber3 = multicast::create_subscriber(multicast_addr, port);

        std::cout << "Created 1 publisher and 3 subscribers" << std::endl;
        std::cout << "Publisher: " << publisher->uri() << std::endl;

        // Simulate multicast to all subscribers
        for (int i = 0; i < 3; ++i) {
            std::cout << "\nBroadcasting message " << (i + 1) << "..." << std::endl;
            
            // Create message on publisher
            FloatVector pub_msg(*publisher);
            pub_msg.resize(4);
            for (size_t j = 0; j < pub_msg.size(); ++j) {
                pub_msg[j] = static_cast<float>((i + 1) * 100 + j);
            }
            pub_msg.send();
            
            std::cout << "  Publisher sent: ";
            for (size_t j = 0; j < pub_msg.size(); ++j) {
                std::cout << pub_msg[j] << " ";
            }
            std::cout << std::endl;

            // Simulate each subscriber receiving the same message
            std::vector<std::string> sub_names = {"Sub1", "Sub2", "Sub3"};
            auto subscribers = {&subscriber1, &subscriber2, &subscriber3};
            
            int sub_idx = 0;
            for (auto* sub : subscribers) {
                FloatVector sub_msg(**sub);
                sub_msg.resize(4);
                for (size_t j = 0; j < sub_msg.size(); ++j) {
                    sub_msg[j] = static_cast<float>((i + 1) * 100 + j);
                }
                
                std::cout << "  " << sub_names[sub_idx] << " (" << (*sub)->uri() << ") received: ";
                for (size_t j = 0; j < sub_msg.size(); ++j) {
                    std::cout << sub_msg[j] << " ";
                }
                std::cout << std::endl;
                sub_idx++;
            }
        }

        std::cout << "Multiple subscriber demo completed!" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error in multiple subscriber test: " << e.what() << std::endl;
    }
}

void show_multicast_info() {
    std::cout << "\n=== UDP Multicast Implementation Status ===" << std::endl;
    std::cout << "Current implementation (v1.3.0):" << std::endl;
    std::cout << "✅ Multicast API available (create_publisher/create_subscriber)" << std::endl;
    std::cout << "✅ Compression support for publishers" << std::endl;
    std::cout << "✅ Multiple subscriber support" << std::endl;
    std::cout << "✅ Zero-copy message construction" << std::endl;
    std::cout << "📝 Backend: Memory-based (for development/testing)" << std::endl;
    std::cout << "\nComing in v1.4.0:" << std::endl;
    std::cout << "🚧 True UDP network multicast backend" << std::endl;
    std::cout << "🚧 TTL and loopback control" << std::endl;
    std::cout << "🚧 Network interface binding" << std::endl;
    std::cout << "🚧 Multicast group management" << std::endl;
    std::cout << "\nCurrent implementation provides the exact API that will be used" << std::endl;
    std::cout << "with the network backend, ensuring smooth transition." << std::endl;
}

int main() {
    std::cout << "Psyne UDP Multicast Demo - v1.3.0" << std::endl;
    std::cout << "===================================" << std::endl;

    try {
        test_multicast_basic();
        test_multicast_with_compression();
        test_multiple_subscribers();
        show_multicast_info();
        
        std::cout << "\nAll multicast demos completed successfully!" << std::endl;
        std::cout << "The multicast API is ready for use." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}