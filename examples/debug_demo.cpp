/**
 * @file debug_demo.cpp
 * @brief Debug and metrics demonstration for Psyne
 * 
 * Shows basic debugging and monitoring capabilities in Psyne v1.3.0
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

using namespace psyne;
using namespace std::chrono;

// Simple message for debugging
class DebugMessage : public Message<DebugMessage> {
public:
    static constexpr uint32_t message_type = 42;
    using Message<DebugMessage>::Message;

    static size_t calculate_size() {
        return 64; // 64 bytes
    }
    
    void set_id(int id) {
        *reinterpret_cast<int*>(data()) = id;
    }
    
    int get_id() const {
        return *reinterpret_cast<const int*>(data());
    }
};

void test_channel_metrics() {
    std::cout << "=== Channel Metrics Demo ===" << std::endl;

    // Create a channel with metrics enabled
    auto channel = create_channel("memory://debug_test", 1024 * 1024, 
                                  ChannelMode::SPSC, ChannelType::MultiType, true);

    std::cout << "Channel created with metrics enabled" << std::endl;
    std::cout << "URI: " << channel->uri() << std::endl;
    std::cout << "Type: " << (channel->type() == ChannelType::MultiType ? "MultiType" : "SingleType") << std::endl;
    std::cout << "Mode: " << (channel->mode() == ChannelMode::SPSC ? "SPSC" : "Other") << std::endl;

    // Send some messages
    std::cout << "\nSending test messages..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        try {
            DebugMessage msg(*channel);
            msg.set_id(i);
            msg.send();
            std::cout << "Sent message " << i << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Failed to send message " << i << ": " << e.what() << std::endl;
        }
    }

    // Try to receive messages
    std::cout << "\nReceiving messages..." << std::endl;
    int received = 0;
    for (int i = 0; i < 20; ++i) {  // Try more than we sent
        size_t size;
        uint32_t type;
        void* msg_data = channel->receive_raw_message(size, type);
        if (msg_data) {
            if (type == DebugMessage::message_type) {
                DebugMessage msg(msg_data, size);
                std::cout << "Received message " << msg.get_id() << std::endl;
                received++;
            }
            channel->release_raw_message(msg_data);
        } else {
            std::cout << "No message available (attempt " << (i+1) << ")" << std::endl;
            break;
        }
    }
    
    std::cout << "Total received: " << received << " messages" << std::endl;

    // Show basic metrics if available
    if (channel->has_metrics()) {
        auto metrics = channel->get_metrics();
        std::cout << "\nChannel Metrics:" << std::endl;
        std::cout << "  Messages sent: " << metrics.messages_sent << std::endl;
        std::cout << "  Messages received: " << metrics.messages_received << std::endl;
        std::cout << "  Bytes sent: " << metrics.bytes_sent << std::endl;
        std::cout << "  Bytes received: " << metrics.bytes_received << std::endl;
    } else {
        std::cout << "\nMetrics not available for this channel" << std::endl;
    }
}

void test_error_handling() {
    std::cout << "\n=== Error Handling Demo ===" << std::endl;
    
    // Test invalid URI
    try {
        auto bad_channel = create_channel("invalid://bad_uri");
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✓ Caught expected exception: " << e.what() << std::endl;
    }
    
    // Test small buffer behavior
    try {
        auto small_channel = create_channel("memory://small", 1024);  // Very small buffer
        
        std::cout << "Testing buffer overflow behavior..." << std::endl;
        int sent = 0;
        for (int i = 0; i < 100; ++i) {  // Try to send many messages
            try {
                DebugMessage msg(*small_channel);
                msg.set_id(i);
                msg.send();
                sent++;
            } catch (const std::exception& e) {
                std::cout << "Buffer full after " << sent << " messages" << std::endl;
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cout << "Channel creation/operation error: " << e.what() << std::endl;
    }
}

void test_threading() {
    std::cout << "\n=== Threading Demo ===" << std::endl;
    
    auto channel = create_channel("memory://thread_test", 4*1024*1024, ChannelMode::SPSC);
    
    const int num_messages = 1000;
    std::atomic<int> sent{0};
    std::atomic<int> received{0};
    
    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < num_messages; ++i) {
            try {
                DebugMessage msg(*channel);
                msg.set_id(i);
                msg.send();
                sent++;
                
                if (i % 100 == 0) {
                    std::cout << "Producer: sent " << i << " messages" << std::endl;
                }
                
                // Small delay to make it realistic
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            } catch (const std::exception& e) {
                std::cout << "Producer error: " << e.what() << std::endl;
                break;
            }
        }
        std::cout << "Producer finished, sent " << sent.load() << " messages" << std::endl;
    });
    
    // Consumer thread
    std::thread consumer([&]() {
        while (received.load() < num_messages) {
            size_t size;
            uint32_t type;
            void* msg_data = channel->receive_raw_message(size, type);
            if (msg_data) {
                if (type == DebugMessage::message_type) {
                    received++;
                    
                    if (received % 100 == 0) {
                        std::cout << "Consumer: received " << received.load() << " messages" << std::endl;
                    }
                }
                channel->release_raw_message(msg_data);
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        std::cout << "Consumer finished, received " << received.load() << " messages" << std::endl;
    });
    
    producer.join();
    consumer.join();
    
    std::cout << "Threading test completed successfully!" << std::endl;
    std::cout << "Final counts - Sent: " << sent.load() << ", Received: " << received.load() << std::endl;
}

void show_debug_tips() {
    std::cout << "\n=== Psyne Debugging Tips ===" << std::endl;
    std::cout << "1. Enable metrics when creating channels for monitoring" << std::endl;
    std::cout << "2. Use try-catch blocks around message operations" << std::endl;
    std::cout << "3. Check return values from receive operations" << std::endl;
    std::cout << "4. Monitor buffer usage in multi-threaded scenarios" << std::endl;
    std::cout << "5. Use atomic counters for thread-safe statistics" << std::endl;
    std::cout << "6. Consider channel modes for your threading model" << std::endl;
    std::cout << "7. Test with small buffers to verify overflow handling" << std::endl;
}

int main() {
    std::cout << "Psyne Debug Demo - v1.3.0" << std::endl;
    std::cout << "=========================" << std::endl;

    try {
        test_channel_metrics();
        test_error_handling();
        test_threading();
        show_debug_tips();
        
        std::cout << "\nAll debug tests completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Demo failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}