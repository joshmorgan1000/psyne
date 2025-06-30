/**
 * @file debug_multicast.cpp
 * @brief Debug demonstration for messaging concepts
 * 
 * Shows debugging techniques for message handling in Psyne v1.3.0
 */

#include <iostream>
#include <psyne/psyne.hpp>

using namespace psyne;

int main() {
    try {
        std::cout << "=== Debug Messaging Demo ===" << std::endl;
        std::cout << "Demonstrating message debugging techniques" << std::endl;

        // Create a channel for debugging
        auto channel = create_channel("memory://debug-channel", 1024 * 1024, ChannelMode::SPSC);

        std::cout << "\nCreating and debugging a FloatVector message..." << std::endl;
        FloatVector msg(*channel);

        std::cout << "Initial message state:" << std::endl;
        std::cout << "  msg.is_valid(): " << msg.is_valid() << std::endl;
        std::cout << "  msg.size(): " << msg.size() << std::endl;
        std::cout << "  msg.capacity(): " << msg.capacity() << std::endl;

        std::cout << "\nResizing message to 5 elements..." << std::endl;
        msg.resize(5);

        std::cout << "After resize:" << std::endl;
        std::cout << "  msg.size(): " << msg.size() << std::endl;
        std::cout << "  msg.capacity(): " << msg.capacity() << std::endl;

        std::cout << "\nFilling message with debug values..." << std::endl;
        for (size_t i = 0; i < msg.size(); ++i) {
            msg[i] = static_cast<float>(i * 10.5f);
            std::cout << "  msg[" << i << "] = " << msg[i] << std::endl;
        }

        std::cout << "\nVerifying message contents:" << std::endl;
        for (size_t i = 0; i < msg.size(); ++i) {
            float expected = static_cast<float>(i * 10.5f);
            if (msg[i] == expected) {
                std::cout << "  ✓ msg[" << i << "] = " << msg[i] << " (correct)" << std::endl;
            } else {
                std::cout << "  ✗ msg[" << i << "] = " << msg[i] << " (expected " << expected << ")" << std::endl;
            }
        }

        std::cout << "\nSending message..." << std::endl;
        msg.send();
        std::cout << "Message sent successfully" << std::endl;

        // Test receiving
        std::cout << "\nTesting message receive..." << std::endl;
        size_t recv_size;
        uint32_t recv_type;
        void* recv_data = channel->receive_raw_message(recv_size, recv_type);
        
        if (recv_data) {
            std::cout << "Received message:" << std::endl;
            std::cout << "  Size: " << recv_size << " bytes" << std::endl;
            std::cout << "  Type: " << recv_type << std::endl;
            
            if (recv_type == FloatVector::message_type) {
                FloatVector received(recv_data, recv_size);
                std::cout << "  Float vector size: " << received.size() << std::endl;
                std::cout << "  Values: ";
                for (size_t i = 0; i < received.size(); ++i) {
                    std::cout << received[i] << " ";
                }
                std::cout << std::endl;
            }
            
            channel->release_raw_message(recv_data);
            std::cout << "Message released successfully" << std::endl;
        } else {
            std::cout << "No message received (this is expected in simulation)" << std::endl;
        }

        std::cout << "\n=== Channel Debug Information ===" << std::endl;
        std::cout << "Channel URI: " << channel->uri() << std::endl;
        std::cout << "Channel Type: " << (channel->type() == ChannelType::MultiType ? "MultiType" : "SingleType") << std::endl;
        std::cout << "Channel Mode: " << (channel->mode() == ChannelMode::SPSC ? "SPSC" : "Other") << std::endl;

        if (channel->has_metrics()) {
            auto metrics = channel->get_metrics();
            std::cout << "\nChannel Metrics:" << std::endl;
            std::cout << "  Messages sent: " << metrics.messages_sent << std::endl;
            std::cout << "  Messages received: " << metrics.messages_received << std::endl;
            std::cout << "  Bytes sent: " << metrics.bytes_sent << std::endl;
            std::cout << "  Bytes received: " << metrics.bytes_received << std::endl;
        } else {
            std::cout << "\nNo metrics available for this channel" << std::endl;
        }

        std::cout << "\n=== Debug Tips ===" << std::endl;
        std::cout << "1. Always check is_valid() before using messages" << std::endl;
        std::cout << "2. Monitor message size and capacity for buffer usage" << std::endl;
        std::cout << "3. Verify message type when receiving raw messages" << std::endl;
        std::cout << "4. Use channel metrics to track performance" << std::endl;
        std::cout << "5. Always release raw messages to prevent leaks" << std::endl;

        std::cout << "\nDebug demo completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Debug demo failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}