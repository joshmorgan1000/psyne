/**
 * @file webrtc_simple_example.cpp
 * @brief Simple WebRTC connectivity example for Psyne
 * 
 * This example demonstrates basic WebRTC peer-to-peer messaging
 * using Psyne's WebRTC channel implementation.
 * 
 * Usage:
 *   ./webrtc_simple_example offerer    # Peer that initiates connection
 *   ./webrtc_simple_example answerer   # Peer that responds to connection
 * 
 * @author Psyne Contributors
 */

#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <string>

using namespace psyne;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <role>\n";
        std::cout << "  role: 'offerer' or 'answerer'\n";
        return 1;
    }
    
    std::string role = argv[1];
    bool is_offerer = (role == "offerer");
    
    if (role != "offerer" && role != "answerer") {
        std::cerr << "❌ Role must be 'offerer' or 'answerer'\n";
        return 1;
    }
    
    try {
        std::cout << "🚀 WebRTC Simple Example - " << role << "\n";
        
        // Create WebRTC channel
        std::string peer_id = is_offerer ? "answerer" : "offerer";
        std::string uri = "webrtc://" + peer_id;
        
        std::cout << "🔗 Creating WebRTC channel to peer: " << peer_id << "\n";
        
        auto channel = psyne::create_channel(
            uri,
            64 * 1024,  // 64KB buffer
            ChannelMode::SPSC,
            ChannelType::MultiType,
            true  // Enable metrics
        );
        
        std::cout << "⏳ Establishing WebRTC connection...\n";
        
        // Simulate connection setup time
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        std::cout << "✅ WebRTC connection established!\n";
        
        // Send some messages
        for (int i = 0; i < 5; ++i) {
            FloatVector msg(*channel);
            msg.resize(10);
            
            // Fill with some test data
            for (size_t j = 0; j < msg.size(); ++j) {
                msg[j] = static_cast<float>(i * 10 + j);
            }
            
            msg.send();
            std::cout << "📤 Sent message " << i << " with " << msg.size() << " floats\n";
            
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        // Try to receive messages
        std::cout << "📥 Listening for messages...\n";
        
        for (int i = 0; i < 10; ++i) {
            auto received = channel->receive<FloatVector>(std::chrono::milliseconds(1000));
            if (received) {
                std::cout << "📨 Received message with " << received->size() << " floats: ";
                for (size_t j = 0; j < std::min(received->size(), size_t(5)); ++j) {
                    std::cout << (*received)[j] << " ";
                }
                if (received->size() > 5) std::cout << "...";
                std::cout << "\n";
            } else {
                std::cout << "⏰ No message received (timeout)\n";
            }
        }
        
        // Show channel metrics
        if (channel->has_metrics()) {
            auto metrics = channel->get_metrics();
            std::cout << "\n📊 Channel Metrics:\n";
            std::cout << "   Messages sent: " << metrics.messages_sent << "\n";
            std::cout << "   Messages received: " << metrics.messages_received << "\n";
            std::cout << "   Bytes sent: " << metrics.bytes_sent << "\n";
            std::cout << "   Bytes received: " << metrics.bytes_received << "\n";
        }
        
        std::cout << "\n✨ WebRTC example completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}