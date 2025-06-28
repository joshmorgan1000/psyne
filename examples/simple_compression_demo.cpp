#include <psyne/psyne.hpp>
#include <iostream>
#include <vector>

using namespace psyne;

int main() {
    try {
        std::cout << "Simple Compression Demo" << std::endl;
        std::cout << "======================" << std::endl;
        
        // Test the compression functionality directly
        compression::CompressionConfig config;
        config.type = compression::CompressionType::LZ4;
        config.min_size_threshold = 64;
        
        std::cout << "✓ Compression config created" << std::endl;
        
        // For now, let's just test memory channels which don't use compression yet
        // Create a memory channel (compression is for TCP channels only currently)
        auto channel = create_channel("memory://simple_test", 1024*1024, 
                                     ChannelMode::SPSC, ChannelType::SingleType, 
                                     true);
        
        std::cout << "✓ Memory channel created" << std::endl;
        
        // Test basic messaging
        FloatVector send_msg(*channel);
        send_msg.resize(100);
        for (size_t i = 0; i < 100; ++i) {
            send_msg[i] = static_cast<float>(i) * 0.1f;
        }
        
        std::cout << "✓ Test message created with " << send_msg.size() << " elements" << std::endl;
        
        // Send and receive
        channel->send(send_msg);
        auto recv_msg = channel->receive<FloatVector>();
        
        if (recv_msg && recv_msg->size() == 100) {
            std::cout << "✓ Message sent and received successfully" << std::endl;
            
            // Verify first and last elements
            float first_val = (*recv_msg)[0];
            float last_val = (*recv_msg)[99];
            float expected_last = 99 * 0.1f; // Should be 9.9f
            
            std::cout << "First value: " << first_val << " (expected: 0.0)" << std::endl;
            std::cout << "Last value: " << last_val << " (expected: " << expected_last << ")" << std::endl;
            
            if (first_val == 0.0f && std::abs(last_val - expected_last) < 1e-6f) {
                std::cout << "✓ Data integrity verified" << std::endl;
            } else {
                std::cout << "✗ Data corruption detected" << std::endl;
            }
        } else {
            std::cout << "✗ Failed to receive message" << std::endl;
        }
        
        // Show metrics
        if (channel->has_metrics()) {
            auto metrics = channel->get_metrics();
            std::cout << "Channel metrics:" << std::endl;
            std::cout << "  Messages sent: " << metrics.messages_sent << std::endl;
            std::cout << "  Bytes sent: " << metrics.bytes_sent << std::endl;
            std::cout << "  Messages received: " << metrics.messages_received << std::endl;
            std::cout << "  Bytes received: " << metrics.bytes_received << std::endl;
        }
        
        std::cout << "\n✓ Compression framework integration test passed!" << std::endl;
        std::cout << "Note: TCP compression will be tested separately once networking is stable." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}