#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace psyne;

int main() {
    std::cout << "TCP Server Example\n";
    std::cout << "==================\n\n";
    
    try {
        // Create server channel - empty host means listen on all interfaces
        auto channel = create_channel("tcp://:9999",
                                      1024 * 1024,  // 1MB buffer
                                      ChannelMode::SPSC,
                                      ChannelType::SingleType);
        
        std::cout << "Server listening on port 9999...\n";
        std::cout << "Waiting for messages...\n\n";
        
        // Process messages
        int count = 0;
        while (true) {
            auto msg = channel->receive<FloatVector>(std::chrono::milliseconds(1000));
            
            if (msg) {
                std::cout << "Received message " << count++ << ": ";
                for (float val : *msg) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
                
                // Echo the message back (modified)
                FloatVector reply(*channel);
                reply.resize(msg->size());
                for (size_t i = 0; i < msg->size(); ++i) {
                    reply[i] = (*msg)[i] * 2.0f;  // Double the values
                }
                
                std::cout << "Sending reply: ";
                for (float val : reply) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
                
                channel->send(reply);
                
            } else {
                std::cout << "." << std::flush;
            }
            
            // Exit after 10 messages
            if (count >= 10) {
                std::cout << "\nReceived 10 messages, shutting down.\n";
                break;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}