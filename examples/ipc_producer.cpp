#include <psyne/psyne.hpp>
#include <iostream>
#include <chrono>
#include <thread>

using namespace psyne;

int main() {
    std::cout << "IPC Producer Example\n";
    std::cout << "===================\n\n";
    
    try {
        // Create an IPC channel - this will create shared memory
        auto channel = create_channel("ipc://demo_channel", 
                                      1024 * 1024,  // 1MB buffer
                                      ChannelMode::SPSC,
                                      ChannelType::SingleType);
        
        std::cout << "Created IPC channel: " << channel->uri() << "\n";
        std::cout << "Sending messages every 100ms...\n\n";
        
        // Send messages
        for (int i = 0; i < 10; ++i) {
            FloatVector msg(*channel);
            
            if (!msg.is_valid()) {
                std::cerr << "Failed to allocate message\n";
                break;
            }
            
            // Fill with test data
            msg.resize(5);
            for (size_t j = 0; j < 5; ++j) {
                msg[j] = static_cast<float>(i * 10 + j);
            }
            
            std::cout << "Sending message " << i << ": ";
            for (float val : msg) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
            
            channel->send(msg);
            
            // Wait a bit
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\nProducer finished. Press Enter to exit...\n";
        std::cin.get();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}