#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace psyne;

void producer() {
    std::cout << "[Producer] Starting...\n";
    
    try {
        auto channel = create_channel("ipc://test_channel", 
                                      1024 * 1024,
                                      ChannelMode::SPSC,
                                      ChannelType::SingleType);
        
        std::cout << "[Producer] Channel created\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        for (int i = 0; i < 5; ++i) {
            std::cout << "[Producer] Creating message " << i << "\n";
            FloatVector msg(*channel);
            
            if (!msg.is_valid()) {
                std::cout << "[Producer] Failed to allocate message\n";
                break;
            }
            
            msg.resize(3);
            msg[0] = i;
            msg[1] = i * 10;
            msg[2] = i * 100;
            
            std::cout << "[Producer] Sending: " << msg[0] << " " << msg[1] << " " << msg[2] << "\n";
            channel->send(msg);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "[Producer] Done\n";
    } catch (const std::exception& e) {
        std::cout << "[Producer] Error: " << e.what() << "\n";
    }
}

void consumer() {
    std::cout << "[Consumer] Starting...\n";
    
    // Wait a bit to ensure producer creates the channel first
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    auto channel = create_channel("ipc://test_channel",
                                  1024 * 1024,
                                  ChannelMode::SPSC,
                                  ChannelType::SingleType);
    
    int count = 0;
    int timeouts = 0;
    while (count < 5 && timeouts < 10) {
        auto msg = channel->receive<FloatVector>(std::chrono::milliseconds(500));
        
        if (msg) {
            std::cout << "[Consumer] Received: ";
            for (float val : *msg) {
                std::cout << val << " ";
            }
            std::cout << "\n";
            count++;
            timeouts = 0;  // Reset timeout counter
        } else {
            std::cout << "[Consumer] Timeout " << ++timeouts << "/10\n";
        }
    }
    
    std::cout << "[Consumer] Done\n";
}

int main() {
    std::cout << "IPC Test - Running producer and consumer in threads\n";
    std::cout << "===================================================\n\n";
    
    std::thread producer_thread(producer);
    std::thread consumer_thread(consumer);
    
    producer_thread.join();
    consumer_thread.join();
    
    std::cout << "\nTest completed\n";
    return 0;
}