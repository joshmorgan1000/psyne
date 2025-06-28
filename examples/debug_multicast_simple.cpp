#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace psyne;

int main() {
    try {
        std::cout << "=== Simple Multicast Debug ===" << std::endl;
        
        const std::string multicast_addr = "239.255.0.1";
        const uint16_t port = 12345;
        
        // Create publisher
        std::cout << "Creating multicast publisher..." << std::endl;
        auto publisher = multicast::create_publisher(multicast_addr, port);
        
        // Create subscriber
        std::cout << "Creating multicast subscriber..." << std::endl;
        auto subscriber = multicast::create_subscriber(multicast_addr, port);
        
        // Wait for subscriber to join
        std::cout << "Waiting for subscriber to join..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Send one message
        std::cout << "Sending one test message..." << std::endl;
        FloatVector msg(*publisher);
        msg.resize(3);
        msg[0] = 1.0f;
        msg[1] = 2.0f;
        msg[2] = 3.0f;
        
        std::cout << "Message prepared with " << msg.size() << " floats" << std::endl;
        std::cout << "Values: " << msg[0] << ", " << msg[1] << ", " << msg[2] << std::endl;
        
        publisher->send(msg);
        std::cout << "Message sent!" << std::endl;
        
        // Try to receive
        std::cout << "Attempting to receive..." << std::endl;
        for (int i = 0; i < 50; ++i) {  // Try for 5 seconds
            auto received = subscriber->receive<FloatVector>();
            if (received) {
                std::cout << "SUCCESS: Received message with " << received->size() << " floats!" << std::endl;
                std::cout << "Values: ";
                for (size_t j = 0; j < received->size(); ++j) {
                    std::cout << (*received)[j] << " ";
                }
                std::cout << std::endl;
                return 0;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "No messages received after 5 seconds" << std::endl;
        return 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}