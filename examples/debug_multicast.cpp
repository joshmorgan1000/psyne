#include <psyne/psyne.hpp>
#include <iostream>

using namespace psyne;

int main() {
    try {
        std::cout << "=== Debug Multicast Issue ===" << std::endl;
        
        const std::string multicast_addr = "239.255.0.1";
        const uint16_t port = 12345;
        
        // Create publisher
        std::cout << "Creating multicast publisher..." << std::endl;
        auto publisher = multicast::create_publisher(multicast_addr, port);
        
        std::cout << "Creating FloatVector message..." << std::endl;
        FloatVector msg(*publisher);
        
        std::cout << "Initial state:" << std::endl;
        std::cout << "  msg.is_valid(): " << msg.is_valid() << std::endl;
        std::cout << "  msg.size(): " << msg.size() << std::endl;
        std::cout << "  msg.capacity(): " << msg.capacity() << std::endl;
        
        std::cout << "Resizing to 5..." << std::endl;
        msg.resize(5);
        
        std::cout << "After resize:" << std::endl;
        std::cout << "  msg.size(): " << msg.size() << std::endl;
        
        std::cout << "Setting values..." << std::endl;
        for (size_t i = 0; i < 5; ++i) {
            msg[i] = static_cast<float>(i * 10);
        }
        
        std::cout << "After setting values:" << std::endl;
        std::cout << "  msg.size(): " << msg.size() << std::endl;
        std::cout << "  Values: ";
        for (size_t i = 0; i < msg.size(); ++i) {
            std::cout << msg[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Before send:" << std::endl;
        std::cout << "  msg.size(): " << msg.size() << std::endl;
        
        publisher->send(msg);
        
        std::cout << "Message sent!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}