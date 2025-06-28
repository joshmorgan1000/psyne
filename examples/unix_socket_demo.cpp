#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace psyne;

void server_example() {
    std::cout << "Starting Unix domain socket server..." << std::endl;
    
    // Create server channel (note the @ prefix for server mode)
    auto server = create_channel("unix://@/tmp/psyne_demo.sock");
    
    std::cout << "Server listening on /tmp/psyne_demo.sock" << std::endl;
    
    // Wait for and receive messages
    for (int i = 0; i < 5; ++i) {
        auto msg = server->receive<FloatVector>(std::chrono::seconds(10));
        if (msg) {
            std::cout << "Server received " << msg->size() << " floats: ";
            for (size_t j = 0; j < std::min(msg->size(), size_t(5)); ++j) {
                std::cout << (*msg)[j] << " ";
            }
            if (msg->size() > 5) std::cout << "...";
            std::cout << std::endl;
        } else {
            std::cout << "Server timeout waiting for message" << std::endl;
        }
    }
    
    server->stop();
    std::cout << "Server stopped" << std::endl;
}

void client_example() {
    // Give server time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "Starting Unix domain socket client..." << std::endl;
    
    // Create client channel (no @ prefix for client mode)
    auto client = create_channel("unix:///tmp/psyne_demo.sock");
    
    std::cout << "Client connected to /tmp/psyne_demo.sock" << std::endl;
    
    // Send some messages
    for (int i = 0; i < 5; ++i) {
        FloatVector msg(*client);
        msg.resize(10);
        
        // Fill with some data
        for (size_t j = 0; j < msg.size(); ++j) {
            msg[j] = static_cast<float>(i * 10 + j);
        }
        
        std::cout << "Client sending message " << i << std::endl;
        msg.send();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    client->stop();
    std::cout << "Client stopped" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [server|client]" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    
    try {
        if (mode == "server") {
            server_example();
        } else if (mode == "client") {
            client_example();
        } else {
            std::cerr << "Invalid mode. Use 'server' or 'client'" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}