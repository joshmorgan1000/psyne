#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace psyne;

void run_server() {
    std::cout << "Starting WebSocket server on port 8080..." << std::endl;
    
    // Create WebSocket server channel
    auto channel = create_channel("ws://:8080", 4 * 1024 * 1024);
    
    std::cout << "Server listening for WebSocket connections..." << std::endl;
    
    // Receive messages
    int message_count = 0;
    while (message_count < 10) {
        auto msg = channel->receive<FloatVector>(std::chrono::milliseconds(100));
        if (msg) {
            std::cout << "Server received message " << ++message_count 
                      << " with " << msg->size() << " floats" << std::endl;
            
            // Echo back with modification
            FloatVector response(*channel);
            response.resize(msg->size());
            for (size_t i = 0; i < msg->size(); ++i) {
                response[i] = (*msg)[i] * 2.0f;
            }
            channel->send(response);
            std::cout << "Server sent response" << std::endl;
        }
    }
    
    std::cout << "Server shutting down" << std::endl;
}

void run_client() {
    std::cout << "Starting WebSocket client..." << std::endl;
    
    // Give server time to start
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Create WebSocket client channel
    auto channel = create_channel("ws://localhost:8080", 4 * 1024 * 1024);
    
    std::cout << "Client connected to WebSocket server" << std::endl;
    
    // Send messages
    for (int i = 0; i < 10; ++i) {
        FloatVector msg(*channel);
        msg.resize(100);
        
        // Fill with test data
        for (size_t j = 0; j < 100; ++j) {
            msg[j] = i + j * 0.01f;
        }
        
        channel->send(msg);
        std::cout << "Client sent message " << i + 1 << std::endl;
        
        // Wait for echo response
        auto response = channel->receive<FloatVector>(std::chrono::seconds(5));
        if (response) {
            std::cout << "Client received response with " << response->size() 
                      << " floats" << std::endl;
            
            // Verify doubled values
            bool correct = true;
            for (size_t j = 0; j < std::min(size_t(10), response->size()); ++j) {
                float expected = (i + j * 0.01f) * 2.0f;
                if (std::abs((*response)[j] - expected) > 0.001f) {
                    correct = false;
                    break;
                }
            }
            std::cout << "Response values are " << (correct ? "correct" : "incorrect") << std::endl;
        } else {
            std::cout << "Client timeout waiting for response" << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "Client shutting down" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [server|client]" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    
    try {
        if (mode == "server") {
            run_server();
        } else if (mode == "client") {
            run_client();
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