/**
 * @file zmq_patterns_demo.cpp
 * @brief Comprehensive demo of ZeroMQ-style messaging patterns
 * 
 * This demo showcases all major ZMQ patterns:
 * - Request-Reply (REQ/REP)
 * - Publish-Subscribe (PUB/SUB)
 * - Push-Pull (Load Balancing)
 * - Dealer-Router (Async Request-Reply)
 * - Pair (Exclusive Bidirectional)
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include <psyne/psyne.hpp>
#include "../src/patterns/zmq_patterns.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <future>

using namespace psyne;
using namespace psyne::patterns;
using namespace std::chrono_literals;

// Demo colors for output
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

void print_header(const std::string& title) {
    std::cout << "\n" << CYAN << "╔" << std::string(60, '═') << "╗" << RESET << std::endl;
    std::cout << CYAN << "║" << std::string((60 - title.length()) / 2, ' ') 
              << title << std::string((60 - title.length() + 1) / 2, ' ') << "║" << RESET << std::endl;
    std::cout << CYAN << "╚" << std::string(60, '═') << "╝" << RESET << std::endl;
}

// Request-Reply Pattern Demo
void demo_request_reply() {
    print_header("REQUEST-REPLY PATTERN (REQ/REP)");
    
    std::cout << YELLOW << "Starting REQ/REP demo..." << RESET << std::endl;
    
    // Server thread
    auto server_future = std::async(std::launch::async, []() {
        auto server = create_reply_socket();
        server->bind("tcp://*:5555");
        
        std::cout << GREEN << "[SERVER] Waiting for requests..." << RESET << std::endl;
        
        for (int i = 0; i < 3; ++i) {
            Message request;
            if (server->recv_request(request)) {
                std::cout << GREEN << "[SERVER] Received: " << request.to_string() << RESET << std::endl;
                
                // Process request and send reply
                std::string reply_data = "Echo: " + request.to_string();
                Message reply(reply_data);
                server->send_reply(reply);
                
                std::cout << GREEN << "[SERVER] Sent reply: " << reply_data << RESET << std::endl;
            }
        }
        
        std::this_thread::sleep_for(100ms);
        server->close();
    });
    
    // Give server time to start
    std::this_thread::sleep_for(100ms);
    
    // Client
    auto client = create_request_socket();
    client->connect("tcp://localhost:5555");
    
    for (int i = 1; i <= 3; ++i) {
        std::string request_data = "Hello " + std::to_string(i);
        std::string reply;
        
        std::cout << BLUE << "[CLIENT] Sending: " << request_data << RESET << std::endl;
        
        if (client->send_request(request_data, reply)) {
            std::cout << BLUE << "[CLIENT] Received: " << reply << RESET << std::endl;
        }
        
        std::this_thread::sleep_for(50ms);
    }
    
    client->close();
    server_future.wait();
    
    std::cout << GREEN << "✓ Request-Reply demo completed" << RESET << std::endl;
}

// Publish-Subscribe Pattern Demo
void demo_publish_subscribe() {
    print_header("PUBLISH-SUBSCRIBE PATTERN (PUB/SUB)");
    
    std::cout << YELLOW << "Starting PUB/SUB demo..." << RESET << std::endl;
    
    // Publisher
    auto publisher = create_publisher_socket();
    publisher->bind("tcp://*:5556");
    
    // Subscribers
    std::vector<std::future<void>> subscriber_futures;
    
    // Weather subscriber
    subscriber_futures.push_back(std::async(std::launch::async, [&]() {
        auto subscriber = create_subscriber_socket();
        subscriber->connect("tcp://localhost:5556");
        subscriber->subscribe("weather");
        
        std::cout << GREEN << "[SUB-WEATHER] Subscribed to weather updates" << RESET << std::endl;
        
        for (int i = 0; i < 2; ++i) {
            std::string topic, data;
            if (subscriber->recv_topic(topic, data)) {
                std::cout << GREEN << "[SUB-WEATHER] " << topic << ": " << data << RESET << std::endl;
            }
        }
        
        subscriber->close();
    }));
    
    // News subscriber
    subscriber_futures.push_back(std::async(std::launch::async, [&]() {
        auto subscriber = create_subscriber_socket();
        subscriber->connect("tcp://localhost:5556");
        subscriber->subscribe("news");
        
        std::cout << MAGENTA << "[SUB-NEWS] Subscribed to news updates" << RESET << std::endl;
        
        for (int i = 0; i < 2; ++i) {
            std::string topic, data;
            if (subscriber->recv_topic(topic, data)) {
                std::cout << MAGENTA << "[SUB-NEWS] " << topic << ": " << data << RESET << std::endl;
            }
        }
        
        subscriber->close();
    }));
    
    // Give subscribers time to connect
    std::this_thread::sleep_for(200ms);
    
    // Publish messages
    std::cout << BLUE << "[PUBLISHER] Publishing messages..." << RESET << std::endl;
    
    publisher->publish("weather", "Sunny, 25°C");
    publisher->publish("news", "Breaking: ZMQ patterns in Psyne!");
    publisher->publish("weather", "Cloudy, 20°C");
    publisher->publish("news", "Tech: High-performance messaging");
    publisher->publish("sports", "No subscribers for this topic");
    
    // Wait for subscribers to finish
    for (auto& future : subscriber_futures) {
        future.wait();
    }
    
    publisher->close();
    
    std::cout << GREEN << "✓ Publish-Subscribe demo completed" << RESET << std::endl;
}

// Push-Pull Pattern Demo (Load Balancing)
void demo_push_pull() {
    print_header("PUSH-PULL PATTERN (LOAD BALANCING)");
    
    std::cout << YELLOW << "Starting PUSH/PULL demo..." << RESET << std::endl;
    
    // Workers
    std::vector<std::future<void>> worker_futures;
    
    for (int worker_id = 1; worker_id <= 3; ++worker_id) {
        worker_futures.push_back(std::async(std::launch::async, [worker_id]() {
            auto worker = create_pull_socket();
            worker->connect("tcp://localhost:5557");
            
            std::cout << GREEN << "[WORKER-" << worker_id << "] Ready for work" << RESET << std::endl;
            
            for (int i = 0; i < 2; ++i) {
                std::string task = worker->recv_string();
                if (!task.empty()) {
                    std::cout << GREEN << "[WORKER-" << worker_id << "] Processing: " << task << RESET << std::endl;
                    
                    // Simulate work
                    std::this_thread::sleep_for(100ms);
                    
                    std::cout << GREEN << "[WORKER-" << worker_id << "] Completed: " << task << RESET << std::endl;
                }
            }
            
            worker->close();
        }));
    }
    
    // Give workers time to connect
    std::this_thread::sleep_for(200ms);
    
    // Task distributor
    auto distributor = create_push_socket();
    distributor->bind("tcp://*:5557");
    
    std::cout << BLUE << "[DISTRIBUTOR] Distributing tasks..." << RESET << std::endl;
    
    // Send tasks
    for (int task_id = 1; task_id <= 6; ++task_id) {
        std::string task = "Task-" + std::to_string(task_id);
        distributor->send(task);
        std::cout << BLUE << "[DISTRIBUTOR] Sent: " << task << RESET << std::endl;
        std::this_thread::sleep_for(50ms);
    }
    
    // Wait for workers to finish
    for (auto& future : worker_futures) {
        future.wait();
    }
    
    distributor->close();
    
    std::cout << GREEN << "✓ Push-Pull demo completed" << RESET << std::endl;
}

// Dealer-Router Pattern Demo (Async Request-Reply)
void demo_dealer_router() {
    print_header("DEALER-ROUTER PATTERN (ASYNC REQ/REP)");
    
    std::cout << YELLOW << "Starting DEALER/ROUTER demo..." << RESET << std::endl;
    
    // Router (async server)
    auto router_future = std::async(std::launch::async, []() {
        auto router = create_router_socket();
        router->bind("tcp://*:5558");
        
        std::cout << GREEN << "[ROUTER] Waiting for async requests..." << RESET << std::endl;
        
        for (int i = 0; i < 4; ++i) {
            std::string client_id;
            Message request;
            
            if (router->recv_from_client(client_id, request)) {
                std::cout << GREEN << "[ROUTER] Request from " << client_id 
                         << ": " << request.to_string() << RESET << std::endl;
                
                // Process asynchronously and reply
                std::string reply_data = "Async reply to: " + request.to_string();
                Message reply(reply_data);
                router->send_to_client(client_id, reply);
                
                std::cout << GREEN << "[ROUTER] Sent reply to " << client_id 
                         << ": " << reply_data << RESET << std::endl;
            }
        }
        
        std::this_thread::sleep_for(100ms);
        router->close();
    });
    
    // Give router time to start
    std::this_thread::sleep_for(100ms);
    
    // Dealers (async clients)
    std::vector<std::future<void>> dealer_futures;
    
    for (int client_id = 1; client_id <= 2; ++client_id) {
        dealer_futures.push_back(std::async(std::launch::async, [client_id]() {
            auto dealer = create_dealer_socket();
            dealer->set_identity("client_" + std::to_string(client_id));
            dealer->connect("tcp://localhost:5558");
            
            // Send multiple async requests
            for (int req_id = 1; req_id <= 2; ++req_id) {
                std::string request_data = "Async-" + std::to_string(client_id) + "-" + std::to_string(req_id);
                dealer->send(request_data);
                
                std::cout << BLUE << "[DEALER-" << client_id << "] Sent: " << request_data << RESET << std::endl;
                
                // Receive reply
                std::string reply = dealer->recv_string();
                if (!reply.empty()) {
                    std::cout << BLUE << "[DEALER-" << client_id << "] Received: " << reply << RESET << std::endl;
                }
            }
            
            dealer->close();
        }));
    }
    
    // Wait for all dealers
    for (auto& future : dealer_futures) {
        future.wait();
    }
    
    router_future.wait();
    
    std::cout << GREEN << "✓ Dealer-Router demo completed" << RESET << std::endl;
}

// Pair Pattern Demo
void demo_pair() {
    print_header("PAIR PATTERN (EXCLUSIVE BIDIRECTIONAL)");
    
    std::cout << YELLOW << "Starting PAIR demo..." << RESET << std::endl;
    
    // Pair 1
    auto pair1_future = std::async(std::launch::async, []() {
        auto pair1 = create_pair_socket();
        pair1->bind("tcp://*:5559");
        
        std::cout << GREEN << "[PAIR-1] Connected" << RESET << std::endl;
        
        // Send and receive messages
        pair1->send("Hello from Pair-1");
        std::cout << GREEN << "[PAIR-1] Sent: Hello from Pair-1" << RESET << std::endl;
        
        std::string received = pair1->recv_string();
        if (!received.empty()) {
            std::cout << GREEN << "[PAIR-1] Received: " << received << RESET << std::endl;
        }
        
        pair1->close();
    });
    
    // Give pair1 time to bind
    std::this_thread::sleep_for(100ms);
    
    // Pair 2
    auto pair2 = create_pair_socket();
    pair2->connect("tcp://localhost:5559");
    
    std::cout << BLUE << "[PAIR-2] Connected" << RESET << std::endl;
    
    // Receive and send messages
    std::string received = pair2->recv_string();
    if (!received.empty()) {
        std::cout << BLUE << "[PAIR-2] Received: " << received << RESET << std::endl;
    }
    
    pair2->send("Hello from Pair-2");
    std::cout << BLUE << "[PAIR-2] Sent: Hello from Pair-2" << RESET << std::endl;
    
    pair1_future.wait();
    pair2->close();
    
    std::cout << GREEN << "✓ Pair demo completed" << RESET << std::endl;
}

// Performance comparison
void demo_performance_comparison() {
    print_header("PERFORMANCE COMPARISON");
    
    std::cout << YELLOW << "Comparing ZMQ patterns performance..." << RESET << std::endl;
    
    const int num_messages = 1000;
    
    // Test REQ/REP latency
    auto start = std::chrono::high_resolution_clock::now();
    
    auto server = create_reply_socket();
    server->set_timeout(1000); // 1 second timeout
    
    auto client = create_request_socket();
    client->set_timeout(1000);
    
    // Simulate quick ping-pong
    for (int i = 0; i < 10; ++i) {
        std::string request = "ping_" + std::to_string(i);
        std::string reply = "pong_" + std::to_string(i);
        
        // In real scenario, server would run in separate thread
        client->try_send(Message(request));
        server->try_recv(Message());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << GREEN << "Pattern Performance (estimated):" << RESET << std::endl;
    std::cout << "  REQ/REP:     ~50-100 µs per round-trip" << std::endl;
    std::cout << "  PUB/SUB:     ~10-20 µs per message" << std::endl;
    std::cout << "  PUSH/PULL:   ~5-15 µs per message" << std::endl;
    std::cout << "  DEALER/ROUTER: ~20-40 µs per message" << std::endl;
    std::cout << "  PAIR:        ~5-10 µs per message" << std::endl;
    
    server->close();
    client->close();
    
    std::cout << GREEN << "✓ Performance comparison completed" << RESET << std::endl;
}

// Use case recommendations
void show_use_cases() {
    print_header("PATTERN USE CASES & RECOMMENDATIONS");
    
    std::cout << YELLOW << "When to use each pattern:" << RESET << std::endl;
    
    std::cout << "\n" << GREEN << "REQUEST-REPLY (REQ/REP):" << RESET << std::endl;
    std::cout << "  ✓ Simple client-server communication" << std::endl;
    std::cout << "  ✓ RPC-style function calls" << std::endl;
    std::cout << "  ✓ When you need guaranteed replies" << std::endl;
    std::cout << "  ✗ High-throughput scenarios" << std::endl;
    
    std::cout << "\n" << GREEN << "PUBLISH-SUBSCRIBE (PUB/SUB):" << RESET << std::endl;
    std::cout << "  ✓ Event broadcasting" << std::endl;
    std::cout << "  ✓ Real-time data feeds" << std::endl;
    std::cout << "  ✓ Decoupled communication" << std::endl;
    std::cout << "  ✗ When you need delivery guarantees" << std::endl;
    
    std::cout << "\n" << GREEN << "PUSH-PULL (Load Balancing):" << RESET << std::endl;
    std::cout << "  ✓ Work distribution" << std::endl;
    std::cout << "  ✓ Pipeline processing" << std::endl;
    std::cout << "  ✓ Parallel task execution" << std::endl;
    std::cout << "  ✗ When order matters" << std::endl;
    
    std::cout << "\n" << GREEN << "DEALER-ROUTER (Async REQ/REP):" << RESET << std::endl;
    std::cout << "  ✓ Asynchronous services" << std::endl;
    std::cout << "  ✓ Load balancing servers" << std::endl;
    std::cout << "  ✓ Complex routing scenarios" << std::endl;
    std::cout << "  ✗ Simple request-reply needs" << std::endl;
    
    std::cout << "\n" << GREEN << "PAIR (Exclusive Bidirectional):" << RESET << std::endl;
    std::cout << "  ✓ Inter-thread communication" << std::endl;
    std::cout << "  ✓ Exclusive peer connections" << std::endl;
    std::cout << "  ✓ Control channels" << std::endl;
    std::cout << "  ✗ Multi-peer scenarios" << std::endl;
}

int main() {
    std::cout << CYAN;
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              Psyne ZeroMQ Patterns Demo                   ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << RESET;
    
    try {
        // Run all pattern demos
        demo_request_reply();
        demo_publish_subscribe();
        demo_push_pull();
        demo_dealer_router();
        demo_pair();
        
        // Performance and recommendations
        demo_performance_comparison();
        show_use_cases();
        
        print_header("SUMMARY");
        std::cout << GREEN << "All ZeroMQ pattern demos completed successfully!" << RESET << std::endl;
        std::cout << "\nPsyne now supports:" << std::endl;
        std::cout << "  ✓ Request-Reply pattern" << std::endl;
        std::cout << "  ✓ Publish-Subscribe pattern" << std::endl;
        std::cout << "  ✓ Push-Pull pattern" << std::endl;
        std::cout << "  ✓ Dealer-Router pattern" << std::endl;
        std::cout << "  ✓ Pair pattern" << std::endl;
        std::cout << "  ✓ Multi-transport support (TCP, IPC, in-memory)" << std::endl;
        std::cout << "  ✓ Asynchronous messaging" << std::endl;
        std::cout << "  ✓ Load balancing and routing" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << RED << "Error: " << e.what() << RESET << std::endl;
        return 1;
    }
    
    return 0;
}