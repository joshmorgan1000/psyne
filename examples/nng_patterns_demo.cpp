/**
 * @file nng_patterns_demo.cpp
 * @brief Comprehensive demonstration of Nanomsg/NNG patterns
 * 
 * This example showcases all three NNG patterns implemented in Psyne:
 * 1. Pipeline Pattern - Work distribution and load balancing
 * 2. Survey Pattern - Distributed queries and response collection
 * 3. Bus Pattern - Multi-way peer-to-peer mesh communication
 * 
 * Usage:
 *   Demo all patterns: ./nng_patterns_demo
 *   Pipeline demo:     ./nng_patterns_demo pipeline
 *   Survey demo:       ./nng_patterns_demo survey  
 *   Bus mesh demo:     ./nng_patterns_demo bus
 *   Distributed sim:   ./nng_patterns_demo distributed
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include <psyne/psyne.hpp>
#include "../src/patterns/nng_patterns.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include <random>
#include <atomic>
#include <future>

using namespace psyne::patterns::nng;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << " " << title << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

void print_usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "  Demo all patterns: ./nng_patterns_demo" << std::endl;
    std::cout << "  Pipeline demo:     ./nng_patterns_demo pipeline" << std::endl;
    std::cout << "  Survey demo:       ./nng_patterns_demo survey" << std::endl;
    std::cout << "  Bus mesh demo:     ./nng_patterns_demo bus" << std::endl;
    std::cout << "  Distributed sim:   ./nng_patterns_demo distributed" << std::endl;
    std::cout << std::endl;
    std::cout << "NNG Patterns demonstrated:" << std::endl;
    std::cout << "  Pipeline - Distributed work processing with load balancing" << std::endl;
    std::cout << "  Survey   - Distributed queries with response collection" << std::endl;
    std::cout << "  Bus      - Multi-way peer-to-peer mesh networking" << std::endl;
}

/**
 * @brief Demonstrate Pipeline pattern for work distribution
 */
void demo_pipeline_pattern() {
    print_separator("NNG Pipeline Pattern Demo - Work Distribution");
    
    std::cout << "The Pipeline pattern distributes work across multiple workers." << std::endl;
    std::cout << "PUSH sockets send work items, PULL sockets receive and process them." << std::endl;
    std::cout << "Features: Load balancing, fault tolerance, scalable processing." << std::endl;
    
    try {
        // Configuration
        const std::string distributor_address = "tcp://127.0.0.1:5555";
        const int num_workers = 3;
        const int work_items = 20;
        
        std::cout << "\nCreating work distributor (PUSH socket)..." << std::endl;
        
        // Create work distributor
        WorkDistributor distributor(distributor_address);
        
        std::cout << "Starting " << num_workers << " worker processes (PULL sockets)..." << std::endl;
        
        // Create workers
        std::vector<std::unique_ptr<WorkProcessor>> workers;
        std::vector<std::future<void>> worker_futures;
        std::atomic<int> work_completed{0};
        
        for (int i = 0; i < num_workers; ++i) {
            auto worker = std::make_unique<WorkProcessor>(
                distributor_address,
                [i, &work_completed](const NNGMessage& work) {
                    std::string work_data = work.to_string();
                    std::cout << "Worker " << i << " processing: " << work_data << std::endl;
                    
                    // Simulate processing time
                    std::this_thread::sleep_for(std::chrono::milliseconds(100 + (i * 50)));
                    
                    work_completed.fetch_add(1);
                    return true;
                }
            );
            
            worker->start_processing();
            workers.push_back(std::move(worker));
        }
        
        // Give workers time to connect
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        std::cout << "\nDistributing " << work_items << " work items..." << std::endl;
        
        // Submit work items
        auto start_time = std::chrono::steady_clock::now();
        
        for (int i = 0; i < work_items; ++i) {
            std::string work_data = "Task_" + std::to_string(i) + "_ProcessMe";
            
            if (distributor.submit_work(work_data)) {
                std::cout << "Submitted: " << work_data << std::endl;
            } else {
                std::cout << "Failed to submit: " << work_data << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        std::cout << "\nWaiting for all work to complete..." << std::endl;
        
        // Wait for completion
        while (work_completed.load() < work_items) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "Progress: " << work_completed.load() << "/" << work_items << " completed" << std::endl;
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\n=== Pipeline Demo Results ===" << std::endl;
        std::cout << "Total work items: " << work_items << std::endl;
        std::cout << "Number of workers: " << num_workers << std::endl;
        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Average time per item: " << (duration.count() / work_items) << " ms" << std::endl;
        std::cout << "Theoretical speedup: " << num_workers << "x" << std::endl;
        
        // Display worker statistics
        for (int i = 0; i < num_workers; ++i) {
            std::cout << "Worker " << i << " processed: " << workers[i]->work_processed() 
                      << " items, failed: " << workers[i]->work_failed() << std::endl;
        }
        
        // Cleanup
        for (auto& worker : workers) {
            worker->stop_processing();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Pipeline demo error: " << e.what() << std::endl;
    }
}

/**
 * @brief Demonstrate Survey pattern for distributed queries
 */
void demo_survey_pattern() {
    print_separator("NNG Survey Pattern Demo - Distributed Queries");
    
    std::cout << "The Survey pattern allows one surveyor to query multiple respondents." << std::endl;
    std::cout << "Useful for: Health checks, consensus protocols, distributed monitoring." << std::endl;
    
    try {
        // Configuration
        const std::string survey_address = "tcp://127.0.0.1:5556";
        const int num_respondents = 4;
        
        std::cout << "\nCreating query engine (SURVEYOR socket)..." << std::endl;
        
        // Create query engine
        QueryEngine query_engine(survey_address);
        
        std::cout << "Starting " << num_respondents << " respondent services..." << std::endl;
        
        // Create respondents with different behaviors
        std::vector<std::unique_ptr<QueryResponder>> respondents;
        
        for (int i = 0; i < num_respondents; ++i) {
            auto respondent = std::make_unique<QueryResponder>(
                survey_address,
                [i](const NNGMessage& query) {
                    std::string question = query.to_string();
                    std::cout << "Respondent " << i << " received query: " << question << std::endl;
                    
                    // Different response behaviors
                    if (question == "health_check") {
                        return NNGMessage("OK_" + std::to_string(i) + "_healthy");
                    } else if (question == "get_load") {
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_real_distribution<> dis(0.1, 0.9);
                        double load = dis(gen);
                        return NNGMessage(std::to_string(load));
                    } else if (question == "get_version") {
                        return NNGMessage("v1.2." + std::to_string(i));
                    } else if (question == "ping") {
                        return NNGMessage("pong_from_" + std::to_string(i));
                    }
                    
                    return NNGMessage("unknown_query");
                }
            );
            
            respondent->start_responding();
            respondents.push_back(std::move(respondent));
        }
        
        // Give respondents time to connect
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        std::cout << "\n=== Conducting Surveys ===" << std::endl;
        
        // Survey 1: Health check
        std::cout << "\n1. Health Check Survey:" << std::endl;
        auto health_responses = query_engine.query_all_string("health_check", std::chrono::milliseconds{1000});
        
        std::cout << "Health check responses (" << health_responses.size() << "/" << num_respondents << "):" << std::endl;
        for (const auto& response : health_responses) {
            std::cout << "  - " << response << std::endl;
        }
        
        // Survey 2: Load monitoring
        std::cout << "\n2. Load Monitoring Survey:" << std::endl;
        auto load_responses = query_engine.query_all_string("get_load", std::chrono::milliseconds{1000});
        
        double total_load = 0.0;
        std::cout << "Load responses:" << std::endl;
        for (const auto& response : load_responses) {
            try {
                double load = std::stod(response);
                total_load += load;
                std::cout << "  - Load: " << std::fixed << std::setprecision(2) << load << std::endl;
            } catch (const std::exception&) {
                std::cout << "  - Invalid response: " << response << std::endl;
            }
        }
        
        if (!load_responses.empty()) {
            double avg_load = total_load / load_responses.size();
            std::cout << "Average system load: " << std::fixed << std::setprecision(2) << avg_load << std::endl;
        }
        
        // Survey 3: Version information
        std::cout << "\n3. Version Information Survey:" << std::endl;
        auto version_responses = query_engine.query_all_string("get_version", std::chrono::milliseconds{1000});
        
        std::cout << "Version responses:" << std::endl;
        for (const auto& response : version_responses) {
            std::cout << "  - " << response << std::endl;
        }
        
        // Survey 4: Ping test with timing
        std::cout << "\n4. Ping Response Time Test:" << std::endl;
        auto ping_start = std::chrono::steady_clock::now();
        auto ping_responses = query_engine.query_all_string("ping", std::chrono::milliseconds{500});
        auto ping_end = std::chrono::steady_clock::now();
        
        auto ping_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ping_end - ping_start);
        
        std::cout << "Ping responses (" << ping_responses.size() << "/" << num_respondents 
                  << ") in " << ping_duration.count() << " ms:" << std::endl;
        for (const auto& response : ping_responses) {
            std::cout << "  - " << response << std::endl;
        }
        
        // Aggregation examples
        std::cout << "\n=== Aggregation Examples ===" << std::endl;
        double sum_load = query_engine.sum_query("get_load", std::chrono::milliseconds{1000});
        double avg_load = query_engine.avg_query("get_load", std::chrono::milliseconds{1000});
        size_t response_count = query_engine.count_query("health_check", std::chrono::milliseconds{1000});
        
        std::cout << "Sum of loads: " << std::fixed << std::setprecision(2) << sum_load << std::endl;
        std::cout << "Average load: " << std::fixed << std::setprecision(2) << avg_load << std::endl;
        std::cout << "Active nodes: " << response_count << "/" << num_respondents << std::endl;
        
        // Display respondent statistics
        std::cout << "\n=== Respondent Statistics ===" << std::endl;
        for (int i = 0; i < num_respondents; ++i) {
            std::cout << "Respondent " << i << " handled: " << respondents[i]->queries_handled() << " queries" << std::endl;
        }
        
        // Cleanup
        for (auto& respondent : respondents) {
            respondent->stop_responding();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Survey demo error: " << e.what() << std::endl;
    }
}

/**
 * @brief Demonstrate Bus pattern for mesh networking
 */
void demo_bus_pattern() {
    print_separator("NNG Bus Pattern Demo - Mesh Network Communication");
    
    std::cout << "The Bus pattern creates a mesh network where every node can" << std::endl;
    std::cout << "communicate with every other node. Messages are automatically routed." << std::endl;
    std::cout << "Features: Peer discovery, message routing, broadcast, fault tolerance." << std::endl;
    
    try {
        // Configuration
        const int num_nodes = 4;
        const std::vector<std::string> node_addresses = {
            "tcp://127.0.0.1:5560",
            "tcp://127.0.0.1:5561", 
            "tcp://127.0.0.1:5562",
            "tcp://127.0.0.1:5563"
        };
        
        std::cout << "\nCreating " << num_nodes << " mesh nodes..." << std::endl;
        
        // Create mesh nodes
        std::vector<std::unique_ptr<MeshNode>> nodes;
        std::vector<std::vector<std::string>> node_messages(num_nodes);
        
        for (int i = 0; i < num_nodes; ++i) {
            std::string node_id = "Node_" + std::to_string(i);
            
            auto node = std::make_unique<MeshNode>(node_id, node_addresses[i]);
            
            // Set message handler for each node
            node->set_message_handler([i, &node_messages](const std::string& from_node, const NNGMessage& msg) {
                std::string message = msg.to_string();
                std::cout << "Node " << i << " received from " << from_node << ": " << message << std::endl;
                node_messages[i].push_back(from_node + ": " + message);
            });
            
            nodes.push_back(std::move(node));
        }
        
        std::cout << "Connecting nodes to form mesh network..." << std::endl;
        
        // Connect nodes to form mesh (each node connects to the next)
        for (int i = 0; i < num_nodes; ++i) {
            int next_node = (i + 1) % num_nodes;
            if (nodes[i]->join_mesh(node_addresses[next_node])) {
                std::cout << "Node " << i << " connected to Node " << next_node << std::endl;
            }
        }
        
        // Give time for mesh to stabilize
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        std::cout << "\n=== Mesh Network Operations ===" << std::endl;
        
        // 1. Broadcast messages
        std::cout << "\n1. Broadcasting messages from each node:" << std::endl;
        for (int i = 0; i < num_nodes; ++i) {
            std::string broadcast_msg = "Broadcast from Node " + std::to_string(i) + " to all peers";
            
            if (nodes[i]->broadcast(broadcast_msg)) {
                std::cout << "Node " << i << " broadcasted: " << broadcast_msg << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // 2. Peer-to-peer messages
        std::cout << "\n2. Peer-to-peer communication:" << std::endl;
        
        // Node 0 sends to Node 2
        std::string p2p_msg1 = "Direct message from Node 0 to Node 2";
        if (nodes[0]->send_to_node("Node_2", p2p_msg1)) {
            std::cout << "Node 0 -> Node 2: " << p2p_msg1 << std::endl;
        }
        
        // Node 3 sends to Node 1  
        std::string p2p_msg2 = "Direct message from Node 3 to Node 1";
        if (nodes[3]->send_to_node("Node_1", p2p_msg2)) {
            std::cout << "Node 3 -> Node 1: " << p2p_msg2 << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // 3. Network topology discovery
        std::cout << "\n3. Network topology information:" << std::endl;
        for (int i = 0; i < num_nodes; ++i) {
            auto topology = nodes[i]->get_topology();
            auto peers = nodes[i]->get_peers();
            
            std::cout << "Node " << i << " (" << topology.local_id << "):" << std::endl;
            std::cout << "  Connected peers: ";
            for (const auto& peer : peers) {
                std::cout << peer << " ";
            }
            std::cout << std::endl;
            
            std::cout << "  Reachable nodes: " << topology.reachable_peers.size() << std::endl;
        }
        
        // 4. Ping tests
        std::cout << "\n4. Network connectivity tests:" << std::endl;
        for (int i = 0; i < num_nodes; ++i) {
            for (int j = 0; j < num_nodes; ++j) {
                if (i != j) {
                    std::string target_node = "Node_" + std::to_string(j);
                    bool ping_success = nodes[i]->ping_node(target_node, std::chrono::milliseconds{200});
                    std::cout << "Node " << i << " -> Node " << j << ": " 
                              << (ping_success ? "REACHABLE" : "UNREACHABLE") << std::endl;
                }
            }
        }
        
        // 5. Node discovery
        std::cout << "\n5. Peer discovery:" << std::endl;
        for (int i = 0; i < num_nodes; ++i) {
            auto discovered = nodes[i]->discover_nodes(std::chrono::milliseconds{1000});
            std::cout << "Node " << i << " discovered " << discovered.size() << " peers: ";
            for (const auto& peer : discovered) {
                std::cout << peer << " ";
            }
            std::cout << std::endl;
        }
        
        // 6. Message statistics
        std::cout << "\n=== Message Reception Summary ===" << std::endl;
        for (int i = 0; i < num_nodes; ++i) {
            std::cout << "Node " << i << " received " << node_messages[i].size() << " messages:" << std::endl;
            for (const auto& msg : node_messages[i]) {
                std::cout << "  - " << msg << std::endl;
            }
        }
        
        // Cleanup
        std::cout << "\nShutting down mesh network..." << std::endl;
        for (auto& node : nodes) {
            node->leave_mesh();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Bus demo error: " << e.what() << std::endl;
    }
}

/**
 * @brief Demonstrate distributed system simulation using all patterns
 */
void demo_distributed_system() {
    print_separator("Distributed System Simulation - All NNG Patterns");
    
    std::cout << "This simulation demonstrates how all three NNG patterns can work together" << std::endl;
    std::cout << "in a distributed system architecture:" << std::endl;
    std::cout << "- Pipeline: Distributed data processing" << std::endl;
    std::cout << "- Survey: System monitoring and health checks" << std::endl;
    std::cout << "- Bus: Peer coordination and mesh communication" << std::endl;
    
    // This would be a comprehensive example combining all patterns
    // For brevity, we'll show the architecture design
    
    std::cout << "\n=== Distributed System Architecture ===" << std::endl;
    std::cout << "┌─────────────┐    ┌─────────────┐    ┌─────────────┐" << std::endl;
    std::cout << "│   Client    │    │ Load Balancer│    │   Monitor   │" << std::endl;
    std::cout << "│  (Survey)   │    │  (Pipeline)  │    │   (Survey)  │" << std::endl;
    std::cout << "└─────────────┘    └─────────────┘    └─────────────┘" << std::endl;
    std::cout << "       │                   │                   │" << std::endl;
    std::cout << "       └─────────┬─────────┼─────────┬─────────┘" << std::endl;
    std::cout << "                 │         │         │" << std::endl;
    std::cout << "        ┌─────────────┐────┴────┐─────────────┐" << std::endl;
    std::cout << "        │   Worker 1  │ Worker 2│   Worker 3  │" << std::endl;
    std::cout << "        │ (Bus+Pipeline) (Bus+Pipeline) (Bus+Pipeline)│" << std::endl;
    std::cout << "        └─────────────┴─────────┴─────────────┘" << std::endl;
    std::cout << "                      Bus Mesh Network" << std::endl;
    
    std::cout << "\nPattern Usage:" << std::endl;
    std::cout << "1. Pipeline - Load balancer distributes work to workers" << std::endl;
    std::cout << "2. Survey - Monitor queries workers for health/load status" << std::endl;
    std::cout << "3. Bus - Workers coordinate among themselves via mesh" << std::endl;
    
    std::cout << "\nBenefits of this architecture:" << std::endl;
    std::cout << "- Scalable work distribution (Pipeline)" << std::endl;
    std::cout << "- Real-time monitoring (Survey)" << std::endl;
    std::cout << "- Fault-tolerant peer communication (Bus)" << std::endl;
    std::cout << "- Automatic load balancing and failover" << std::endl;
    
    std::cout << "\nIn a real implementation, you would:" << std::endl;
    std::cout << "1. Start multiple worker processes" << std::endl;
    std::cout << "2. Connect them via bus mesh for coordination" << std::endl;
    std::cout << "3. Use pipeline for work distribution" << std::endl;
    std::cout << "4. Use survey for monitoring and health checks" << std::endl;
    std::cout << "5. Implement automatic scaling based on load" << std::endl;
}

/**
 * @brief Run all pattern demonstrations
 */
void demo_all_patterns() {
    print_separator("Psyne NNG Patterns - Complete Demonstration");
    
    std::cout << "This demo showcases all Nanomsg/NNG-compatible patterns in Psyne:" << std::endl;
    std::cout << "1. Pipeline Pattern - Scalable work distribution" << std::endl;
    std::cout << "2. Survey Pattern - Distributed query/response" << std::endl;
    std::cout << "3. Bus Pattern - Mesh network communication" << std::endl;
    std::cout << std::endl;
    std::cout << "Each pattern addresses different distributed system needs:" << std::endl;
    std::cout << "- Pipeline: Load balancing, parallel processing" << std::endl;
    std::cout << "- Survey: Monitoring, consensus, data collection" << std::endl;
    std::cout << "- Bus: P2P networking, fault tolerance, coordination" << std::endl;
    
    std::cout << "\nPress Enter to continue with demonstrations..." << std::endl;
    std::cin.get();
    
    // Run all demos
    demo_pipeline_pattern();
    
    std::cout << "\nPress Enter to continue to Survey demo..." << std::endl;
    std::cin.get();
    
    demo_survey_pattern();
    
    std::cout << "\nPress Enter to continue to Bus demo..." << std::endl;
    std::cin.get();
    
    demo_bus_pattern();
    
    std::cout << "\nPress Enter to see distributed system architecture..." << std::endl;
    std::cin.get();
    
    demo_distributed_system();
    
    print_separator("NNG Patterns Demo Complete!");
    std::cout << "All three NNG patterns have been demonstrated successfully." << std::endl;
    std::cout << "These patterns provide the foundation for building scalable," << std::endl;
    std::cout << "fault-tolerant distributed systems with Psyne." << std::endl;
}

int main(int argc, char* argv[]) {
    print_separator("Psyne NNG Patterns Demonstration");
    
    if (argc == 1) {
        // Run all demonstrations
        demo_all_patterns();
    } else {
        std::string pattern = argv[1];
        
        if (pattern == "pipeline") {
            demo_pipeline_pattern();
        } else if (pattern == "survey") {
            demo_survey_pattern();
        } else if (pattern == "bus") {
            demo_bus_pattern();
        } else if (pattern == "distributed") {
            demo_distributed_system();
        } else {
            std::cerr << "Error: Unknown pattern '" << pattern << "'" << std::endl;
            print_usage();
            return 1;
        }
    }
    
    return 0;
}