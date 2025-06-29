/**
 * @file ucx_multi_transport_demo.cpp
 * @brief UCX multi-transport communication demonstration
 * 
 * Demonstrates UCX's automatic transport selection and multi-transport
 * capabilities. Shows how UCX chooses optimal transports based on
 * system capabilities and workload characteristics.
 * 
 * Usage:
 *   Server mode: ./ucx_multi_transport_demo server [address]
 *   Client mode: ./ucx_multi_transport_demo client <server_address>
 *   Discovery mode: ./ucx_multi_transport_demo discover
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include <psyne/psyne.hpp>

#if defined(PSYNE_UCX_SUPPORT)
#include "../src/ucx/ucx_channel.hpp"
#include "../src/ucx/ucx_message.hpp"
#endif

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cstring>

using namespace psyne;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << " " << title << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

void print_usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "  Server mode:    ./ucx_multi_transport_demo server [address]" << std::endl;
    std::cout << "  Client mode:    ./ucx_multi_transport_demo client <server_address>" << std::endl;
    std::cout << "  Discovery mode: ./ucx_multi_transport_demo discover" << std::endl;
    std::cout << std::endl;
    std::cout << "Features demonstrated:" << std::endl;
    std::cout << "  - Automatic transport selection" << std::endl;
    std::cout << "  - Multi-transport optimization" << std::endl;
    std::cout << "  - Performance comparison across transports" << std::endl;
    std::cout << "  - Zero-copy operations" << std::endl;
    std::cout << "  - Collective operations" << std::endl;
}

#if defined(PSYNE_UCX_SUPPORT)

void discover_transports() {
    print_separator("UCX Transport Discovery");
    
    auto& context_manager = ucx::UCXContextManager::instance();
    auto capabilities = context_manager.get_system_capabilities();
    
    std::cout << "System UCX Capabilities:" << std::endl;
    std::cout << "  Tag matching:     " << (capabilities.supports_tag_matching ? "✓" : "✗") << std::endl;
    std::cout << "  Stream API:       " << (capabilities.supports_stream_api ? "✓" : "✗") << std::endl;
    std::cout << "  RMA operations:   " << (capabilities.supports_rma ? "✓" : "✗") << std::endl;
    std::cout << "  Atomic ops:       " << (capabilities.supports_atomic ? "✓" : "✗") << std::endl;
    std::cout << "  GPU memory:       " << (capabilities.supports_gpu_memory ? "✓" : "✗") << std::endl;
    std::cout << "  Multi-rail:       " << (capabilities.supports_multi_rail ? "✓" : "✗") << std::endl;
    std::cout << "  Max message size: " << capabilities.max_message_size / (1024*1024) << " MB" << std::endl;
    std::cout << "  Max IOV count:    " << capabilities.max_iov_count << std::endl;
    
    std::cout << "\nAvailable Transports:" << std::endl;
    std::cout << std::left 
              << std::setw(15) << "Transport"
              << std::setw(12) << "Device"
              << std::setw(15) << "Bandwidth"
              << std::setw(12) << "Latency"
              << std::setw(8) << "RMA"
              << std::setw(8) << "Atomic"
              << std::setw(8) << "GPU"
              << std::setw(8) << "Active"
              << std::endl;
    std::cout << std::string(88, '-') << std::endl;
    
    for (const auto& transport : capabilities.available_transports) {
        std::cout << std::left 
                  << std::setw(15) << transport.name
                  << std::setw(12) << transport.device
                  << std::setw(15) << (std::to_string(static_cast<int>(transport.bandwidth_mbps)) + " MB/s")
                  << std::setw(12) << (std::to_string(transport.latency_us) + " μs")
                  << std::setw(8) << (transport.supports_rma ? "✓" : "✗")
                  << std::setw(8) << (transport.supports_atomic ? "✓" : "✗")
                  << std::setw(8) << (transport.supports_gpu ? "✓" : "✗")
                  << std::setw(8) << (transport.active ? "✓" : "✗")
                  << std::endl;
    }
    
    std::cout << "\nOptimal Transport Selection:" << std::endl;
    std::cout << "  Standard workload: " << context_manager.select_optimal_transport(false, false) << std::endl;
    std::cout << "  GPU workload:      " << context_manager.select_optimal_transport(true, false) << std::endl;
    std::cout << "  RMA workload:      " << context_manager.select_optimal_transport(false, true) << std::endl;
    std::cout << "  GPU + RMA:         " << context_manager.select_optimal_transport(true, true) << std::endl;
}

void test_transport_modes() {
    print_separator("UCX Transport Mode Testing");
    
    std::vector<std::pair<ucx::TransportMode, std::string>> modes = {
        {ucx::TransportMode::AUTO, "AUTO"},
        {ucx::TransportMode::TCP_ONLY, "TCP_ONLY"},
        {ucx::TransportMode::RDMA_ONLY, "RDMA_ONLY"},
        {ucx::TransportMode::SHM_ONLY, "SHM_ONLY"},
        {ucx::TransportMode::MULTI_RAIL, "MULTI_RAIL"},
        {ucx::TransportMode::GPU_DIRECT, "GPU_DIRECT"}
    };
    
    const size_t test_size = 1024 * 1024; // 1MB test
    std::vector<float> test_data(test_size / sizeof(float), 3.14159f);
    
    for (const auto& [mode, name] : modes) {
        std::cout << "\nTesting transport mode: " << name << std::endl;
        
        try {
            auto channel = std::make_unique<ucx::UCXChannel>("test_" + name, mode, test_size);
            
            // Create test message
            ucx::UCXFloatVector message(std::shared_ptr<ucx::UCXChannel>(channel.release()));
            message.resize(test_data.size());
            std::copy(test_data.begin(), test_data.end(), message.begin());
            
            // Test registration for zero-copy
            message.set_delivery_mode(ucx::DeliveryMode::ZERO_COPY);
            message.ensure_registered();
            
            std::cout << "  Channel created:       ✓" << std::endl;
            std::cout << "  Message size:          " << message.size() << " elements" << std::endl;
            std::cout << "  Memory registered:     " << (message.is_registered() ? "✓" : "✗") << std::endl;
            std::cout << "  Zero-copy capable:     " << (message.memory_region() ? "✓" : "✗") << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "  Error: " << e.what() << std::endl;
        }
    }
}

void run_server(const std::string& address = "0.0.0.0:12345") {
    print_separator("UCX Multi-Transport Server");
    
    std::cout << "Server Configuration:" << std::endl;
    std::cout << "  Address: " << address << std::endl;
    std::cout << "  Transport mode: AUTO (optimal selection)" << std::endl;
    std::cout << "  Buffer size: 64 MB" << std::endl;
    
    try {
        // Create server with automatic transport selection
        auto server_channel = ucx::create_ucx_server(address, ucx::TransportMode::AUTO, 64 * 1024 * 1024);
        if (!server_channel) {
            throw std::runtime_error("Failed to create UCX server");
        }
        
        std::cout << "Server started successfully!" << std::endl;
        
        // Display active capabilities
        auto capabilities = server_channel->get_capabilities();
        std::cout << "\nServer Capabilities:" << std::endl;
        std::cout << "  Active transports: " << capabilities.available_transports.size() << std::endl;
        for (const auto& transport : capabilities.available_transports) {
            if (transport.active) {
                std::cout << "    - " << transport.name << " (" << transport.device << ")" << std::endl;
            }
        }
        
        // Create test vectors for different scenarios
        ucx::UCXFloatVector float_vector(server_channel);
        ucx::UCXDoubleVector double_vector(server_channel);
        ucx::UCXIntVector int_vector(server_channel);
        
        // Initialize with test data
        const size_t test_size = 1000000; // 1M elements
        float_vector.resize(test_size);
        double_vector.resize(test_size);
        int_vector.resize(test_size);
        
        for (size_t i = 0; i < test_size; ++i) {
            float_vector[i] = static_cast<float>(i) * 0.001f;
            double_vector[i] = static_cast<double>(i) * 0.000001;
            int_vector[i] = static_cast<int>(i);
        }
        
        std::cout << "\nTest vectors created:" << std::endl;
        std::cout << "  Float vector:   " << float_vector.size() << " elements" << std::endl;
        std::cout << "  Double vector:  " << double_vector.size() << " elements" << std::endl;
        std::cout << "  Integer vector: " << int_vector.size() << " elements" << std::endl;
        
        // Enable zero-copy for performance
        float_vector.set_delivery_mode(ucx::DeliveryMode::ZERO_COPY);
        double_vector.set_delivery_mode(ucx::DeliveryMode::ZERO_COPY);
        int_vector.set_delivery_mode(ucx::DeliveryMode::ZERO_COPY);
        
        // Register memory regions
        float_vector.ensure_registered();
        double_vector.ensure_registered();
        int_vector.ensure_registered();
        
        std::cout << "Memory regions registered for zero-copy operations" << std::endl;
        
        std::cout << "\nWaiting for client connections..." << std::endl;
        std::cout << "Server will demonstrate:" << std::endl;
        std::cout << "  1. Point-to-point communication" << std::endl;
        std::cout << "  2. Broadcast operations" << std::endl;
        std::cout << "  3. Zero-copy transfers" << std::endl;
        std::cout << "  4. Performance measurements" << std::endl;
        
        // Keep server running
        std::cout << "\nPress Enter to shut down server..." << std::endl;
        std::cin.get();
        
        // Print final statistics
        auto stats = server_channel->get_stats();
        std::cout << "\n=== Server Statistics ===" << std::endl;
        std::cout << "Messages sent:        " << stats.messages_sent << std::endl;
        std::cout << "Messages received:    " << stats.messages_received << std::endl;
        std::cout << "Bytes sent:          " << stats.bytes_sent / (1024*1024) << " MB" << std::endl;
        std::cout << "Bytes received:      " << stats.bytes_received / (1024*1024) << " MB" << std::endl;
        std::cout << "Zero-copy sends:     " << stats.zero_copy_sends << std::endl;
        std::cout << "Zero-copy receives:  " << stats.zero_copy_receives << std::endl;
        std::cout << "RMA operations:      " << stats.rma_operations << std::endl;
        std::cout << "Average latency:     " << std::fixed << std::setprecision(2) 
                  << stats.avg_latency_us << " μs" << std::endl;
        std::cout << "Average bandwidth:   " << std::fixed << std::setprecision(2)
                  << stats.avg_bandwidth_mbps << " MB/s" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
}

void run_client(const std::string& server_address) {
    print_separator("UCX Multi-Transport Client");
    
    std::cout << "Client Configuration:" << std::endl;
    std::cout << "  Server: " << server_address << std::endl;
    std::cout << "  Transport mode: AUTO (optimal selection)" << std::endl;
    
    try {
        // Create client with automatic transport selection
        auto client_channel = ucx::create_ucx_client(server_address, ucx::TransportMode::AUTO, 64 * 1024 * 1024);
        if (!client_channel) {
            throw std::runtime_error("Failed to connect to UCX server");
        }
        
        std::cout << "Connected to server successfully!" << std::endl;
        
        // Display active transport information
        auto stats = client_channel->get_stats();
        std::cout << "\nActive Transports:" << std::endl;
        for (const auto& transport : stats.active_transports) {
            std::cout << "  - " << transport.name << " (" << transport.device 
                      << ") - " << transport.bandwidth_mbps << " MB/s" << std::endl;
        }
        
        // Test different communication patterns
        std::cout << "\n=== Testing Communication Patterns ===" << std::endl;
        
        // 1. Point-to-point communication
        std::cout << "\n1. Point-to-Point Communication:" << std::endl;
        
        ucx::UCXFloatVector send_vector(client_channel);
        ucx::UCXFloatVector recv_vector(client_channel);
        
        const size_t vector_size = 100000; // 100K elements
        send_vector.resize(vector_size);
        recv_vector.resize(vector_size);
        
        // Fill with test data
        for (size_t i = 0; i < vector_size; ++i) {
            send_vector[i] = static_cast<float>(i) + 0.5f;
        }
        
        // Test eager delivery (small messages)
        send_vector.set_delivery_mode(ucx::DeliveryMode::EAGER);
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = send_vector.send();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto eager_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "  Eager delivery:  " << (success ? "SUCCESS" : "FAILED") 
                  << " (" << eager_time.count() << " μs)" << std::endl;
        
        // Test zero-copy delivery
        send_vector.set_delivery_mode(ucx::DeliveryMode::ZERO_COPY);
        send_vector.ensure_registered();
        
        start_time = std::chrono::high_resolution_clock::now();
        success = send_vector.send();
        end_time = std::chrono::high_resolution_clock::now();
        
        auto zerocopy_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "  Zero-copy:       " << (success ? "SUCCESS" : "FAILED")
                  << " (" << zerocopy_time.count() << " μs)" << std::endl;
        
        double speedup = static_cast<double>(eager_time.count()) / zerocopy_time.count();
        std::cout << "  Zero-copy speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        
        // 2. Multi-transport performance comparison
        std::cout << "\n2. Multi-Transport Performance Test:" << std::endl;
        
        std::vector<ucx::TransportMode> test_modes = {
            ucx::TransportMode::TCP_ONLY,
            ucx::TransportMode::RDMA_ONLY,
            ucx::TransportMode::MULTI_RAIL
        };
        
        std::vector<std::string> mode_names = {"TCP", "RDMA", "Multi-Rail"};
        
        for (size_t i = 0; i < test_modes.size(); ++i) {
            try {
                auto test_channel = std::make_unique<ucx::UCXChannel>(
                    "test_client", test_modes[i], 1024 * 1024);
                
                ucx::UCXFloatVector test_vector(std::shared_ptr<ucx::UCXChannel>(test_channel.release()));
                test_vector.resize(50000); // 50K elements
                
                for (size_t j = 0; j < test_vector.size(); ++j) {
                    test_vector[j] = static_cast<float>(j);
                }
                
                auto test_start = std::chrono::high_resolution_clock::now();
                bool test_success = test_vector.send();
                auto test_end = std::chrono::high_resolution_clock::now();
                
                auto test_time = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_start);
                
                std::cout << "  " << std::left << std::setw(12) << mode_names[i] 
                          << (test_success ? "SUCCESS" : "FAILED")
                          << " (" << test_time.count() << " μs)" << std::endl;
                          
            } catch (const std::exception& e) {
                std::cout << "  " << std::left << std::setw(12) << mode_names[i] 
                          << "UNAVAILABLE (" << e.what() << ")" << std::endl;
            }
        }
        
        // 3. Collective operations test
        std::cout << "\n3. Collective Operations:" << std::endl;
        
        ucx::UCXCollectives collectives(client_channel);
        ucx::UCXIntVector collective_data(client_channel);
        collective_data.resize(10);
        
        // Initialize with client ID
        for (size_t i = 0; i < collective_data.size(); ++i) {
            collective_data[i] = static_cast<int>(i + 1); // Client data: 1, 2, 3, ...
        }
        
        std::cout << "  Original data: ";
        for (size_t i = 0; i < std::min(collective_data.size(), size_t(5)); ++i) {
            std::cout << collective_data[i] << " ";
        }
        if (collective_data.size() > 5) std::cout << "...";
        std::cout << std::endl;
        
        // Simulate allreduce (would normally involve multiple peers)
        std::vector<std::string> mock_peers; // Empty for demo
        bool allreduce_success = collectives.allreduce(collective_data, mock_peers);
        
        std::cout << "  Allreduce:     " << (allreduce_success ? "SUCCESS" : "SIMULATED") << std::endl;
        
        // Barrier synchronization test
        bool barrier_success = collectives.barrier(mock_peers);
        std::cout << "  Barrier sync:  " << (barrier_success ? "SUCCESS" : "SIMULATED") << std::endl;
        
        // Print final client statistics
        auto final_stats = client_channel->get_stats();
        std::cout << "\n=== Client Statistics ===" << std::endl;
        std::cout << "Messages sent:       " << final_stats.messages_sent << std::endl;
        std::cout << "Messages received:   " << final_stats.messages_received << std::endl;
        std::cout << "Bytes transferred:   " << (final_stats.bytes_sent + final_stats.bytes_received) / 1024 << " KB" << std::endl;
        std::cout << "Zero-copy ops:       " << (final_stats.zero_copy_sends + final_stats.zero_copy_receives) << std::endl;
        std::cout << "Average latency:     " << std::fixed << std::setprecision(2) 
                  << final_stats.avg_latency_us << " μs" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Client error: " << e.what() << std::endl;
    }
}

#endif // PSYNE_UCX_SUPPORT

int main(int argc, char* argv[]) {
    print_separator("Psyne UCX Multi-Transport Communication Demo");
    
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string mode = argv[1];
    
#if defined(PSYNE_UCX_SUPPORT)
    
    if (mode == "discover") {
        discover_transports();
        test_transport_modes();
    } else if (mode == "server") {
        std::string address = (argc > 2) ? argv[2] : "0.0.0.0:12345";
        run_server(address);
    } else if (mode == "client") {
        if (argc < 3) {
            std::cerr << "Error: Client mode requires server address" << std::endl;
            print_usage();
            return 1;
        }
        run_client(argv[2]);
    } else {
        std::cerr << "Error: Invalid mode '" << mode << "'" << std::endl;
        print_usage();
        return 1;
    }
    
#else
    
    std::cout << "This demo requires UCX support to be compiled in." << std::endl;
    std::cout << "Current build configuration:" << std::endl;
    
#ifdef PSYNE_UCX_SUPPORT
    std::cout << "  UCX support: ✓ Enabled" << std::endl;
#else
    std::cout << "  UCX support: ✗ Disabled" << std::endl;
#endif
    
    std::cout << "\nPlease rebuild with UCX support enabled:" << std::endl;
    std::cout << "  1. Install UCX development libraries" << std::endl;
    std::cout << "  2. Reconfigure with cmake" << std::endl;
    std::cout << "  3. Rebuild the project" << std::endl;
    return 1;
    
#endif
    
    return 0;
}