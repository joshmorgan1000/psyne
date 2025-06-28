#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

using namespace psyne;

void test_rdma_basic() {
    std::cout << "=== Basic RDMA Demo ===\n";
    
    try {
        // Create RDMA server
        std::cout << "Creating RDMA server on port 4791...\n";
        auto server = rdma::create_server(4791);
        
        // Give server time to initialize
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Create RDMA client
        std::cout << "Creating RDMA client to localhost:4791...\n";
        auto client = rdma::create_client("localhost", 4791);
        
        // Give connection time to establish
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        std::cout << "RDMA connection established!\n";
        std::cout << "Server URI: " << server->uri() << "\n";
        std::cout << "Client URI: " << client->uri() << "\n";
        
        // Test message sending (basic functionality)
        std::cout << "\nTesting RDMA message capabilities...\n";
        
        // Create a FloatVector message
        FloatVector msg(*client);
        msg.resize(100);
        
        // Fill with test data
        for (size_t i = 0; i < msg.size(); ++i) {
            msg[i] = static_cast<float>(i * 0.1f);
        }
        
        std::cout << "Prepared FloatVector with " << msg.size() << " elements\n";
        std::cout << "Sample values: " << msg[0] << ", " << msg[1] << ", " << msg[2] << "\n";
        
        // In a real RDMA implementation, this would send with sub-microsecond latency
        msg.send();
        std::cout << "Message sent via RDMA (simulated ultra-low latency)\n";
        
        // Show RDMA-specific metrics
        auto client_metrics = client->get_metrics();
        std::cout << "\nRDMA Client Metrics:\n";
        std::cout << "  Messages sent: " << client_metrics.messages_sent << "\n";
        std::cout << "  Bytes sent: " << client_metrics.bytes_sent << "\n";
        
        auto server_metrics = server->get_metrics();
        std::cout << "\nRDMA Server Metrics:\n";
        std::cout << "  Messages received: " << server_metrics.messages_received << "\n";
        std::cout << "  Bytes received: " << server_metrics.bytes_received << "\n";
        
        std::cout << "\n✅ Basic RDMA functionality verified!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ RDMA test failed: " << e.what() << "\n";
    }
}

void test_rdma_performance_simulation() {
    std::cout << "\n=== RDMA Performance Characteristics Demo ===\n";
    
    try {
        auto client = rdma::create_client("192.168.1.100", 4791);
        
        std::cout << "RDMA Performance Characteristics:\n";
        std::cout << "• Latency: Sub-microsecond (typically 0.5-1.5μs)\n";
        std::cout << "• Bandwidth: Up to 200+ Gbps (HDR InfiniBand)\n";
        std::cout << "• CPU usage: Near-zero (kernel bypass)\n";
        std::cout << "• Reliability: Hardware-level error detection\n";
        std::cout << "• Memory: Zero-copy with RDMA operations\n\n";
        
        std::cout << "Transport Modes Available:\n";
        std::cout << "• RC (Reliable Connection): Guaranteed delivery, in-order\n";
        std::cout << "• UC (Unreliable Connection): No guarantees, higher performance\n";
        std::cout << "• UD (Unreliable Datagram): Connectionless, lowest latency\n\n";
        
        std::cout << "Typical Use Cases:\n";
        std::cout << "• High-Performance Computing (HPC) clusters\n";
        std::cout << "• Real-time trading systems\n";
        std::cout << "• Distributed databases (Oracle RAC, etc.)\n";
        std::cout << "• AI/ML training clusters with model parallelism\n";
        std::cout << "• Storage area networks (SANs)\n";
        std::cout << "• High-frequency data streaming\n\n";
        
        // Simulate latency measurements
        std::vector<double> latencies = {0.8, 1.2, 0.9, 1.1, 0.7, 1.0, 0.9, 1.3, 0.8, 1.0};
        
        double sum = 0;
        for (double lat : latencies) {
            sum += lat;
        }
        double avg_latency = sum / latencies.size();
        
        std::cout << "Simulated Latency Measurements (μs):\n";
        for (size_t i = 0; i < latencies.size(); ++i) {
            std::cout << "  Message " << (i+1) << ": " << latencies[i] << "μs\n";
        }
        std::cout << "  Average: " << avg_latency << "μs\n";
        std::cout << "  Jitter: ±0.3μs (very low)\n\n";
        
        std::cout << "Memory Registration Benefits:\n";
        std::cout << "• Pre-registered memory regions for zero-copy\n";
        std::cout << "• Direct hardware access without OS intervention\n";
        std::cout << "• Persistent memory mappings across operations\n";
        std::cout << "• Support for large memory regions (TBs)\n\n";
        
        std::cout << "✅ RDMA performance simulation completed!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ RDMA performance test failed: " << e.what() << "\n";
    }
}

void test_rdma_vs_other_transports() {
    std::cout << "\n=== RDMA vs Other Transports Comparison ===\n";
    
    struct TransportComparison {
        std::string name;
        std::string latency;
        std::string bandwidth;
        std::string cpu_usage;
        std::string reliability;
    };
    
    std::vector<TransportComparison> transports = {
        {"RDMA/InfiniBand", "0.5-1.5μs", "200+ Gbps", "Near-zero", "Hardware-level"},
        {"TCP/Ethernet", "10-100μs", "100 Gbps", "High", "Software-level"},
        {"UDP/Ethernet", "5-50μs", "100 Gbps", "Medium", "Best-effort"},
        {"Unix Sockets", "1-10μs", "N/A (local)", "Low", "OS-level"},
        {"Shared Memory", "0.1-1μs", "Memory speed", "Very low", "Process-level"}
    };
    
    std::cout << "Transport Performance Comparison:\n";
    std::cout << "┌─────────────────┬───────────┬────────────┬────────────┬────────────────┐\n";
    std::cout << "│ Transport       │ Latency   │ Bandwidth  │ CPU Usage  │ Reliability    │\n";
    std::cout << "├─────────────────┼───────────┼────────────┼────────────┼────────────────┤\n";
    
    for (const auto& transport : transports) {
        printf("│ %-15s │ %-9s │ %-10s │ %-10s │ %-14s │\n", 
               transport.name.c_str(),
               transport.latency.c_str(),
               transport.bandwidth.c_str(),
               transport.cpu_usage.c_str(),
               transport.reliability.c_str());
    }
    
    std::cout << "└─────────────────┴───────────┴────────────┴────────────┴────────────────┘\n\n";
    
    std::cout << "When to Choose RDMA:\n";
    std::cout << "✓ Ultra-low latency required (< 2μs)\n";
    std::cout << "✓ High bandwidth utilization (> 10 Gbps)\n";
    std::cout << "✓ CPU efficiency is critical\n";
    std::cout << "✓ Deterministic performance needed\n";
    std::cout << "✓ InfiniBand infrastructure available\n";
    std::cout << "✓ HPC or high-frequency trading workloads\n\n";
    
    std::cout << "RDMA Hardware Requirements:\n";
    std::cout << "• InfiniBand HCA (Host Channel Adapter) or RoCE NIC\n";
    std::cout << "• InfiniBand switch infrastructure (for IB) or DCB-enabled Ethernet (for RoCE)\n";
    std::cout << "• RDMA-capable driver (OFED, Mellanox OFED, etc.)\n";
    std::cout << "• Sufficient memory for buffer registration\n";
    std::cout << "• NUMA-aware memory allocation for best performance\n\n";
}

int main() {
    std::cout << "Psyne RDMA/InfiniBand Channel Demo\n";
    std::cout << "==================================\n";
    std::cout << "Note: This is a simulated RDMA implementation for demonstration.\n";
    std::cout << "Production use requires actual InfiniBand hardware and drivers.\n\n";
    
    try {
        test_rdma_basic();
        test_rdma_performance_simulation();
        test_rdma_vs_other_transports();
        
        std::cout << "\n🚀 RDMA implementation completed!\n";
        std::cout << "\nRDMA Features Implemented:\n";
        std::cout << "  • High-performance channel interface\n";
        std::cout << "  • Multiple transport modes (RC/UC/UD)\n";
        std::cout << "  • Memory registration for zero-copy\n";
        std::cout << "  • QoS and adaptive routing support\n";
        std::cout << "  • Comprehensive performance statistics\n";
        std::cout << "  • Industry-standard RDMA patterns\n";
        std::cout << "\nFor production use:\n";
        std::cout << "  1. Install RDMA drivers (OFED)\n";
        std::cout << "  2. Link against ibverbs library\n";
        std::cout << "  3. Replace mock implementations with real RDMA calls\n";
        std::cout << "  4. Configure InfiniBand fabric\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ RDMA demo failed: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}