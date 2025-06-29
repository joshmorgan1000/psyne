/**
 * @file collective_demo.cpp
 * @brief Demonstration of collective communication operations
 * 
 * This example shows how to use psyne's collective operations for
 * distributed computing, including broadcast, all-reduce, scatter/gather.
 * 
 * To run this demo:
 * 1. Start multiple instances with different ranks
 * 2. Each instance needs the same peer URIs list
 * 
 * Example for 3 processes:
 *   ./collective_demo 0 3 tcp://localhost:5000 tcp://localhost:5001 tcp://localhost:5002
 *   ./collective_demo 1 3 tcp://localhost:5000 tcp://localhost:5001 tcp://localhost:5002
 *   ./collective_demo 2 3 tcp://localhost:5000 tcp://localhost:5001 tcp://localhost:5002
 */

#include <psyne/collective.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <thread>
#include <iomanip>

using namespace psyne;
using namespace psyne::collective;

// Helper function to print vectors
template<typename T>
void print_vector(const std::string& label, const std::vector<T>& vec, size_t rank) {
    std::cout << "[Rank " << rank << "] " << label << ": ";
    for (const auto& val : vec) {
        std::cout << std::setw(6) << val << " ";
    }
    std::cout << std::endl;
}

// Demonstrate broadcast operation
void demo_broadcast(std::shared_ptr<CollectiveGroup> group) {
    const auto rank = group->rank();
    const auto size = group->size();
    
    std::cout << "\n=== Broadcast Demo ===" << std::endl;
    
    // Data to broadcast
    std::vector<float> data(10);
    
    if (rank == 0) {
        // Root initializes data
        std::iota(data.begin(), data.end(), 1.0f);
        std::cout << "[Rank 0] Broadcasting data..." << std::endl;
        print_vector("Original", data, rank);
    } else {
        // Non-root has zeros
        std::fill(data.begin(), data.end(), 0.0f);
        print_vector("Before broadcast", data, rank);
    }
    
    // Execute broadcast from rank 0
    Broadcast<float> bcast(group);
    bcast.execute(std::span<float>(data), 0);
    
    // All ranks should now have the same data
    print_vector("After broadcast", data, rank);
}

// Demonstrate all-reduce operation
void demo_allreduce(std::shared_ptr<CollectiveGroup> group) {
    const auto rank = group->rank();
    const auto size = group->size();
    
    std::cout << "\n=== All-Reduce Demo (Sum) ===" << std::endl;
    
    // Each rank contributes different values
    std::vector<float> data(8);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = rank + 1.0f + i * 0.1f;
    }
    
    print_vector("Local data", data, rank);
    
    // Execute all-reduce with sum operation
    AllReduce<float> allreduce(group);
    allreduce.execute(std::span<float>(data), ReduceOp::Sum);
    
    // All ranks should have the sum
    print_vector("After all-reduce", data, rank);
    
    // Verify correctness
    float expected_sum = 0;
    for (size_t r = 0; r < size; ++r) {
        expected_sum += (r + 1.0f);
    }
    std::cout << "[Rank " << rank << "] First element should be: " 
              << expected_sum << " (actual: " << data[0] << ")" << std::endl;
}

// Demonstrate scatter operation
void demo_scatter(std::shared_ptr<CollectiveGroup> group) {
    const auto rank = group->rank();
    const auto size = group->size();
    
    std::cout << "\n=== Scatter Demo ===" << std::endl;
    
    const size_t chunk_size = 4;
    std::vector<int> send_data;
    std::vector<int> recv_data(chunk_size);
    
    if (rank == 0) {
        // Root has all data to scatter
        send_data.resize(size * chunk_size);
        std::iota(send_data.begin(), send_data.end(), 1);
        print_vector("Data to scatter", send_data, rank);
    }
    
    // Execute scatter from rank 0
    Scatter<int> scatter(group);
    scatter.execute(std::span<const int>(send_data), 
                   std::span<int>(recv_data), 0);
    
    // Each rank should have its chunk
    print_vector("Received chunk", recv_data, rank);
}

// Demonstrate gather operation
void demo_gather(std::shared_ptr<CollectiveGroup> group) {
    const auto rank = group->rank();
    const auto size = group->size();
    
    std::cout << "\n=== Gather Demo ===" << std::endl;
    
    // Each rank sends its rank ID repeated
    std::vector<int> send_data(4, rank);
    std::vector<int> recv_data;
    
    if (rank == 0) {
        recv_data.resize(size * send_data.size());
    }
    
    print_vector("Sending", send_data, rank);
    
    // Execute gather to rank 0
    Gather<int> gather(group);
    gather.execute(std::span<const int>(send_data),
                  std::span<int>(recv_data), 0);
    
    if (rank == 0) {
        print_vector("Gathered data", recv_data, rank);
    }
}

// Demonstrate all-gather operation
void demo_allgather(std::shared_ptr<CollectiveGroup> group) {
    const auto rank = group->rank();
    const auto size = group->size();
    
    std::cout << "\n=== All-Gather Demo ===" << std::endl;
    
    // Each rank contributes its rank + 10
    std::vector<double> send_data = { 
        static_cast<double>(rank + 10), 
        static_cast<double>(rank + 20) 
    };
    std::vector<double> recv_data(size * send_data.size());
    
    print_vector("Local contribution", send_data, rank);
    
    // Execute all-gather
    AllGather<double> allgather(group);
    allgather.execute(std::span<const double>(send_data),
                     std::span<double>(recv_data));
    
    print_vector("All-gathered data", recv_data, rank);
}

// Performance benchmark for all-reduce
void benchmark_allreduce(std::shared_ptr<CollectiveGroup> group) {
    const auto rank = group->rank();
    const auto size = group->size();
    
    std::cout << "\n=== All-Reduce Performance Benchmark ===" << std::endl;
    
    // Test different data sizes
    std::vector<size_t> test_sizes = {1024, 16384, 262144, 1048576}; // 1KB to 1MB
    
    for (auto data_size : test_sizes) {
        std::vector<float> data(data_size / sizeof(float), 1.0f);
        
        // Warm up
        AllReduce<float> allreduce(group);
        allreduce.execute(std::span<float>(data), ReduceOp::Sum);
        
        // Benchmark
        const int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            allreduce.execute(std::span<float>(data), ReduceOp::Sum);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (rank == 0) {
            double avg_time = duration.count() / static_cast<double>(iterations);
            double bandwidth = (data_size * size * 2) / avg_time; // MB/s (multiply by 2 for reduce+broadcast)
            
            std::cout << "Data size: " << std::setw(8) << data_size << " bytes, "
                     << "Avg time: " << std::setw(8) << std::fixed << std::setprecision(2) 
                     << avg_time << " µs, "
                     << "Bandwidth: " << std::setw(8) << bandwidth << " MB/s" 
                     << std::endl;
        }
    }
}

// Demonstrate ML-style gradient aggregation
void demo_ml_gradient_aggregation(std::shared_ptr<CollectiveGroup> group) {
    const auto rank = group->rank();
    const auto size = group->size();
    
    std::cout << "\n=== ML Gradient Aggregation Demo ===" << std::endl;
    
    // Simulate gradient tensors from different workers
    const size_t model_params = 1000;
    std::vector<float> gradients(model_params);
    
    // Each worker computes different gradients (simulate with random values)
    std::srand(rank + 1);
    for (auto& g : gradients) {
        g = (std::rand() / static_cast<float>(RAND_MAX)) * 0.1f - 0.05f;
    }
    
    // Calculate local gradient stats
    float local_sum = std::accumulate(gradients.begin(), gradients.end(), 0.0f);
    float local_mean = local_sum / gradients.size();
    
    std::cout << "[Rank " << rank << "] Local gradient mean: " << local_mean << std::endl;
    
    // All-reduce to average gradients across all workers
    AllReduce<float> allreduce(group);
    allreduce.execute(std::span<float>(gradients), ReduceOp::Sum);
    
    // Divide by number of workers to get average
    for (auto& g : gradients) {
        g /= size;
    }
    
    // Calculate global gradient stats
    float global_sum = std::accumulate(gradients.begin(), gradients.end(), 0.0f);
    float global_mean = global_sum / gradients.size();
    
    std::cout << "[Rank " << rank << "] Global gradient mean after aggregation: " 
              << global_mean << std::endl;
    
    // Simulate parameter update
    std::vector<float> parameters(model_params, 1.0f);
    const float learning_rate = 0.01f;
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        parameters[i] -= learning_rate * gradients[i];
    }
    
    if (rank == 0) {
        std::cout << "Parameters updated with aggregated gradients!" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <rank> <world_size> <uri1> <uri2> ..." << std::endl;
        std::cerr << "Example: " << argv[0] << " 0 3 tcp://localhost:5000 tcp://localhost:5001 tcp://localhost:5002" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    const CollectiveGroup::RankId rank = std::stoi(argv[1]);
    const size_t world_size = std::stoi(argv[2]);
    
    if (argc - 3 != static_cast<int>(world_size)) {
        std::cerr << "Number of URIs must match world_size" << std::endl;
        return 1;
    }
    
    std::vector<std::string> peer_uris;
    for (int i = 3; i < argc; ++i) {
        peer_uris.push_back(argv[i]);
    }
    
    std::cout << "=== Collective Operations Demo ===" << std::endl;
    std::cout << "Rank: " << rank << " / " << world_size << std::endl;
    std::cout << "URIs: ";
    for (const auto& uri : peer_uris) {
        std::cout << uri << " ";
    }
    std::cout << std::endl;
    
    try {
        // Create collective group
        auto group = create_collective_group(rank, peer_uris, "ring");
        
        // Wait for all processes to start
        std::cout << "Waiting for all processes to connect..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Synchronize all processes
        std::cout << "Synchronizing..." << std::endl;
        group->barrier();
        
        // Run demos
        demo_broadcast(group);
        group->barrier();
        
        demo_allreduce(group);
        group->barrier();
        
        demo_scatter(group);
        group->barrier();
        
        demo_gather(group);
        group->barrier();
        
        demo_allgather(group);
        group->barrier();
        
        benchmark_allreduce(group);
        group->barrier();
        
        demo_ml_gradient_aggregation(group);
        group->barrier();
        
        std::cout << "\n[Rank " << rank << "] All demos completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}