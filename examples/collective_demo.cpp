/**
 * @file collective_demo.cpp
 * @brief Demonstration of collective-like communication patterns using Psyne
 *
 * This example shows how to implement collective communication patterns
 * (broadcast, all-reduce, scatter/gather) using Psyne's basic messaging:
 * - Point-to-point messaging with coordination
 * - Multi-channel patterns for distributed operations
 * - Synchronization using message passing
 * - ML-style gradient aggregation simulation
 *
 * Note: This is a simulation of collective operations using Psyne's current API.
 * True collective operations would require additional coordination infrastructure.
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <psyne/psyne.hpp>
#include <thread>
#include <vector>

using namespace psyne;

// Simple coordination message
class CoordinationMessage : public Message<CoordinationMessage> {
public:
    static constexpr uint32_t message_type = 500;
    
    struct Data {
        uint32_t operation_id;
        uint32_t sender_rank;
        uint32_t sequence_number;
        uint32_t data_size;
        char payload[1024]; // Variable payload
    };
    
    static size_t calculate_size() noexcept {
        return sizeof(Data);
    }
    
    Data& data() { return *reinterpret_cast<Data*>(Message::data()); }
    const Data& data() const { return *reinterpret_cast<const Data*>(Message::data()); }
    
    void set_data(uint32_t op_id, uint32_t rank, uint32_t seq, const void* payload_data, size_t payload_size) {
        data().operation_id = op_id;
        data().sender_rank = rank;
        data().sequence_number = seq;
        data().data_size = static_cast<uint32_t>(std::min(payload_size, sizeof(Data::payload)));
        if (payload_data && data().data_size > 0) {
            std::memcpy(data().payload, payload_data, data().data_size);
        }
    }
};

// Simulated collective group
class SimulatedCollectiveGroup {
private:
    uint32_t rank_;
    uint32_t world_size_;
    std::vector<std::shared_ptr<Channel<CoordinationMessage>>> channels_;
    
public:
    SimulatedCollectiveGroup(uint32_t rank, uint32_t world_size) 
        : rank_(rank), world_size_(world_size) {
        
        // Create channels for communication with other ranks
        for (uint32_t r = 0; r < world_size_; ++r) {
            if (r != rank_) {
                std::string channel_name = "memory://collective_" + std::to_string(rank_) + "_to_" + std::to_string(r);
                auto channel = Channel::get_or_create<CoordinationMessage>(channel_name);
                channels_.push_back(channel);
            } else {
                channels_.push_back(nullptr); // Self channel
            }
        }
    }
    
    uint32_t rank() const { return rank_; }
    uint32_t size() const { return world_size_; }
    
    // Simple barrier simulation
    void barrier() {
        if (world_size_ == 1) return;
        
        // Send barrier message to all other ranks
        for (uint32_t r = 0; r < world_size_; ++r) {
            if (r != rank_ && channels_[r]) {
                try {
                    CoordinationMessage msg(*channels_[r]);
                    msg.set_data(999, rank_, 0, nullptr, 0); // Barrier operation ID
                    msg.send();
                } catch (...) {
                    // Ignore failures for now
                }
            }
        }
        
        // Receive barrier messages from all other ranks
        uint32_t received = 0;
        while (received < world_size_ - 1) {
            for (uint32_t r = 0; r < world_size_; ++r) {
                if (r != rank_ && channels_[r]) {
                    size_t size;
                    uint32_t type;
                    void* data = channels_[r]->receive_message(size, type);
                    if (data) {
                        received++;
                        channels_[r]->release_message(data);
                    }
                }
            }
            if (received < world_size_ - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }
    
    // Send data to specific rank
    void send_to_rank(uint32_t target_rank, const void* data, size_t size, uint32_t op_id, uint32_t seq) {
        if (target_rank >= world_size_ || target_rank == rank_ || !channels_[target_rank]) {
            return;
        }
        
        try {
            CoordinationMessage msg(*channels_[target_rank]);
            msg.set_data(op_id, rank_, seq, data, size);
            msg.send();
        } catch (...) {
            // Ignore failures
        }
    }
    
    // Receive data from specific rank
    bool receive_from_rank(uint32_t source_rank, void* data, size_t max_size, uint32_t expected_op_id) {
        if (source_rank >= world_size_ || source_rank == rank_ || !channels_[source_rank]) {
            return false;
        }
        
        size_t msg_size;
        uint32_t type;
        void* msg_data = channels_[source_rank]->receive_message(msg_size, type);
        if (msg_data) {
            CoordinationMessage temp_msg(*channels_[source_rank]);
            std::memcpy(temp_msg.Message::data(), msg_data, msg_size);
            
            if (temp_msg.data().operation_id == expected_op_id) {
                size_t copy_size = std::min(max_size, static_cast<size_t>(temp_msg.data().data_size));
                std::memcpy(data, temp_msg.data().payload, copy_size);
                channels_[source_rank]->release_message(msg_data);
                return true;
            }
            
            channels_[source_rank]->release_message(msg_data);
        }
        return false;
    }
};

// Helper function to print vectors
template <typename T>
void print_vector(const std::string &label, const std::vector<T> &vec, uint32_t rank) {
    std::cout << "[Rank " << rank << "] " << label << ": ";
    for (size_t i = 0; i < std::min(vec.size(), size_t(8)); ++i) {
        std::cout << std::setw(6) << vec[i] << " ";
    }
    if (vec.size() > 8) std::cout << "...";
    std::cout << std::endl;
}

// Demonstrate simulated broadcast operation
void demo_broadcast(std::shared_ptr<SimulatedCollectiveGroup> group) {
    const auto rank = group->rank();
    const auto size = group->size();

    std::cout << "\n=== Simulated Broadcast Demo ===" << std::endl;

    // Data to broadcast
    std::vector<float> data(8);

    if (rank == 0) {
        // Root initializes data
        std::iota(data.begin(), data.end(), 1.0f);
        std::cout << "[Rank 0] Broadcasting data..." << std::endl;
        print_vector("Original", data, rank);
        
        // Send to all other ranks
        for (uint32_t r = 1; r < size; ++r) {
            group->send_to_rank(r, data.data(), data.size() * sizeof(float), 100, 0);
        }
    } else {
        // Non-root receives data
        std::fill(data.begin(), data.end(), 0.0f);
        print_vector("Before broadcast", data, rank);
        
        // Attempt to receive from rank 0
        std::vector<float> temp_data(8);
        bool received = false;
        for (int attempt = 0; attempt < 100 && !received; ++attempt) {
            if (group->receive_from_rank(0, temp_data.data(), temp_data.size() * sizeof(float), 100)) {
                data = temp_data;
                received = true;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        if (received) {
            print_vector("After broadcast", data, rank);
        } else {
            std::cout << "[Rank " << rank << "] Failed to receive broadcast" << std::endl;
        }
    }
}

// Demonstrate simulated all-reduce operation (sum)
void demo_allreduce(std::shared_ptr<SimulatedCollectiveGroup> group) {
    const auto rank = group->rank();
    const auto size = group->size();

    std::cout << "\n=== Simulated All-Reduce Demo (Sum) ===" << std::endl;

    // Each rank contributes different values
    std::vector<float> data(4);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(rank + 1) + i * 0.1f;
    }

    print_vector("Local data", data, rank);

    if (size == 1) {
        print_vector("After all-reduce (single rank)", data, rank);
        return;
    }

    // Simple all-reduce simulation: everyone sends to rank 0, rank 0 sums and broadcasts back
    if (rank == 0) {
        // Rank 0 collects from all other ranks
        std::vector<std::vector<float>> all_data(size);
        all_data[0] = data; // Own data
        
        // Receive from other ranks
        for (uint32_t r = 1; r < size; ++r) {
            std::vector<float> temp_data(4);
            bool received = false;
            for (int attempt = 0; attempt < 100 && !received; ++attempt) {
                if (group->receive_from_rank(r, temp_data.data(), temp_data.size() * sizeof(float), 200)) {
                    all_data[r] = temp_data;
                    received = true;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        }
        
        // Sum all data
        std::vector<float> result(4, 0.0f);
        for (const auto& rank_data : all_data) {
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] += rank_data[i];
            }
        }
        
        data = result;
        print_vector("Sum result", data, rank);
        
        // Broadcast result back to all ranks
        for (uint32_t r = 1; r < size; ++r) {
            group->send_to_rank(r, result.data(), result.size() * sizeof(float), 201, 0);
        }
    } else {
        // Other ranks send their data to rank 0
        group->send_to_rank(0, data.data(), data.size() * sizeof(float), 200, 0);
        
        // Receive result from rank 0
        std::vector<float> result(4);
        bool received = false;
        for (int attempt = 0; attempt < 100 && !received; ++attempt) {
            if (group->receive_from_rank(0, result.data(), result.size() * sizeof(float), 201)) {
                data = result;
                received = true;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        if (received) {
            print_vector("After all-reduce", data, rank);
        } else {
            std::cout << "[Rank " << rank << "] Failed to receive all-reduce result" << std::endl;
        }
    }
}

// Demonstrate simulated scatter operation
void demo_scatter(std::shared_ptr<SimulatedCollectiveGroup> group) {
    const auto rank = group->rank();
    const auto size = group->size();

    std::cout << "\n=== Simulated Scatter Demo ===" << std::endl;

    const size_t chunk_size = 3;
    std::vector<int> send_data;
    std::vector<int> recv_data(chunk_size);

    if (rank == 0) {
        // Root has all data to scatter
        send_data.resize(size * chunk_size);
        std::iota(send_data.begin(), send_data.end(), 1);
        print_vector("Data to scatter", send_data, rank);
        
        // Keep own chunk
        for (size_t i = 0; i < chunk_size; ++i) {
            recv_data[i] = send_data[i];
        }
        
        // Send chunks to other ranks
        for (uint32_t r = 1; r < size; ++r) {
            std::vector<int> chunk(chunk_size);
            for (size_t i = 0; i < chunk_size; ++i) {
                chunk[i] = send_data[r * chunk_size + i];
            }
            group->send_to_rank(r, chunk.data(), chunk.size() * sizeof(int), 300, 0);
        }
    } else {
        // Other ranks receive their chunk
        bool received = false;
        for (int attempt = 0; attempt < 100 && !received; ++attempt) {
            if (group->receive_from_rank(0, recv_data.data(), recv_data.size() * sizeof(int), 300)) {
                received = true;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        if (!received) {
            std::cout << "[Rank " << rank << "] Failed to receive scatter data" << std::endl;
        }
    }

    print_vector("Received chunk", recv_data, rank);
}

// Demonstrate ML-style gradient aggregation
void demo_ml_gradient_aggregation(std::shared_ptr<SimulatedCollectiveGroup> group) {
    const auto rank = group->rank();
    const auto size = group->size();

    std::cout << "\n=== ML Gradient Aggregation Demo ===" << std::endl;

    // Simulate gradient tensors from different workers
    const size_t model_params = 100; // Reduced for demo
    std::vector<float> gradients(model_params);

    // Each worker computes different gradients (simulate with pattern)
    for (size_t i = 0; i < gradients.size(); ++i) {
        gradients[i] = static_cast<float>(rank + 1) * 0.01f + i * 0.001f;
    }

    // Calculate local gradient stats
    float local_sum = std::accumulate(gradients.begin(), gradients.end(), 0.0f);
    float local_mean = local_sum / gradients.size();

    std::cout << "[Rank " << rank << "] Local gradient mean: " << std::fixed 
              << std::setprecision(4) << local_mean << std::endl;

    if (size == 1) {
        std::cout << "[Rank " << rank << "] Single worker - no aggregation needed" << std::endl;
        return;
    }

    // Simulate all-reduce for gradients (same pattern as demo_allreduce but with different data)
    if (rank == 0) {
        // Collect gradients from all workers
        std::vector<std::vector<float>> all_gradients(size);
        all_gradients[0] = gradients;
        
        for (uint32_t r = 1; r < size; ++r) {
            std::vector<float> worker_gradients(model_params);
            bool received = false;
            for (int attempt = 0; attempt < 100 && !received; ++attempt) {
                if (group->receive_from_rank(r, worker_gradients.data(), 
                                           worker_gradients.size() * sizeof(float), 400)) {
                    all_gradients[r] = worker_gradients;
                    received = true;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        }
        
        // Average all gradients
        std::vector<float> averaged_gradients(model_params, 0.0f);
        for (const auto& worker_grads : all_gradients) {
            for (size_t i = 0; i < averaged_gradients.size(); ++i) {
                averaged_gradients[i] += worker_grads[i];
            }
        }
        for (auto& g : averaged_gradients) {
            g /= size;
        }
        
        gradients = averaged_gradients;
        
        // Broadcast averaged gradients back
        for (uint32_t r = 1; r < size; ++r) {
            group->send_to_rank(r, averaged_gradients.data(), 
                              averaged_gradients.size() * sizeof(float), 401, 0);
        }
    } else {
        // Send gradients to parameter server (rank 0)
        group->send_to_rank(0, gradients.data(), gradients.size() * sizeof(float), 400, 0);
        
        // Receive averaged gradients
        std::vector<float> averaged_gradients(model_params);
        bool received = false;
        for (int attempt = 0; attempt < 100 && !received; ++attempt) {
            if (group->receive_from_rank(0, averaged_gradients.data(), 
                                       averaged_gradients.size() * sizeof(float), 401)) {
                gradients = averaged_gradients;
                received = true;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        if (!received) {
            std::cout << "[Rank " << rank << "] Failed to receive averaged gradients" << std::endl;
        }
    }

    // Calculate global gradient stats
    float global_sum = std::accumulate(gradients.begin(), gradients.end(), 0.0f);
    float global_mean = global_sum / gradients.size();

    std::cout << "[Rank " << rank << "] Global gradient mean after aggregation: " 
              << std::fixed << std::setprecision(4) << global_mean << std::endl;

    // Simulate parameter update
    std::vector<float> parameters(model_params, 1.0f);
    const float learning_rate = 0.01f;

    for (size_t i = 0; i < parameters.size(); ++i) {
        parameters[i] -= learning_rate * gradients[i];
    }

    std::cout << "[Rank " << rank << "] Parameters updated with aggregated gradients!" << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <rank> <world_size>" << std::endl;
    std::cout << "Example: " << program_name << " 0 3" << std::endl;
    std::cout << "Note: This demo simulates collective operations using in-memory channels." << std::endl;
    std::cout << "      For true distributed operations, use network channels." << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse command line arguments
    const uint32_t rank = std::stoi(argv[1]);
    const uint32_t world_size = std::stoi(argv[2]);

    if (rank >= world_size) {
        std::cerr << "Rank must be less than world_size" << std::endl;
        return 1;
    }

    std::cout << "=== Collective Operations Simulation ===" << std::endl;
    std::cout << "Rank: " << rank << " / " << world_size << std::endl;
    std::cout << "Note: Using simulated collective operations with memory channels" << std::endl;

    try {
        // Create simulated collective group
        auto group = std::make_shared<SimulatedCollectiveGroup>(rank, world_size);

        // Wait a bit for initialization
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

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

        demo_ml_gradient_aggregation(group);
        group->barrier();

        std::cout << "\n[Rank " << rank << "] All demos completed successfully!" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "[Rank " << rank << "] Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}