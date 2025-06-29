/**
 * @file ucx_collective_ops_demo.cpp
 * @brief UCX collective operations demonstration
 * 
 * Demonstrates advanced UCX collective communication patterns including:
 * - Broadcast (one-to-many)
 * - Scatter (distribute data)
 * - Gather (collect data)
 * - Allreduce (distributed reduction)
 * - Barrier synchronization
 * 
 * This example simulates a distributed machine learning scenario where
 * multiple nodes need to exchange gradients and parameters.
 * 
 * Usage:
 *   Coordinator: ./ucx_collective_ops_demo coordinator <num_workers>
 *   Worker:      ./ucx_collective_ops_demo worker <worker_id> <coordinator_address>
 *   Simulate:    ./ucx_collective_ops_demo simulate
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
#include <random>
#include <numeric>
#include <algorithm>

using namespace psyne;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << " " << title << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

void print_usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "  Coordinator: ./ucx_collective_ops_demo coordinator <num_workers>" << std::endl;
    std::cout << "  Worker:      ./ucx_collective_ops_demo worker <worker_id> <coordinator_address>" << std::endl;
    std::cout << "  Simulate:    ./ucx_collective_ops_demo simulate" << std::endl;
    std::cout << std::endl;
    std::cout << "Collective Operations Demonstrated:" << std::endl;
    std::cout << "  - Broadcast: Send model parameters to all workers" << std::endl;
    std::cout << "  - Scatter:   Distribute training data across workers" << std::endl;
    std::cout << "  - Gather:    Collect local gradients from workers" << std::endl;
    std::cout << "  - Allreduce: Distributed gradient averaging" << std::endl;
    std::cout << "  - Barrier:   Synchronization across all nodes" << std::endl;
}

#if defined(PSYNE_UCX_SUPPORT)

/**
 * @brief Simulated ML model parameters
 */
struct ModelParameters {
    std::vector<float> weights;
    std::vector<float> biases;
    float learning_rate;
    int epoch;
    
    ModelParameters(size_t num_weights = 1000, size_t num_biases = 10)
        : weights(num_weights), biases(num_biases), learning_rate(0.01f), epoch(0) {
        
        // Initialize with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        for (auto& w : weights) {
            w = dist(gen);
        }
        for (auto& b : biases) {
            b = dist(gen);
        }
    }
    
    void print_summary() const {
        std::cout << "Model Parameters:" << std::endl;
        std::cout << "  Weights: " << weights.size() << " parameters" << std::endl;
        std::cout << "  Biases:  " << biases.size() << " parameters" << std::endl;
        std::cout << "  Learning rate: " << learning_rate << std::endl;
        std::cout << "  Epoch: " << epoch << std::endl;
        
        // Show some sample values
        if (!weights.empty()) {
            std::cout << "  Sample weights: ";
            for (size_t i = 0; i < std::min(weights.size(), size_t(5)); ++i) {
                std::cout << std::fixed << std::setprecision(4) << weights[i] << " ";
            }
            if (weights.size() > 5) std::cout << "...";
            std::cout << std::endl;
        }
    }
};

/**
 * @brief Simulated training batch
 */
struct TrainingBatch {
    std::vector<std::vector<float>> inputs;
    std::vector<int> labels;
    
    TrainingBatch(size_t batch_size = 32, size_t input_dim = 100, int num_classes = 10) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> input_dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> label_dist(0, num_classes - 1);
        
        inputs.resize(batch_size);
        labels.resize(batch_size);
        
        for (size_t i = 0; i < batch_size; ++i) {
            inputs[i].resize(input_dim);
            for (auto& val : inputs[i]) {
                val = input_dist(gen);
            }
            labels[i] = label_dist(gen);
        }
    }
    
    size_t size() const { return inputs.size(); }
    size_t input_dim() const { return inputs.empty() ? 0 : inputs[0].size(); }
};

/**
 * @brief Coordinator node for distributed training
 */
class TrainingCoordinator {
public:
    TrainingCoordinator(int num_workers) 
        : num_workers_(num_workers), model_params_(1000, 10) {
        
        // Create UCX channel optimized for ML workloads
        channel_ = ucx::create_ml_ucx_channel("coordinator", true, 64 * 1024 * 1024);
        if (!channel_) {
            throw std::runtime_error("Failed to create coordinator channel");
        }
        
        collectives_ = std::make_unique<ucx::UCXCollectives>(channel_);
        
        std::cout << "Training Coordinator initialized:" << std::endl;
        std::cout << "  Number of workers: " << num_workers_ << std::endl;
        std::cout << "  Channel type: ML-optimized UCX" << std::endl;
        model_params_.print_summary();
    }
    
    void run_training_simulation() {
        print_separator("Distributed Training Simulation");
        
        const int num_epochs = 5;
        const size_t data_per_worker = 1000;
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "\n--- Epoch " << (epoch + 1) << " ---" << std::endl;
            model_params_.epoch = epoch;
            
            // 1. Broadcast model parameters to all workers
            std::cout << "1. Broadcasting model parameters..." << std::endl;
            broadcast_model_parameters();
            
            // 2. Scatter training data to workers
            std::cout << "2. Scattering training data..." << std::endl;
            scatter_training_data(data_per_worker);
            
            // 3. Barrier - wait for all workers to start training
            std::cout << "3. Synchronizing workers..." << std::endl;
            barrier_synchronization();
            
            // 4. Simulate training time
            std::cout << "4. Workers training locally..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            // 5. Gather gradients from all workers
            std::cout << "5. Gathering gradients..." << std::endl;
            gather_gradients();
            
            // 6. Allreduce for gradient averaging
            std::cout << "6. Performing allreduce for gradient averaging..." << std::endl;
            allreduce_gradients();
            
            // 7. Update model parameters
            std::cout << "7. Updating model parameters..." << std::endl;
            update_model_parameters();
            
            // 8. Final barrier
            std::cout << "8. Epoch completion barrier..." << std::endl;
            barrier_synchronization();
            
            std::cout << "Epoch " << (epoch + 1) << " completed successfully!" << std::endl;
        }
        
        print_training_summary();
    }
    
private:
    int num_workers_;
    ModelParameters model_params_;
    std::shared_ptr<ucx::UCXChannel> channel_;
    std::unique_ptr<ucx::UCXCollectives> collectives_;
    std::vector<ucx::UCXFloatVector> worker_gradients_;
    
    void broadcast_model_parameters() {
        // Create UCX vector for model weights
        ucx::UCXFloatVector weights_msg(channel_);
        weights_msg.resize(model_params_.weights.size());
        std::copy(model_params_.weights.begin(), model_params_.weights.end(), weights_msg.begin());
        
        // Create UCX vector for biases
        ucx::UCXFloatVector biases_msg(channel_);
        biases_msg.resize(model_params_.biases.size());
        std::copy(model_params_.biases.begin(), model_params_.biases.end(), biases_msg.begin());
        
        // Simulate broadcast to workers
        std::vector<std::string> worker_addresses;
        for (int i = 0; i < num_workers_; ++i) {
            worker_addresses.push_back("worker_" + std::to_string(i));
        }
        
        // Enable zero-copy for large parameter transfers
        weights_msg.set_delivery_mode(ucx::DeliveryMode::ZERO_COPY);
        biases_msg.set_delivery_mode(ucx::DeliveryMode::ZERO_COPY);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        bool weights_success = weights_msg.broadcast(worker_addresses);
        bool biases_success = biases_msg.broadcast(worker_addresses);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "  Weights broadcast: " << (weights_success ? "SUCCESS" : "SIMULATED") << std::endl;
        std::cout << "  Biases broadcast:  " << (biases_success ? "SUCCESS" : "SIMULATED") << std::endl;
        std::cout << "  Transfer time:     " << duration.count() << " μs" << std::endl;
        std::cout << "  Data size:         " << 
                     (model_params_.weights.size() + model_params_.biases.size()) * sizeof(float) / 1024 
                  << " KB" << std::endl;
    }
    
    void scatter_training_data(size_t data_per_worker) {
        // Generate training dataset
        const size_t total_samples = data_per_worker * num_workers_;
        const size_t input_dim = 100;
        
        std::vector<float> training_inputs(total_samples * input_dim);
        std::vector<int> training_labels(total_samples);
        
        // Initialize with synthetic data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> input_dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> label_dist(0, 9);
        
        for (size_t i = 0; i < training_inputs.size(); ++i) {
            training_inputs[i] = input_dist(gen);
        }
        for (size_t i = 0; i < training_labels.size(); ++i) {
            training_labels[i] = label_dist(gen);
        }
        
        // Create UCX vectors for scatter operation
        ucx::UCXFloatVector inputs_msg(channel_);
        ucx::UCXIntVector labels_msg(channel_);
        
        inputs_msg.resize(training_inputs.size());
        labels_msg.resize(training_labels.size());
        
        std::copy(training_inputs.begin(), training_inputs.end(), inputs_msg.begin());
        std::copy(training_labels.begin(), training_labels.end(), labels_msg.begin());
        
        // Simulate scatter operation
        std::vector<std::string> worker_addresses;
        for (int i = 0; i < num_workers_; ++i) {
            worker_addresses.push_back("worker_" + std::to_string(i));
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        bool inputs_success = inputs_msg.scatter(worker_addresses, data_per_worker * input_dim);
        bool labels_success = labels_msg.scatter(worker_addresses, data_per_worker);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "  Training inputs scatter: " << (inputs_success ? "SUCCESS" : "SIMULATED") << std::endl;
        std::cout << "  Training labels scatter: " << (labels_success ? "SUCCESS" : "SIMULATED") << std::endl;
        std::cout << "  Scatter time:           " << duration.count() << " μs" << std::endl;
        std::cout << "  Data per worker:        " << data_per_worker << " samples" << std::endl;
    }
    
    void barrier_synchronization() {
        std::vector<std::string> worker_addresses;
        for (int i = 0; i < num_workers_; ++i) {
            worker_addresses.push_back("worker_" + std::to_string(i));
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = collectives_->barrier(worker_addresses);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "  Barrier sync: " << (success ? "SUCCESS" : "SIMULATED") 
                  << " (" << duration.count() << " μs)" << std::endl;
    }
    
    void gather_gradients() {
        // Simulate gathering gradients from all workers
        worker_gradients_.clear();
        worker_gradients_.reserve(num_workers_);
        
        for (int i = 0; i < num_workers_; ++i) {
            ucx::UCXFloatVector worker_grad(channel_);
            worker_grad.resize(model_params_.weights.size());
            
            // Simulate gradient computation for worker i
            std::random_device rd;
            std::mt19937 gen(rd() + i); // Different seed per worker
            std::normal_distribution<float> grad_dist(0.0f, 0.01f);
            
            for (size_t j = 0; j < worker_grad.size(); ++j) {
                worker_grad[j] = grad_dist(gen);
            }
            
            worker_gradients_.push_back(std::move(worker_grad));
        }
        
        std::vector<std::string> worker_addresses;
        for (int i = 0; i < num_workers_; ++i) {
            worker_addresses.push_back("worker_" + std::to_string(i));
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        // In real scenario: gather from actual workers
        // For simulation: use pre-generated gradients
        bool success = true; // Simulated success
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "  Gradient gather: " << (success ? "SUCCESS" : "FAILED") 
                  << " (" << duration.count() << " μs)" << std::endl;
        std::cout << "  Gradients from " << worker_gradients_.size() << " workers" << std::endl;
    }
    
    void allreduce_gradients() {
        if (worker_gradients_.empty()) return;
        
        // Create averaged gradient vector
        ucx::UCXFloatVector averaged_gradients(channel_);
        averaged_gradients.resize(model_params_.weights.size());
        
        // Initialize with zeros
        std::fill(averaged_gradients.begin(), averaged_gradients.end(), 0.0f);
        
        // Sum all gradients
        for (const auto& worker_grad : worker_gradients_) {
            for (size_t i = 0; i < averaged_gradients.size(); ++i) {
                averaged_gradients[i] += worker_grad[i];
            }
        }
        
        // Average
        for (auto& grad : averaged_gradients) {
            grad /= static_cast<float>(num_workers_);
        }
        
        std::vector<std::string> worker_addresses;
        for (int i = 0; i < num_workers_; ++i) {
            worker_addresses.push_back("worker_" + std::to_string(i));
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = collectives_->allreduce(averaged_gradients, worker_addresses);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "  Allreduce: " << (success ? "SUCCESS" : "SIMULATED") 
                  << " (" << duration.count() << " μs)" << std::endl;
        
        // Calculate gradient statistics
        float grad_sum = std::accumulate(averaged_gradients.begin(), averaged_gradients.end(), 0.0f);
        float grad_mean = grad_sum / averaged_gradients.size();
        
        std::cout << "  Gradient mean: " << std::scientific << std::setprecision(3) << grad_mean << std::endl;
    }
    
    void update_model_parameters() {
        // Apply averaged gradients to model parameters
        for (size_t i = 0; i < model_params_.weights.size() && !worker_gradients_.empty(); ++i) {
            // Simple SGD update
            float avg_grad = 0.0f;
            for (const auto& worker_grad : worker_gradients_) {
                avg_grad += worker_grad[i];
            }
            avg_grad /= static_cast<float>(num_workers_);
            
            model_params_.weights[i] -= model_params_.learning_rate * avg_grad;
        }
        
        std::cout << "  Model parameters updated with learning rate " << model_params_.learning_rate << std::endl;
    }
    
    void print_training_summary() {
        auto stats = channel_->get_stats();
        
        print_separator("Training Session Summary");
        
        std::cout << "Training completed successfully!" << std::endl;
        std::cout << "Final model state:" << std::endl;
        model_params_.print_summary();
        
        std::cout << "\nCommunication Statistics:" << std::endl;
        std::cout << "  Messages sent:       " << stats.messages_sent << std::endl;
        std::cout << "  Messages received:   " << stats.messages_received << std::endl;
        std::cout << "  Total data transfer: " << (stats.bytes_sent + stats.bytes_received) / (1024*1024) << " MB" << std::endl;
        std::cout << "  Zero-copy operations:" << (stats.zero_copy_sends + stats.zero_copy_receives) << std::endl;
        std::cout << "  Average latency:     " << std::fixed << std::setprecision(2) 
                  << stats.avg_latency_us << " μs" << std::endl;
        std::cout << "  Average bandwidth:   " << std::fixed << std::setprecision(2)
                  << stats.avg_bandwidth_mbps << " MB/s" << std::endl;
    }
};

void run_coordinator(int num_workers) {
    print_separator("UCX Distributed Training Coordinator");
    
    try {
        TrainingCoordinator coordinator(num_workers);
        coordinator.run_training_simulation();
        
    } catch (const std::exception& e) {
        std::cerr << "Coordinator error: " << e.what() << std::endl;
    }
}

void run_worker(int worker_id, const std::string& coordinator_address) {
    print_separator("UCX Training Worker " + std::to_string(worker_id));
    
    std::cout << "Worker Configuration:" << std::endl;
    std::cout << "  Worker ID: " << worker_id << std::endl;
    std::cout << "  Coordinator: " << coordinator_address << std::endl;
    
    try {
        // Create worker channel
        auto worker_channel = ucx::create_ucx_client(coordinator_address, ucx::TransportMode::AUTO);
        if (!worker_channel) {
            throw std::runtime_error("Failed to connect to coordinator");
        }
        
        std::cout << "Connected to coordinator successfully!" << std::endl;
        
        // Simulate worker training loop
        std::cout << "Worker ready for training..." << std::endl;
        std::cout << "Press Enter to simulate training completion..." << std::endl;
        std::cin.get();
        
        auto stats = worker_channel->get_stats();
        std::cout << "\nWorker " << worker_id << " Statistics:" << std::endl;
        std::cout << "  Messages exchanged: " << (stats.messages_sent + stats.messages_received) << std::endl;
        std::cout << "  Data transferred:   " << (stats.bytes_sent + stats.bytes_received) / 1024 << " KB" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Worker error: " << e.what() << std::endl;
    }
}

void run_simulation() {
    print_separator("UCX Collective Operations Simulation");
    
    std::cout << "This simulation demonstrates UCX collective operations" << std::endl;
    std::cout << "in a controlled environment without requiring multiple processes." << std::endl;
    
    try {
        // Simulate a 4-worker distributed training scenario
        const int num_workers = 4;
        std::cout << "\nSimulating distributed training with " << num_workers << " workers..." << std::endl;
        
        TrainingCoordinator coordinator(num_workers);
        coordinator.run_training_simulation();
        
    } catch (const std::exception& e) {
        std::cerr << "Simulation error: " << e.what() << std::endl;
    }
}

#endif // PSYNE_UCX_SUPPORT

int main(int argc, char* argv[]) {
    print_separator("Psyne UCX Collective Operations Demo");
    
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string mode = argv[1];
    
#if defined(PSYNE_UCX_SUPPORT)
    
    if (mode == "coordinator") {
        if (argc < 3) {
            std::cerr << "Error: Coordinator mode requires number of workers" << std::endl;
            print_usage();
            return 1;
        }
        int num_workers = std::stoi(argv[2]);
        run_coordinator(num_workers);
        
    } else if (mode == "worker") {
        if (argc < 4) {
            std::cerr << "Error: Worker mode requires worker ID and coordinator address" << std::endl;
            print_usage();
            return 1;
        }
        int worker_id = std::stoi(argv[2]);
        std::string coordinator_address = argv[3];
        run_worker(worker_id, coordinator_address);
        
    } else if (mode == "simulate") {
        run_simulation();
        
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
    
    std::cout << "\nTo enable UCX support:" << std::endl;
    std::cout << "  1. Install UCX development libraries (libucp-dev, libucs-dev)" << std::endl;
    std::cout << "  2. Reconfigure with cmake" << std::endl;
    std::cout << "  3. Rebuild the project" << std::endl;
    return 1;
    
#endif
    
    return 0;
}