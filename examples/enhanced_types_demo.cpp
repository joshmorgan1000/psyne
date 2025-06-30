// Enhanced Message Types Demo - showcases additional message types in psyne
// Demonstrates complex numbers, ML tensors, and sparse matrices

#include <algorithm>
#include <complex>
#include <iostream>
#include <psyne/psyne.hpp>
#include <random>
#include <vector>

using namespace psyne;
using namespace psyne::types;

void demo_complex_vectors() {
    std::cout << "\n=== Complex Vector Demo ===\n";

    auto channel = create_channel("memory://complex", 10 * 1024 * 1024);

    // Create complex vector for signal processing
    ComplexVectorF signal(*channel);
    signal.initialize();
    signal.resize(64); // 64 complex samples

    // Generate a complex sinusoid
    for (size_t i = 0; i < signal.size(); ++i) {
        float phase = 2.0f * M_PI * i / signal.size();
        signal[i] = std::complex<float>(cos(phase), sin(phase));
    }

    std::cout << "Signal power: " << signal.power() << std::endl;

    // Apply conjugate (for correlation operations)
    signal.conjugate();
    std::cout << "Conjugated signal power: " << signal.power() << std::endl;

    // Send the signal
    signal.send();

    // Receive and verify
    size_t msg_size;
    uint32_t msg_type;
    void* msg_data = channel->receive_raw_message(msg_size, msg_type);
    
    if (msg_data && msg_type == ComplexVectorF::message_type) {
        ComplexVectorF received(msg_data, msg_size);
        std::cout << "Received signal with " << received.size() << " samples\n";
        std::cout << "First sample: " << received[0] << std::endl;
        channel->release_raw_message(msg_data);
    }
}

void demo_ml_tensors() {
    std::cout << "\n=== ML Tensor Demo ===\n";

    auto channel = create_channel("memory://tensors", 100 * 1024 * 1024);

    // Create a tensor for a batch of images
    MLTensorF batch(*channel);
    batch.initialize();

    // Set shape: [batch_size, channels, height, width]
    std::vector<uint32_t> shape = {32, 3, 224, 224}; // ImageNet size
    batch.set_shape(shape, MLTensorF::Layout::NCHW);

    std::cout << "Tensor shape: ";
    for (auto dim : batch.shape()) {
        std::cout << dim << " ";
    }
    std::cout << "\n";
    std::cout << "Total elements: " << batch.total_elements() << "\n";
    std::cout << "Layout: " << (batch.layout() == MLTensorF::Layout::NCHW ? "NCHW" : "Other") << "\n";

    // Fill with dummy data (normally would be actual image data)
    float* data = batch.data();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t i = 0; i < std::min(100u, batch.total_elements()); ++i) {
        data[i] = dist(gen);
    }

    // Calculate simple statistics
    float sum = 0.0f;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (uint32_t i = 0; i < std::min(1000u, batch.total_elements()); ++i) {
        sum += data[i];
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    std::cout << "Sample statistics (first 1000 elements):\n";
    std::cout << "  Mean: " << sum / 1000.0f << "\n";
    std::cout << "  Min: " << min_val << "\n";
    std::cout << "  Max: " << max_val << "\n";

    batch.send();
}

void demo_sparse_matrices() {
    std::cout << "\n=== Sparse Matrix Demo ===\n";

    auto channel = create_channel("memory://sparse", 10 * 1024 * 1024);

    // Create a sparse matrix for graph adjacency
    SparseMatrixF adjacency(*channel);
    adjacency.initialize();

    // Set up a small graph (10 nodes, ~20 edges)
    uint32_t num_nodes = 10;
    uint32_t num_edges = 20;
    adjacency.set_structure(num_nodes, num_nodes, num_edges);

    // Fill with random connections
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> node_dist(0, num_nodes - 1);
    std::uniform_real_distribution<float> weight_dist(0.1f, 1.0f);

    // Build CSR format
    uint32_t* row_ptr = adjacency.row_pointers();
    uint32_t* col_idx = adjacency.column_indices();
    float* values = adjacency.values();

    // Simple random graph generation
    uint32_t edge_count = 0;
    for (uint32_t row = 0; row < num_nodes; ++row) {
        row_ptr[row] = edge_count;
        
        // Add 1-3 random connections per node
        uint32_t connections = 1 + (gen() % 3);
        for (uint32_t c = 0; c < connections && edge_count < num_edges; ++c) {
            col_idx[edge_count] = node_dist(gen);
            values[edge_count] = weight_dist(gen);
            edge_count++;
        }
    }
    row_ptr[num_nodes] = edge_count;

    std::cout << "Sparse matrix: " << adjacency.rows() << "x" << adjacency.cols() << "\n";
    std::cout << "Non-zero elements: " << edge_count << "\n";
    std::cout << "Sparsity: " << (1.0f - float(edge_count) / (num_nodes * num_nodes)) * 100 << "%\n";

    // Demonstrate matrix-vector multiplication
    std::vector<float> x(num_nodes, 1.0f); // Input vector
    std::vector<float> y(num_nodes, 0.0f); // Output vector

    adjacency.multiply_vector(x.data(), y.data());

    std::cout << "Matrix-vector product (first 5 elements): ";
    for (size_t i = 0; i < 5 && i < num_nodes; ++i) {
        std::cout << y[i] << " ";
    }
    std::cout << "\n";

    adjacency.send();
}

void demo_zero_copy_benefits() {
    std::cout << "\n=== Zero-Copy Benefits Demo ===\n";

    auto channel = create_channel("memory://zerocopy", 100 * 1024 * 1024);

    // Create a large ML tensor
    MLTensorF tensor(*channel);
    tensor.initialize();
    
    // Large tensor shape
    std::vector<uint32_t> shape = {1, 512, 512, 3}; // High-res image
    tensor.set_shape(shape, MLTensorF::Layout::NHWC);

    float* data_ptr = tensor.data();
    std::cout << "Tensor data pointer: " << data_ptr << "\n";
    
    // Fill with data
    for (uint32_t i = 0; i < tensor.total_elements(); ++i) {
        data_ptr[i] = float(i % 256) / 255.0f;
    }

    // Send (zero-copy)
    tensor.send();

    // Receive
    size_t msg_size;
    uint32_t msg_type;
    void* msg_data = channel->receive_raw_message(msg_size, msg_type);
    
    if (msg_data && msg_type == MLTensorF::message_type) {
        MLTensorF received(msg_data, msg_size);
        float* received_ptr = received.data();
        
        std::cout << "Received data pointer: " << received_ptr << "\n";
        std::cout << "Same memory? " << (data_ptr == received_ptr ? "YES (zero-copy!)" : "NO") << "\n";
        std::cout << "Data size: " << received.total_elements() * sizeof(float) / (1024.0 * 1024.0) << " MB\n";
        
        channel->release_raw_message(msg_data);
    }
}

int main() {
    std::cout << "Psyne Enhanced Types Demo\n";
    std::cout << "=========================\n";
    std::cout << "Demonstrating advanced message types for AI/ML applications\n";

    demo_complex_vectors();
    demo_ml_tensors();
    demo_sparse_matrices();
    demo_zero_copy_benefits();

    std::cout << "\nDemo completed successfully!\n";
    return 0;
}