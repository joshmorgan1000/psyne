#include <psyne/psyne.hpp>
#include <iostream>
#include <chrono>

using namespace psyne;

int main() {
    std::cout << "Psyne Fixed-Size Message Demo\n";
    std::cout << "=============================\n\n";
    
    // Create a single-type channel optimized for 64-dimensional float vectors
    auto channel = create_channel("memory://embeddings", 100 * 1024 * 1024, 
                                 ChannelMode::SPSC, ChannelType::SingleType);
    
    std::cout << "Channel created for Vec64f (64-dimensional float vectors)\n";
    
    // Create a fixed-size vector - no dynamic allocation!
    Vec64f embedding(*channel);
    
    if (!embedding.is_valid()) {
        std::cerr << "Failed to allocate message\n";
        return 1;
    }
    
    // Fill with test data
    for (size_t i = 0; i < 64; ++i) {
        embedding[i] = std::sin(i * 0.1f);
    }
    
    // Use Eigen for linear algebra operations (zero-copy!)
    auto eigen_view = embedding.as_eigen();
    float norm = eigen_view.norm();
    std::cout << "Embedding L2 norm: " << norm << "\n";
    
    // Normalize the embedding
    eigen_view.normalize();
    std::cout << "Normalized embedding (first 5 values): ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << embedding[i] << " ";
    }
    std::cout << "...\n";
    
    // Send the message
    channel->send(embedding);
    
    // Receive and process
    auto received = channel->receive_single<Vec64f>();
    if (received) {
        // Direct Eigen operations on received data
        auto recv_eigen = received->as_eigen();
        float dot_product = recv_eigen.dot(recv_eigen);
        std::cout << "Dot product with self (should be ~1.0): " << dot_product << "\n";
        
        // Demonstrate zero-copy nature
        std::cout << "\nZero-copy verification:\n";
        std::cout << "  Sent from:     " << static_cast<void*>(embedding.data()) << "\n";
        std::cout << "  Received at:   " << static_cast<const void*>(received->data()) << "\n";
        std::cout << "  Both are views into the ring buffer - no copying occurred\n";
    }
    
    // Demonstrate matrix operations
    std::cout << "\n--- Fixed-Size Matrix Demo ---\n";
    
    FixedFloatMatrix<16, 16> attention_weights(*channel);
    if (attention_weights.is_valid()) {
        // Fill with identity matrix using Eigen
        auto matrix_eigen = attention_weights.as_eigen();
        matrix_eigen.setIdentity();
        
        // Add some noise
        for (size_t i = 0; i < 16; ++i) {
            for (size_t j = 0; j < 16; ++j) {
                attention_weights.at(i, j) += (i + j) * 0.01f;
            }
        }
        
        // Compute determinant using Eigen
        float det = matrix_eigen.determinant();
        std::cout << "Matrix determinant: " << det << "\n";
        
        // The memory is perfectly aligned for GPU operations
        std::cout << "Matrix data pointer (GPU-ready): " << static_cast<void*>(attention_weights.data()) << "\n";
        std::cout << "Matrix size: " << sizeof(float) * 16 * 16 << " bytes\n";
    }
    
    std::cout << "\nDemo completed successfully!\n";
    return 0;
}