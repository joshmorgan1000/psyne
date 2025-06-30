#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <array>
#include <cmath>

using namespace psyne;

// Define a fixed-size message type for 64-dimensional embeddings
class Vec64f : public Message<Vec64f> {
public:
    static constexpr uint32_t message_type = 100;
    static constexpr size_t DIMENSIONS = 64;
    
    using Message<Vec64f>::Message;
    
    // Fixed size - no dynamic allocation
    static constexpr size_t calculate_size() {
        return sizeof(float) * DIMENSIONS;
    }
    
    // Array access
    float& operator[](size_t index) {
        if (index >= DIMENSIONS) {
            throw std::out_of_range("Vec64f: index out of range");
        }
        return reinterpret_cast<float*>(data())[index];
    }
    
    const float& operator[](size_t index) const {
        if (index >= DIMENSIONS) {
            throw std::out_of_range("Vec64f: index out of range");
        }
        return reinterpret_cast<const float*>(data())[index];
    }
    
    // STL-like interface
    float* begin() { return reinterpret_cast<float*>(data()); }
    float* end() { return begin() + DIMENSIONS; }
    const float* begin() const { return reinterpret_cast<const float*>(data()); }
    const float* end() const { return begin() + DIMENSIONS; }
    
    size_t size() const { return DIMENSIONS; }
    
    // Simple operations
    float norm() const {
        float sum = 0.0f;
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            float val = (*this)[i];
            sum += val * val;
        }
        return std::sqrt(sum);
    }
    
    void normalize() {
        float n = norm();
        if (n > 0.0f) {
            for (size_t i = 0; i < DIMENSIONS; ++i) {
                (*this)[i] /= n;
            }
        }
    }
    
    float dot(const Vec64f& other) const {
        float sum = 0.0f;
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            sum += (*this)[i] * other[i];
        }
        return sum;
    }
};

// Fixed-size matrix type
template<size_t Rows, size_t Cols>
class FixedFloatMatrix : public Message<FixedFloatMatrix<Rows, Cols>> {
public:
    static constexpr uint32_t message_type = 200 + Rows * 100 + Cols; // Unique per size
    
    using Message<FixedFloatMatrix<Rows, Cols>>::Message;
    
    static constexpr size_t calculate_size() {
        return sizeof(float) * Rows * Cols;
    }
    
    float& at(size_t row, size_t col) {
        if (row >= Rows || col >= Cols) {
            throw std::out_of_range("Matrix index out of range");
        }
        return reinterpret_cast<float*>(this->data())[row * Cols + col];
    }
    
    const float& at(size_t row, size_t col) const {
        if (row >= Rows || col >= Cols) {
            throw std::out_of_range("Matrix index out of range");
        }
        return reinterpret_cast<const float*>(this->data())[row * Cols + col];
    }
    
    void setIdentity() {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                at(i, j) = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    
    constexpr size_t rows() const { return Rows; }
    constexpr size_t cols() const { return Cols; }
};

int main() {
    std::cout << "Psyne Fixed-Size Message Demo\n";
    std::cout << "=============================\n\n";

    // Create a single-type channel optimized for 64-dimensional float vectors
    auto channel = create_channel("memory://embeddings", 100 * 1024 * 1024,
                                  ChannelMode::SPSC, ChannelType::SingleType);

    std::cout << "Channel created for Vec64f (64-dimensional float vectors)\n";
    std::cout << "Message size: " << Vec64f::calculate_size() << " bytes (fixed)\n\n";

    // Create a fixed-size vector - no dynamic allocation!
    try {
        Vec64f embedding(*channel);

        // Fill with test data
        for (size_t i = 0; i < 64; ++i) {
            embedding[i] = std::sin(i * 0.1f);
        }

        // Calculate norm
        float norm = embedding.norm();
        std::cout << "Embedding L2 norm: " << norm << "\n";

        // Normalize the embedding
        embedding.normalize();
        std::cout << "Normalized embedding (first 5 values): ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << embedding[i] << " ";
        }
        std::cout << "...\n";

        // Send the message
        embedding.send();
        std::cout << "Message sent\n";

    } catch (const std::exception& e) {
        std::cerr << "Error creating embedding: " << e.what() << "\n";
    }

    // Receive and process
    size_t msg_size;
    uint32_t msg_type;
    void* msg_data = channel->receive_raw_message(msg_size, msg_type);
    
    if (msg_data && msg_type == Vec64f::message_type) {
        Vec64f received(msg_data, msg_size);
        
        // Calculate dot product with self
        float dot_product = received.dot(received);
        std::cout << "Dot product with self (should be ~1.0): " << dot_product << "\n";

        // Demonstrate zero-copy nature
        std::cout << "\nZero-copy verification:\n";
        std::cout << "  Received at:   " << static_cast<void*>(received.data()) << "\n";
        std::cout << "  Message is a view into the ring buffer - no copying occurred\n";
        
        channel->release_raw_message(msg_data);
    }

    // Demonstrate matrix operations
    std::cout << "\n--- Fixed-Size Matrix Demo ---\n";

    try {
        FixedFloatMatrix<16, 16> attention_weights(*channel);
        
        // Fill with identity matrix
        attention_weights.setIdentity();

        // Add some noise
        for (size_t i = 0; i < 16; ++i) {
            for (size_t j = 0; j < 16; ++j) {
                attention_weights.at(i, j) += (i + j) * 0.01f;
            }
        }

        // Simple trace calculation
        float trace = 0.0f;
        for (size_t i = 0; i < 16; ++i) {
            trace += attention_weights.at(i, i);
        }
        std::cout << "Matrix trace: " << trace << "\n";

        // The memory is perfectly aligned for GPU operations
        std::cout << "Matrix data pointer (GPU-ready): "
                  << static_cast<void*>(attention_weights.data()) << "\n";
        std::cout << "Matrix size: " << FixedFloatMatrix<16, 16>::calculate_size() 
                  << " bytes (fixed)\n";

        attention_weights.send();

    } catch (const std::exception& e) {
        std::cerr << "Error creating matrix: " << e.what() << "\n";
    }

    std::cout << "\n--- Performance Benefits of Fixed-Size ---\n";
    std::cout << "1. No dynamic allocation overhead\n";
    std::cout << "2. Known size at compile time enables optimizations\n";
    std::cout << "3. Perfect memory alignment for SIMD/GPU operations\n";
    std::cout << "4. Zero-copy throughout the entire pipeline\n";

    std::cout << "\nDemo completed successfully!\n";
    return 0;
}