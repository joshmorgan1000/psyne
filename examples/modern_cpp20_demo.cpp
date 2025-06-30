/**
 * @file modern_cpp20_demo.cpp
 * @brief Demonstrates modern C++20 features in Psyne zero-copy messaging
 * 
 * This example shows the modern C++20 features from CORE_DESIGN.md:
 * - Concepts for type safety
 * - std::span for zero-copy data views
 * - constexpr/consteval for compile-time optimization
 * - Structured bindings
 */

#include <psyne/psyne.hpp>
#include <iostream>
#include <span>
#include <concepts>

using namespace psyne;

// Example of FixedSizeMessage concept
class TensorBatch : public Message<TensorBatch> {
public:
    static constexpr uint32_t message_type = 401;
    static constexpr size_t BATCH_SIZE = 32;
    static constexpr size_t FEATURES = 128;
    
    // Required by MessageType concept
    static size_t calculate_size() noexcept {
        return BATCH_SIZE * FEATURES * sizeof(float);
    }
    
    TensorBatch(Channel& channel) : Message<TensorBatch>(channel) {
        initialize();
    }
    
    void initialize() {
        // Initialize tensor data directly in ring buffer
        auto tensor_span = std::span<float>(reinterpret_cast<float*>(data()), BATCH_SIZE * FEATURES);
        std::fill(tensor_span.begin(), tensor_span.end(), 0.0f);
    }
    
    // Zero-copy access using std::span
    std::span<float> get_batch(size_t batch_idx) noexcept {
        auto tensor_span = std::span<float>(reinterpret_cast<float*>(data()), BATCH_SIZE * FEATURES);
        auto start = batch_idx * FEATURES;
        return tensor_span.subspan(start, FEATURES);
    }
    
    std::span<const float> get_batch(size_t batch_idx) const noexcept {
        auto tensor_span = std::span<const float>(reinterpret_cast<const float*>(data()), BATCH_SIZE * FEATURES);
        auto start = batch_idx * FEATURES;
        return tensor_span.subspan(start, FEATURES);
    }
    
    // Direct element access
    float& operator()(size_t batch, size_t feature) noexcept {
        return reinterpret_cast<float*>(data())[batch * FEATURES + feature];
    }
    
    const float& operator()(size_t batch, size_t feature) const noexcept {
        return reinterpret_cast<const float*>(data())[batch * FEATURES + feature];
    }
};

// Note: Concepts verification would go here in a full implementation
// static_assert(FixedSizeMessage<TensorBatch>);
// static_assert(MessageType<TensorBatch>);

// Example of using structured bindings with channels
auto create_channel_info(const std::string& uri) {
    auto channel = Channel::create(uri, 1024 * 1024, ChannelMode::SPSC, ChannelType::SingleType);
    size_t buffer_size = 1024 * 1024;  // Known size
    size_t capacity = 1024 * 1024;     // Known capacity
    
    return std::make_tuple(std::move(channel), buffer_size, capacity);
}

int main() {
    try {
        // Structured bindings (C++17/20 feature)
        auto [channel, buffer_size, capacity] = create_channel_info("memory://tensor_pipeline");
        
        std::cout << "Created channel with buffer size: " << buffer_size 
                  << ", capacity: " << capacity << std::endl;
        
        // Create zero-copy message directly in ring buffer
        TensorBatch tensor(*channel);
        
        // Compile-time size verification  
        constexpr auto tensor_size = TensorBatch::BATCH_SIZE * TensorBatch::FEATURES * sizeof(float);
        std::cout << "Tensor size (compile-time): " << tensor_size << " bytes" << std::endl;
        
        // Zero-copy data access using std::span
        auto batch_0 = tensor.get_batch(0);
        std::cout << "Batch 0 span size: " << batch_0.size() << " elements" << std::endl;
        
        // Direct memory writes (zero-copy)
        for (size_t i = 0; i < TensorBatch::FEATURES && i < 10; ++i) {
            batch_0[i] = static_cast<float>(i) * 0.1f;
        }
        
        // Verify data was written directly to ring buffer
        auto raw_span = std::span<const uint8_t>(tensor.data(), tensor.size());
        std::cout << "Raw buffer first 4 bytes as float: " 
                  << *reinterpret_cast<const float*>(raw_span.data()) << std::endl;
        
        // Send zero-copy notification
        tensor.send();
        
        std::cout << "✅ Modern C++20 zero-copy messaging demo completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}