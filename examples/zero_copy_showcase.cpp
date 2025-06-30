/**
 * @file zero_copy_showcase.cpp
 * @brief Demonstrates Psyne's zero-copy messaging architecture
 * 
 * This example shows how messages are views into pre-allocated ring buffers,
 * with no memory copies during transport.
 */

#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace psyne;

// Custom tensor message for AI/ML workloads
class TensorMessage : public Message<TensorMessage> {
public:
    static constexpr size_t BATCH_SIZE = 32;
    static constexpr size_t FEATURES = 512;
    
    static consteval size_t calculate_size() noexcept {
        return BATCH_SIZE * FEATURES * sizeof(float);
    }
    
    TensorMessage(Channel& channel) : Message<TensorMessage>(channel) {}
    
    // Zero-copy access to tensor data
    std::span<float> as_tensor() noexcept {
        return typed_data_span<float>();
    }
    
    float& at(size_t batch, size_t feature) noexcept {
        return as_tensor()[batch * FEATURES + feature];
    }
};

// Verify our message satisfies zero-copy concepts
static_assert(FixedSizeMessage<TensorMessage>);

void demonstrate_zero_copy_principles() {
    std::cout << "🚀 Psyne Zero-Copy Architecture Demo\n";
    std::cout << "====================================\n\n";
    
    // Create SPSC channel with pre-allocated ring buffer
    auto channel = Channel::create("memory://tensors", 
                                  64 * 1024 * 1024,  // 64MB ring buffer
                                  ChannelMode::SPSC,
                                  ChannelType::SingleType);
    
    std::cout << "✅ Created SPSC channel with 64MB ring buffer\n";
    std::cout << "   - Zero atomics for single producer/consumer\n";
    std::cout << "   - Pre-allocated memory (no runtime allocations)\n\n";
    
    // Demonstrate zero-copy write
    std::cout << "📝 Writing tensor directly to ring buffer:\n";
    
    // Message constructor reserves space in ring buffer
    TensorMessage tensor(*channel);
    
    std::cout << "   - Reserved " << tensor.size() << " bytes at offset " 
              << tensor.offset() << "\n";
    std::cout << "   - Data pointer: " << static_cast<void*>(tensor.data()) << "\n";
    
    // Fill tensor with data (writing directly to ring buffer)
    auto tensor_data = tensor.as_tensor();
    for (size_t i = 0; i < tensor_data.size(); ++i) {
        tensor_data[i] = static_cast<float>(i) * 0.1f;
    }
    
    std::cout << "   - Wrote " << (tensor.size() / 1024 / 1024) << "MB directly to buffer\n";
    
    // Send is just a notification - no data movement!
    tensor.send();
    std::cout << "   - Sent notification (no memory copy!)\n\n";
    
    // Demonstrate zero-copy read
    std::cout << "📖 Reading tensor with zero-copy view:\n";
    
    // Get view into ring buffer
    auto read_span = channel->buffer_span();
    if (!read_span.empty()) {
        std::cout << "   - Got view at: " << static_cast<void*>(read_span.data()) << "\n";
        std::cout << "   - Size: " << read_span.size() << " bytes\n";
        std::cout << "   - First value: " << *reinterpret_cast<float*>(read_span.data()) << "\n";
        
        // Advance read pointer when done
        channel->advance_read_pointer(TensorMessage::calculate_size());
        std::cout << "   - Advanced read pointer (no deallocation needed)\n";
    }
    
    std::cout << "\n🎯 Key Zero-Copy Principles Demonstrated:\n";
    std::cout << "   1. Messages are views into ring buffer (offset + size)\n";
    std::cout << "   2. Data written directly to final destination\n";
    std::cout << "   3. Send() is just a notification\n";
    std::cout << "   4. No memory copies, no allocations\n";
    std::cout << "   5. SPSC uses zero atomics for maximum speed\n";
}

void demonstrate_network_zero_copy() {
    std::cout << "\n\n🌐 Network Transport (TCP) Zero-Copy:\n";
    std::cout << "=====================================\n\n";
    
    // TCP channels also use ring buffers
    auto tcp_channel = Channel::create("tcp://:5010", 
                                      32 * 1024 * 1024,  // 32MB
                                      ChannelMode::SPSC,
                                      ChannelType::SingleType);
    
    std::cout << "✅ TCP channel created on port 5010\n";
    std::cout << "   - Ring buffer for zero-copy queuing\n";
    std::cout << "   - Scatter-gather I/O (no intermediate buffers)\n";
    std::cout << "   - Direct streaming from ring buffer to socket\n\n";
    
    // For network channels, the flow is:
    // 1. Producer writes to ring buffer (zero-copy)
    // 2. TCP channel streams from ring buffer to socket (scatter-gather)
    // 3. Receiver writes from socket to ring buffer (one copy - network boundary)
    // 4. Consumer reads from ring buffer (zero-copy)
    
    std::cout << "📊 Network flow:\n";
    std::cout << "   Producer → Ring Buffer → Socket → Network → Socket → Ring Buffer → Consumer\n";
    std::cout << "             ^^^^^^^^^^^            ^^^^^^^            ^^^^^^^^^^^\n";
    std::cout << "             Zero-copy              One copy           Zero-copy\n";
    std::cout << "                                   (network boundary)\n";
}

int main() {
    try {
        demonstrate_zero_copy_principles();
        demonstrate_network_zero_copy();
        
        std::cout << "\n✨ Demo complete! Psyne provides true zero-copy messaging.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}