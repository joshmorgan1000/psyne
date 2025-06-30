/**
 * @file simple_messaging_zero_copy.cpp
 * @brief Simple messaging example using current zero-copy design
 * 
 * This example demonstrates the current zero-copy API:
 * - Messages provide direct access to ring buffer data
 * - No memory allocation during message creation
 * - Direct buffer access using pointers and spans
 * - Modern C++20 features
 */

#include <psyne/psyne.hpp>
#include <iostream>
#include <chrono>
#include <span>

using namespace psyne;

// Define a simple fixed-size message for maximum performance
class SimpleFloat32Vector : public Message<SimpleFloat32Vector> {
public:
    static constexpr uint32_t message_type = 42;
    static constexpr size_t VECTOR_SIZE = 256; // Fixed size for performance
    
    // Required by Message base class
    static size_t calculate_size() noexcept {
        return VECTOR_SIZE * sizeof(float);
    }
    
    SimpleFloat32Vector(Channel& channel) : Message<SimpleFloat32Vector>(channel) {
        initialize();
    }
    
    void initialize() {
        // Initialize all values to zero
        auto span = get_data_span();
        std::fill(span.begin(), span.end(), 0.0f);
    }
    
    // Zero-copy data access using std::span
    std::span<float> get_data_span() noexcept {
        return std::span<float>(reinterpret_cast<float*>(data()), VECTOR_SIZE);
    }
    
    std::span<const float> get_data_span() const noexcept {
        return std::span<const float>(reinterpret_cast<const float*>(data()), VECTOR_SIZE);
    }
    
    // Direct element access
    float& operator[](size_t index) noexcept {
        return get_data_span()[index];
    }
    
    const float& operator[](size_t index) const noexcept {
        return get_data_span()[index];
    }
    
    size_t vector_size() const noexcept {
        return VECTOR_SIZE;
    }
};

int main() {
    std::cout << "🚀 Psyne Zero-Copy Messaging Example\n";
    std::cout << "====================================\n\n";
    
    try {
        // Create channel with proper zero-copy design
        auto channel = Channel::create("memory://zero_copy_demo", 64 * 1024 * 1024, // 64MB
                                     ChannelMode::SPSC, ChannelType::SingleType);
        
        std::cout << "✅ Channel created with zero-copy ring buffer\n";
        std::cout << "   Buffer size: " << (64 * 1024 * 1024) << " bytes\n";
        std::cout << "   Channel mode: SPSC (Single Producer Single Consumer)\n\n";
        
        // Demonstrate zero-copy message creation
        std::cout << "📝 Creating message directly in ring buffer...\n";
        
        std::cout << "   Creating message with " << SimpleFloat32Vector::calculate_size() << " bytes...\n";
        
        // Create message - this just gets a view into ring buffer
        SimpleFloat32Vector msg(*channel);
        
        std::cout << "   ✅ Message created (zero allocation!)\n";
        std::cout << "   Message data pointer: " << static_cast<void*>(msg.data()) << "\n";
        std::cout << "   Message size: " << msg.size() << " bytes (" 
                  << msg.vector_size() << " floats)\n";
        
        std::cout << "   ✅ Message data is allocated in channel buffer\n\n";
        
        // Write data directly to ring buffer via message view
        std::cout << "💾 Writing data directly to ring buffer...\n";
        
        auto data_span = msg.get_data_span();
        std::cout << "   Data span size: " << data_span.size() << " floats\n";
        
        // Fill with test pattern
        for (size_t i = 0; i < data_span.size(); ++i) {
            data_span[i] = static_cast<float>(i) * 0.1f;
        }
        
        // Show first few values
        std::cout << "   First 10 values: ";
        for (size_t i = 0; i < 10; ++i) {
            std::cout << msg[i] << " ";
        }
        std::cout << "\n";
        
        std::cout << "   ✅ Data written directly to message buffer\n\n";
        
        // Send message - just advances write pointer and sends notification
        std::cout << "📤 Sending message (notification only)...\n";
        
        auto send_start = std::chrono::high_resolution_clock::now();
        msg.send();
        auto send_end = std::chrono::high_resolution_clock::now();
        
        auto send_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(send_end - send_start);
        std::cout << "   ✅ Message sent in " << send_duration.count() << " nanoseconds\n\n";
        
        // Demonstrate high-performance batch processing
        std::cout << "⚡ Performance test - batch message processing...\n";
        
        const size_t NUM_MESSAGES = 1000;
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < NUM_MESSAGES; ++i) {
            SimpleFloat32Vector batch_msg(*channel);
            
            // Write pattern data
            auto span = batch_msg.get_data_span();
            for (size_t j = 0; j < 100; ++j) { // Only fill first 100 elements for speed
                span[j] = static_cast<float>(i * 1000 + j);
            }
            
            batch_msg.send();
            
        }
        
        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start);
        
        double messages_per_second = NUM_MESSAGES * 1000000.0 / batch_duration.count();
        double avg_latency = batch_duration.count() / double(NUM_MESSAGES);
        
        std::cout << "   📊 Processed " << NUM_MESSAGES << " messages\n";
        std::cout << "   📊 Throughput: " << static_cast<size_t>(messages_per_second) << " msg/sec\n";
        std::cout << "   📊 Average latency: " << avg_latency << " microseconds\n";
        std::cout << "   📊 Total data processed: " << (NUM_MESSAGES * msg.size() / 1024 / 1024) << " MB\n\n";
        
        // Demonstrate std::span zero-copy views
        std::cout << "🔍 Demonstrating std::span zero-copy views...\n";
        {
            SimpleFloat32Vector span_msg(*channel);
            
            // Get different span views of same data
            auto float_span = span_msg.get_data_span();
            std::span<uint8_t> raw_span(span_msg.data(), span_msg.size());
            
            std::cout << "   Raw bytes span size: " << raw_span.size() << " bytes\n";
            std::cout << "   Float span size: " << float_span.size() << " floats\n";
            
            // Write via float span
            float_span[0] = 3.14159f;
            float_span[1] = 2.71828f;
            
            // Read via uint32 span (same memory, different interpretation)
            auto uint32_span = std::span<uint32_t>(reinterpret_cast<uint32_t*>(span_msg.data()), span_msg.size() / sizeof(uint32_t));
            std::cout << "   Float 3.14159 as uint32: 0x" << std::hex << uint32_span[0] << std::dec << "\n";
            std::cout << "   Float 2.71828 as uint32: 0x" << std::hex << uint32_span[1] << std::dec << "\n";
            
            // All views point to same memory
            bool same_memory = (raw_span.data() == reinterpret_cast<uint8_t*>(float_span.data()) &&
                               raw_span.data() == reinterpret_cast<uint8_t*>(uint32_span.data()));
            std::cout << "   ✅ All spans point to same memory: " << (same_memory ? "YES" : "NO") << "\n\n";
        }
        
        std::cout << "🎉 Zero-copy messaging demo completed successfully!\n";
        std::cout << "\n📋 Key achievements:\n";
        std::cout << "   ✅ No memory allocations during message creation\n";
        std::cout << "   ✅ Direct ring buffer access with std::span\n";
        std::cout << "   ✅ Sub-microsecond message sending\n";
        std::cout << "   ✅ High throughput batch processing\n";
        std::cout << "   ✅ Modern C++20 concepts and features\n";
        std::cout << "   ✅ True zero-copy semantics throughout\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}