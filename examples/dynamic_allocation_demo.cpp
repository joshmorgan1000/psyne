#include <psyne/psyne.hpp>
#include <psyne/memory/dynamic_slab_allocator.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <random>

using namespace psyne;
using namespace std::chrono_literals;

// Variable size message for testing dynamic allocation
class VariableMessage : public Message<VariableMessage> {
public:
    static constexpr uint32_t message_type = 600;
    
    template<typename Channel>
    explicit VariableMessage(Channel& channel, size_t payload_size) 
        : Message<VariableMessage>(channel)
        , payload_size_(payload_size) {}
    
    explicit VariableMessage(const void* data, size_t size) 
        : Message<VariableMessage>(data, size)
        , payload_size_(size - sizeof(size_t)) {}
    
    static size_t calculate_size_for(size_t payload) {
        return sizeof(size_t) + payload;
    }
    
    // Override to return variable size
    static constexpr size_t calculate_size() { 
        // This won't be called for variable messages
        return 0;
    }
    
    void set_pattern(uint8_t pattern) {
        if (!data_) return;
        
        // Store size first
        *reinterpret_cast<size_t*>(data_) = payload_size_;
        
        // Fill with pattern
        uint8_t* payload = data_ + sizeof(size_t);
        std::memset(payload, pattern, payload_size_);
    }
    
    bool verify_pattern(uint8_t expected) const {
        if (!data_) return false;
        
        size_t stored_size = *reinterpret_cast<const size_t*>(data_);
        if (stored_size != payload_size_) return false;
        
        const uint8_t* payload = data_ + sizeof(size_t);
        for (size_t i = 0; i < payload_size_; ++i) {
            if (payload[i] != expected) return false;
        }
        return true;
    }
    
    size_t get_payload_size() const { return payload_size_; }
    
    void before_send() {}
    
private:
    size_t payload_size_;
};

void demonstrate_dynamic_slab() {
    std::cout << "\n=== Dynamic Slab Allocator Demo ===" << std::endl;
    
    memory::DynamicSlabConfig config;
    config.initial_slab_size = 64 * 1024 * 1024;  // Start with 64MB
    config.max_slab_size = 1024 * 1024 * 1024;    // Max 1GB
    config.high_water_mark = 0.75;                // Grow at 75% usage
    config.low_water_mark = 0.25;                 // Shrink below 25% usage
    config.growth_factor = 2.0;                   // Double size when growing
    
    memory::DynamicSlabAllocator allocator(config);
    
    // Allocate increasingly larger chunks
    std::vector<std::pair<void*, size_t>> allocations;
    size_t sizes[] = {64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024};
    
    for (int round = 0; round < 3; ++round) {
        std::cout << "\nRound " << (round + 1) << ":" << std::endl;
        
        for (size_t size : sizes) {
            void* ptr = allocator.allocate(size);
            if (ptr) {
                allocations.push_back({ptr, size});
                auto stats = allocator.get_stats();
                std::cout << "  Allocated " << (size / (1024 * 1024)) << " MB"
                         << " (slabs: " << stats.num_slabs
                         << ", usage: " << std::fixed << std::setprecision(1) 
                         << (stats.usage_ratio * 100) << "%"
                         << ", capacity: " << (stats.total_capacity / (1024 * 1024)) << " MB)" << std::endl;
            }
        }
    }
    
    auto peak_stats = allocator.get_stats();
    std::cout << "\nPeak statistics:" << std::endl;
    std::cout << "  Total slabs: " << peak_stats.num_slabs << std::endl;
    std::cout << "  Total capacity: " << (peak_stats.total_capacity / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Total used: " << (peak_stats.total_used / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Smallest slab: " << (peak_stats.smallest_slab_size / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Largest slab: " << (peak_stats.largest_slab_size / (1024 * 1024)) << " MB" << std::endl;
    
    // Deallocate half to demonstrate shrinking
    std::cout << "\nDeallocating half of allocations..." << std::endl;
    size_t deallocated = 0;
    for (size_t i = 0; i < allocations.size(); i += 2) {
        allocator.deallocate(allocations[i].first, allocations[i].second);
        deallocated++;
    }
    
    allocator.perform_maintenance();
    
    auto after_dealloc_stats = allocator.get_stats();
    std::cout << "\nAfter deallocation:" << std::endl;
    std::cout << "  Total slabs: " << after_dealloc_stats.num_slabs << std::endl;
    std::cout << "  Total capacity: " << (after_dealloc_stats.total_capacity / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Total used: " << (after_dealloc_stats.total_used / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Usage ratio: " << std::fixed << std::setprecision(1) 
              << (after_dealloc_stats.usage_ratio * 100) << "%" << std::endl;
    std::cout << "  Number of growths: " << final_stats.num_growths << std::endl;
    std::cout << "  Recommended buffer size: " << allocator.get_recommended_buffer_size() << " bytes" << std::endl;
}

void demonstrate_dynamic_ring_buffer() {
    std::cout << "\n=== Dynamic Ring Buffer Demo ===" << std::endl;
    
    DynamicRingBuffer<SingleProducer, SingleConsumer>::Config config;
    config.initial_size = 1024;      // Start with 1KB
    config.max_size = 64 * 1024;     // Max 64KB
    config.resize_up_threshold = 0.7; // Resize at 70% full
    config.resize_check_interval = 1s; // Check every second
    
    DynamicSPSCRingBuffer buffer(config);
    
    // Producer thread - variable rate
    std::atomic<bool> running{true};
    std::thread producer([&]() {
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> size_dist(64, 512);
        std::uniform_int_distribution<int> burst_dist(1, 10);
        
        while (running) {
            // Burst mode - send multiple messages
            int burst_size = burst_dist(rng);
            
            for (int i = 0; i < burst_size; ++i) {
                size_t msg_size = size_dist(rng);
                auto handle = buffer.reserve(msg_size);
                
                if (handle) {
                    // Simulate message content
                    std::memset(handle->data, i % 256, msg_size);
                    handle->commit();
                } else {
                    std::cout << "  Failed to reserve " << msg_size << " bytes" << std::endl;
                }
            }
            
            // Variable sleep to create different load patterns
            std::this_thread::sleep_for(std::chrono::milliseconds(burst_dist(rng) * 10));
        }
    });
    
    // Consumer thread - steady rate
    std::thread consumer([&]() {
        size_t total_consumed = 0;
        
        while (running || !buffer.empty()) {
            auto handle = buffer.read();
            if (handle) {
                total_consumed += handle->size;
                // Simulate processing
                std::this_thread::sleep_for(5ms);
            } else {
                std::this_thread::sleep_for(10ms);
            }
        }
        
        std::cout << "  Total consumed: " << total_consumed << " bytes" << std::endl;
    });
    
    // Monitor thread
    for (int i = 0; i < 10; ++i) {
        std::this_thread::sleep_for(1s);
        
        auto stats = buffer.get_stats();
        std::cout << "  [" << i << "s] Buffer size: " << stats.current_size
                 << ", Peak usage: " << stats.peak_usage
                 << ", Avg usage: " << (stats.average_usage * 100) << "%"
                 << ", Resizes: " << stats.resize_count << std::endl;
    }
    
    running = false;
    producer.join();
    consumer.join();
    
    auto final_stats = buffer.get_stats();
    std::cout << "\nFinal buffer statistics:" << std::endl;
    std::cout << "  Total writes: " << final_stats.total_writes << std::endl;
    std::cout << "  Total reads: " << final_stats.total_reads << std::endl;
    std::cout << "  Failed reserves: " << final_stats.failed_reserves << std::endl;
    std::cout << "  Resize count: " << final_stats.resize_count << std::endl;
    std::cout << "  Final size: " << final_stats.current_size << " bytes" << std::endl;
}

void demonstrate_adaptive_channel() {
    std::cout << "\n=== Adaptive Channel Demo ===" << std::endl;
    std::cout << "Note: This demonstrates how dynamic allocation could be integrated with channels" << std::endl;
    
    // Create a channel with dynamic buffer
    const std::string uri = "memory://adaptive_demo";
    
    // In a real implementation, we would modify Channel to support dynamic buffers
    // For now, just show the concept
    DynamicSPSCRingBuffer dynamic_buffer;
    
    std::cout << "  Initial buffer size: " << dynamic_buffer.capacity() << " bytes" << std::endl;
    
    // Simulate varying load
    for (int phase = 0; phase < 3; ++phase) {
        std::cout << "\nPhase " << (phase + 1) << ":" << std::endl;
        
        // Different message sizes per phase
        size_t msg_sizes[] = {128, 512, 2048};
        size_t msg_size = msg_sizes[phase];
        int msg_count = 100 / (phase + 1);  // Fewer messages in later phases
        
        // Send messages
        for (int i = 0; i < msg_count; ++i) {
            auto handle = dynamic_buffer.reserve(msg_size);
            if (handle) {
                std::memset(handle->data, phase, msg_size);
                handle->commit();
            }
        }
        
        // Consume half
        for (int i = 0; i < msg_count / 2; ++i) {
            auto handle = dynamic_buffer.read();
            if (!handle) break;
        }
        
        auto stats = dynamic_buffer.get_stats();
        std::cout << "  Buffer resized to: " << stats.current_size << " bytes" << std::endl;
        std::cout << "  Current usage: " << (stats.average_usage * 100) << "%" << std::endl;
    }
}

int main() {
    std::cout << "Dynamic Allocation Demo" << std::endl;
    std::cout << "======================" << std::endl;
    
    try {
        demonstrate_dynamic_slab();
        demonstrate_dynamic_ring_buffer();
        demonstrate_adaptive_channel();
        
        std::cout << "\nDemo completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}