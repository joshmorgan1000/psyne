/**
 * @file concept_based_demo.cpp
 * @brief Demonstration of pure concept-based substrates (no inheritance!)
 * 
 * This shows how substrates can be completely different implementations
 * that just satisfy the behavioral concepts.
 */

#include "../include/psyne/concepts/substrate_concepts.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdlib>
#include <cstring>

// Simple message for demo
struct DemoMessage {
    int id;
    float value;
    char name[32];
    
    DemoMessage() : id(0), value(0.0f) {
        std::memset(name, 0, sizeof(name));
    }
    
    DemoMessage(int id, float value, const char* name) 
        : id(id), value(value) {
        std::strncpy(this->name, name, sizeof(this->name) - 1);
    }
};

// Memory-only substrate (no inheritance!)
struct MemorySubstrate {
    void* allocate_memory_slab(size_t size) {
        std::cout << "MemorySubstrate: Allocating " << size << " bytes\n";
        return std::aligned_alloc(64, size);
    }
    
    void deallocate_memory_slab(void* ptr) {
        std::cout << "MemorySubstrate: Deallocating memory\n";
        std::free(ptr);
    }
    
    void transport_send(void* data, size_t size) {
        auto* msg = static_cast<DemoMessage*>(data);
        std::cout << "MemorySubstrate: In-memory transport - ID:" << msg->id 
                  << " Value:" << msg->value << " Name:'" << msg->name << "'\n";
    }
    
    void transport_receive(void* buffer, size_t size) {
        std::cout << "MemorySubstrate: Ready for in-memory receive\n";
    }
    
    const char* substrate_name() const { return "Memory"; }
    bool is_zero_copy() const { return true; }
    bool is_cross_process() const { return false; }
};

// CSV file substrate (completely different implementation!)
struct CSVSubstrate {
    CSVSubstrate() {
        csv_file_.open("messages.csv", std::ios::out | std::ios::app);
        if (csv_file_.is_open()) {
            csv_file_ << "ID,Value,Name\n"; // Header
        }
    }
    
    ~CSVSubstrate() {
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
    }
    
    void* allocate_memory_slab(size_t size) {
        std::cout << "CSVSubstrate: Allocating " << size << " bytes for CSV processing\n";
        return std::malloc(size);
    }
    
    void deallocate_memory_slab(void* ptr) {
        std::cout << "CSVSubstrate: Deallocating CSV memory\n";
        std::free(ptr);
    }
    
    void transport_send(void* data, size_t size) {
        auto* msg = static_cast<DemoMessage*>(data);
        if (csv_file_.is_open()) {
            csv_file_ << msg->id << "," << msg->value << ",\"" << msg->name << "\"\n";
            csv_file_.flush();
            std::cout << "CSVSubstrate: Wrote message to CSV file\n";
        }
    }
    
    void transport_receive(void* buffer, size_t size) {
        std::cout << "CSVSubstrate: Ready to read from CSV file\n";
        // Could implement CSV parsing here
    }
    
    const char* substrate_name() const { return "CSV"; }
    bool is_zero_copy() const { return false; }
    bool is_cross_process() const { return true; }
    
private:
    std::ofstream csv_file_;
};

// Mock GPU substrate (shows how GPU vendors could plug in)
struct MockGPUSubstrate {
    void* allocate_memory_slab(size_t size) {
        std::cout << "MockGPU: Allocating " << size << " bytes of unified memory\n";
        // In real implementation: cudaMallocManaged(&ptr, size);
        return std::aligned_alloc(64, size);
    }
    
    void deallocate_memory_slab(void* ptr) {
        std::cout << "MockGPU: Deallocating GPU memory\n";
        // In real implementation: cudaFree(ptr);
        std::free(ptr);
    }
    
    void transport_send(void* data, size_t size) {
        auto* msg = static_cast<DemoMessage*>(data);
        std::cout << "MockGPU: GPU kernel launch for message ID:" << msg->id << "\n";
        std::cout << "  -> Synchronizing device...\n";
        std::cout << "  -> Launching CUDA kernel...\n";
        std::cout << "  -> Processing complete!\n";
    }
    
    void transport_receive(void* buffer, size_t size) {
        std::cout << "MockGPU: GPU-based receive operation\n";
    }
    
    const char* substrate_name() const { return "MockGPU"; }
    bool is_zero_copy() const { return true; }
    bool is_cross_process() const { return false; }
};

// Simple SPSC pattern (no inheritance!)
struct SPSCPattern {
    void* coordinate_allocation(void* slab, size_t max_messages, size_t message_size) {
        size_t offset = allocation_count_ * message_size;
        allocation_count_++;
        std::cout << "SPSC: Coordinating allocation #" << allocation_count_ << "\n";
        return static_cast<char*>(slab) + offset;
    }
    
    void* coordinate_receive() {
        std::cout << "SPSC: Coordinating receive\n";
        return nullptr; // Simplified for demo
    }
    
    void producer_sync() { /* lock-free */ }
    void consumer_sync() { /* lock-free */ }
    
    const char* pattern_name() const { return "SPSC"; }
    bool needs_locks() const { return false; }
    size_t max_producers() const { return 1; }
    size_t max_consumers() const { return 1; }
    
private:
    size_t allocation_count_ = 0;
};

// Simple message lens (no inheritance!)
template<typename T>
struct MessageLens {
    explicit MessageLens(void* substrate_memory) 
        : memory_view_(static_cast<T*>(substrate_memory)) {
        new (memory_view_) T{};
    }
    
    template<typename... Args>
    MessageLens(void* substrate_memory, Args&&... args)
        : memory_view_(static_cast<T*>(substrate_memory)) {
        new (memory_view_) T(std::forward<Args>(args)...);
    }
    
    T* operator->() { return memory_view_; }
    const T* operator->() const { return memory_view_; }
    T& operator*() { return *memory_view_; }
    const T& operator*() const { return *memory_view_; }
    
    void* raw_memory() const { return memory_view_; }
    size_t size() const { return sizeof(T); }
    
private:
    T* memory_view_;
};

// Concept-based channel bridge (no inheritance!)
template<typename MessageType, typename SubstrateType, typename PatternType>
    requires psyne::concepts::ChannelConfiguration<MessageType, SubstrateType, PatternType>
struct ConceptBasedChannel {
    explicit ConceptBasedChannel(size_t slab_size = 1024) {
        slab_memory_ = substrate_.allocate_memory_slab(slab_size);
        max_messages_ = slab_size / sizeof(MessageType);
        
        std::cout << "ConceptChannel: Created with " << substrate_.substrate_name() 
                  << " substrate and " << pattern_.pattern_name() << " pattern\n";
    }
    
    ~ConceptBasedChannel() {
        if (slab_memory_) {
            substrate_.deallocate_memory_slab(slab_memory_);
        }
    }
    
    MessageLens<MessageType> create_message() {
        void* memory = pattern_.coordinate_allocation(slab_memory_, max_messages_, sizeof(MessageType));
        return MessageLens<MessageType>(memory);
    }
    
    template<typename... Args>
    MessageLens<MessageType> create_message(Args&&... args) {
        void* memory = pattern_.coordinate_allocation(slab_memory_, max_messages_, sizeof(MessageType));
        return MessageLens<MessageType>(memory, std::forward<Args>(args)...);
    }
    
    void send_message(const MessageLens<MessageType>& message) {
        substrate_.transport_send(message.raw_memory(), message.size());
    }
    
    const char* substrate_name() const { return substrate_.substrate_name(); }
    const char* pattern_name() const { return pattern_.pattern_name(); }
    bool is_zero_copy() const { return substrate_.is_zero_copy(); }
    
private:
    SubstrateType substrate_;
    PatternType pattern_;
    void* slab_memory_ = nullptr;
    size_t max_messages_ = 0;
};

void demonstrate_concept_based_substrates() {
    std::cout << "\n=== Concept-Based Substrates Demo ===\n";
    std::cout << "NO INHERITANCE! PURE DUCK TYPING!\n\n";
    
    // 1. Memory substrate
    {
        std::cout << "1. Testing Memory Substrate:\n";
        ConceptBasedChannel<DemoMessage, MemorySubstrate, SPSCPattern> memory_channel;
        
        auto msg = memory_channel.create_message(100, 3.14f, "MemoryMsg");
        std::cout << "   Created: " << msg->name << "\n";
        memory_channel.send_message(msg);
        std::cout << "\n";
    }
    
    // 2. CSV substrate  
    {
        std::cout << "2. Testing CSV Substrate:\n";
        ConceptBasedChannel<DemoMessage, CSVSubstrate, SPSCPattern> csv_channel;
        
        auto msg1 = csv_channel.create_message(200, 2.71f, "CSVMsg1");
        auto msg2 = csv_channel.create_message(201, 1.41f, "CSVMsg2");
        
        csv_channel.send_message(msg1);
        csv_channel.send_message(msg2);
        std::cout << "   Check messages.csv file!\n\n";
    }
    
    // 3. GPU substrate
    {
        std::cout << "3. Testing Mock GPU Substrate:\n";
        ConceptBasedChannel<DemoMessage, MockGPUSubstrate, SPSCPattern> gpu_channel;
        
        auto msg = gpu_channel.create_message(300, 9.99f, "GPUMsg");
        gpu_channel.send_message(msg);
        std::cout << "\n";
    }
    
    std::cout << "=== All Substrates Worked! ===\n";
    std::cout << "✅ Memory substrate - pure in-memory\n";
    std::cout << "✅ CSV substrate - file persistence\n";
    std::cout << "✅ GPU substrate - device processing\n";
    std::cout << "✅ Same interface, completely different implementations!\n";
    std::cout << "✅ NO BASE CLASSES! PURE CONCEPTS!\n";
    std::cout << "🚀 PLUGIN ECOSYSTEM READY!\n";
}

int main() {
    std::cout << "Concept-Based Substrate Demonstration\n";
    std::cout << "====================================\n";
    std::cout << "Substrates as pure behavioral concepts!\n";
    
    try {
        demonstrate_concept_based_substrates();
        
        std::cout << "\nPlugin Development Guide:\n";
        std::cout << "========================\n";
        std::cout << "To create your own substrate, just implement:\n";
        std::cout << "• allocate_memory_slab(size_t) -> void*\n";
        std::cout << "• deallocate_memory_slab(void*)\n";
        std::cout << "• transport_send(void*, size_t)\n";
        std::cout << "• transport_receive(void*, size_t)\n";
        std::cout << "• substrate_name() -> const char*\n";
        std::cout << "• is_zero_copy() -> bool\n";
        std::cout << "• is_cross_process() -> bool\n\n";
        
        std::cout << "That's it! No inheritance, no virtual functions!\n";
        std::cout << "Pure duck typing with C++20 concepts! 🦆\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}