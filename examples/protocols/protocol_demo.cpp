/**
 * @file protocol_demo.cpp
 * @brief Demonstration of protocols as intelligent buffers between substrates
 *
 * This shows how protocols provide semantic data transformation while
 * substrates handle physical transport - clean separation of concerns!
 */

#include "psyne/protocol/tdt_compression.hpp"
#include "psyne/concepts/substrate_concepts.hpp"
#include "psyne/concepts/protocol_concepts.hpp"
#include "logger.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>

using namespace psyne;
using namespace psyne::protocol;

/**
 * @brief Simple tensor message for testing
 */
struct TensorMessage {
    uint32_t tensor_id;
    uint32_t width, height, channels;
    float data[];  // Variable-sized tensor data
    
    size_t tensor_size() const {
        return width * height * channels * sizeof(float);
    }
    
    size_t total_size() const {
        return sizeof(TensorMessage) + tensor_size();
    }
    
    void fill_with_random_data() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        size_t num_floats = width * height * channels;
        for (size_t i = 0; i < num_floats; ++i) {
            data[i] = dist(gen);
        }
    }
};

/**
 * @brief Mock substrate for demonstration (could be TCP, UDP, IPC, etc.)
 */
struct MockNetworkSubstrate {
    mutable std::vector<uint8_t> last_sent_data;
    mutable size_t total_bytes_sent = 0;
    
    void* allocate_memory_slab(size_t size) {
        log_info("MockSubstrate: Allocated ", size, " bytes");
        return std::aligned_alloc(64, size);
    }
    
    void deallocate_memory_slab(void* ptr) {
        log_info("MockSubstrate: Deallocated memory");
        std::free(ptr);
    }
    
    void transport_send(void* data, size_t size) {
        // Simulate network send
        last_sent_data.assign(static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + size);
        total_bytes_sent += size;
        log_info("MockSubstrate: Sent ", size, " bytes over network");
    }
    
    void transport_receive(void* buffer, size_t size) {
        // Simulate network receive
        if (last_sent_data.size() >= size) {
            std::memcpy(buffer, last_sent_data.data(), size);
            log_info("MockSubstrate: Received ", size, " bytes from network");
        }
    }
    
    const char* substrate_name() const { return "MockNetwork"; }
    bool is_zero_copy() const { return false; }
    bool is_cross_process() const { return true; }
    
    // Mock network metrics
    size_t get_total_bytes_sent() const { return total_bytes_sent; }
    const std::vector<uint8_t>& get_last_data() const { return last_sent_data; }
};

/**
 * @brief Channel with protocol composition
 */
template<typename MessageType, typename SubstrateType, typename ProtocolType>
    requires psyne::concepts::Substrate<SubstrateType> && psyne::concepts::Protocol<ProtocolType>
struct ProtocolChannel {
    explicit ProtocolChannel(size_t slab_size = 1024 * 1024) 
        : slab_size_(slab_size) {
        slab_memory_ = substrate_.allocate_memory_slab(slab_size);
        
        log_info("ProtocolChannel: Created with ", substrate_.substrate_name(), 
                 " substrate and ", protocol_.protocol_name(), " protocol");
    }
    
    ~ProtocolChannel() {
        if (slab_memory_) {
            substrate_.deallocate_memory_slab(slab_memory_);
        }
    }
    
    MessageType* create_message(size_t tensor_size) {
        size_t total_size = sizeof(MessageType) + tensor_size;
        if (allocation_offset_ + total_size > slab_size_) {
            allocation_offset_ = 0; // Simple wrap-around
        }
        
        void* memory = static_cast<char*>(slab_memory_) + allocation_offset_;
        allocation_offset_ += total_size;
        
        return static_cast<MessageType*>(memory);
    }
    
    void send_message(MessageType* message) {
        size_t message_size = message->total_size();
        
        // 1. Protocol analyzes and potentially transforms the data
        protocol_.analyze_data(message, message_size);
        
        if (protocol_.should_transform(message, message_size)) {
            log_info("Channel: Protocol deciding to transform data");
            
            // 2. Protocol encodes the data
            auto encoded_data = protocol_.encode(message, message_size);
            
            // 3. Substrate transports the encoded data
            substrate_.transport_send(encoded_data.data(), encoded_data.size());
            
            log_info("Channel: Sent ", message_size, " bytes as ", encoded_data.size(), 
                     " bytes (", protocol_.transformation_ratio(), "x ratio)");
        } else {
            log_info("Channel: Protocol passing through without transformation");
            
            // Direct transport without protocol transformation
            substrate_.transport_send(message, message_size);
            
            log_info("Channel: Sent ", message_size, " bytes uncompressed");
        }
    }
    
    MessageType* receive_message() {
        // This is simplified - in real implementation, we'd need to know the size
        // For demo purposes, we'll decode the last sent data
        const auto& raw_data = substrate_.get_last_data();
        if (raw_data.empty()) return nullptr;
        
        try {
            // 1. Protocol decodes the data
            auto decoded_data = protocol_.decode(raw_data);
            
            // 2. Return pointer to decoded message
            MessageType* result = create_message(decoded_data.size() - sizeof(MessageType));
            std::memcpy(result, decoded_data.data(), decoded_data.size());
            
            log_info("Channel: Received and decoded ", raw_data.size(), 
                     " bytes back to ", decoded_data.size(), " bytes");
            
            return result;
            
        } catch (const std::exception& e) {
            log_error("Channel: Failed to decode received data: ", e.what());
            return nullptr;
        }
    }
    
    // Update protocol with network conditions
    void update_network_conditions(double bandwidth_mbps, double latency_ms, double cpu_usage) {
        protocol_.update_network_metrics(bandwidth_mbps, latency_ms);
        protocol_.update_system_metrics(cpu_usage);
        
        log_info("Channel: Updated conditions - BW: ", bandwidth_mbps, 
                 " Mbps, Latency: ", latency_ms, "ms, CPU: ", cpu_usage * 100, "%");
    }
    
    // Access underlying components
    const SubstrateType& substrate() const { return substrate_; }
    const ProtocolType& protocol() const { return protocol_; }

private:
    SubstrateType substrate_;
    ProtocolType protocol_;
    void* slab_memory_ = nullptr;
    size_t slab_size_;
    size_t allocation_offset_ = 0;
};

/**
 * @brief Test different scenarios
 */
void demonstrate_protocol_intelligence() {
    log_info("=== Protocol Intelligence Demo ===");
    
    // Create channel with TDT compression protocol
    ProtocolChannel<TensorMessage, MockNetworkSubstrate, TDTCompressionProtocol> channel;
    
    // Test 1: Small tensor (should pass through)
    {
        log_info("\n1. Testing small tensor (should pass through):");
        
        auto* small_tensor = channel.create_message(32 * sizeof(float)); // 128 bytes
        small_tensor->tensor_id = 1;
        small_tensor->width = 8;
        small_tensor->height = 4;
        small_tensor->channels = 1;
        small_tensor->fill_with_random_data();
        
        size_t original_size = small_tensor->total_size();
        channel.send_message(small_tensor);
        
        size_t sent_size = channel.substrate().get_total_bytes_sent();
        log_info("Result: ", original_size, " -> ", sent_size, " bytes (passthrough expected)");
    }
    
    // Test 2: Large tensor on fast network (should pass through)
    {
        log_info("\n2. Testing large tensor on fast network (should pass through):");
        
        channel.update_network_conditions(1000.0, 1.0, 0.3); // Fast network, low CPU
        
        auto* large_tensor = channel.create_message(256 * 256 * 3 * sizeof(float)); // 768KB
        large_tensor->tensor_id = 2;
        large_tensor->width = 256;
        large_tensor->height = 256;
        large_tensor->channels = 3;
        large_tensor->fill_with_random_data();
        
        size_t baseline_sent = channel.substrate().get_total_bytes_sent();
        size_t original_size = large_tensor->total_size();
        
        channel.send_message(large_tensor);
        
        size_t sent_size = channel.substrate().get_total_bytes_sent() - baseline_sent;
        log_info("Result: ", original_size, " -> ", sent_size, " bytes (passthrough on fast network)");
    }
    
    // Test 3: Large tensor on slow network (should compress)
    {
        log_info("\n3. Testing large tensor on slow network (should compress):");
        
        channel.update_network_conditions(50.0, 10.0, 0.4); // Slow network, moderate CPU
        
        auto* compress_tensor = channel.create_message(128 * 128 * 32 * sizeof(float)); // 2MB
        compress_tensor->tensor_id = 3;
        compress_tensor->width = 128;
        compress_tensor->height = 128;
        compress_tensor->channels = 32;
        compress_tensor->fill_with_random_data();
        
        size_t baseline_sent = channel.substrate().get_total_bytes_sent();
        size_t original_size = compress_tensor->total_size();
        
        channel.send_message(compress_tensor);
        
        size_t sent_size = channel.substrate().get_total_bytes_sent() - baseline_sent;
        double compression_ratio = static_cast<double>(original_size) / sent_size;
        
        log_info("Result: ", original_size, " -> ", sent_size, " bytes (", 
                 compression_ratio, "x compression on slow network)");
    }
    
    // Test 4: High CPU usage (should pass through)
    {
        log_info("\n4. Testing with high CPU usage (should pass through):");
        
        channel.update_network_conditions(30.0, 20.0, 0.9); // Slow network but high CPU
        
        auto* cpu_tensor = channel.create_message(64 * 64 * 16 * sizeof(float)); // 256KB
        cpu_tensor->tensor_id = 4;
        cpu_tensor->width = 64;
        cpu_tensor->height = 64;
        cpu_tensor->channels = 16;
        cpu_tensor->fill_with_random_data();
        
        size_t baseline_sent = channel.substrate().get_total_bytes_sent();
        size_t original_size = cpu_tensor->total_size();
        
        channel.send_message(cpu_tensor);
        
        size_t sent_size = channel.substrate().get_total_bytes_sent() - baseline_sent;
        log_info("Result: ", original_size, " -> ", sent_size, " bytes (passthrough due to high CPU)");
    }
    
    // Test 5: Round-trip test
    {
        log_info("\n5. Testing round-trip compression/decompression:");
        
        channel.update_network_conditions(25.0, 15.0, 0.5); // Force compression
        
        auto* roundtrip_tensor = channel.create_message(100 * 100 * 8 * sizeof(float)); // 320KB
        roundtrip_tensor->tensor_id = 999;
        roundtrip_tensor->width = 100;
        roundtrip_tensor->height = 100;
        roundtrip_tensor->channels = 8;
        roundtrip_tensor->fill_with_random_data();
        
        // Store original data for comparison
        size_t original_size = roundtrip_tensor->total_size();
        std::vector<uint8_t> original_data(original_size);
        std::memcpy(original_data.data(), roundtrip_tensor, original_size);
        
        // Send message
        channel.send_message(roundtrip_tensor);
        
        // Receive message back
        auto* received_tensor = channel.receive_message();
        
        if (received_tensor) {
            bool data_matches = (std::memcmp(original_data.data(), received_tensor, original_size) == 0);
            log_info("Round-trip test: ", data_matches ? "✅ PASS" : "❌ FAIL");
            log_info("Original tensor ID: ", *reinterpret_cast<uint32_t*>(original_data.data()));
            log_info("Received tensor ID: ", received_tensor->tensor_id);
        } else {
            log_error("Round-trip test: ❌ FAIL - no data received");
        }
    }
    
    log_info("\n=== Protocol Demo Complete ===");
    log_info("Key insight: Protocols provide intelligent semantic transformation");
    log_info("while substrates handle physical transport - clean separation!");
}

/**
 * @brief Show composability with different substrates
 */
void demonstrate_substrate_composability() {
    log_info("\n=== Substrate Composability Demo ===");
    log_info("Same TDT protocol can work with ANY substrate!");
    
    // Mock different substrate types
    struct TCPSubstrate {
        void* allocate_memory_slab(size_t size) { return std::malloc(size); }
        void deallocate_memory_slab(void* ptr) { std::free(ptr); }
        void transport_send(void* data, size_t size) { 
            log_info("TCP: Sent ", size, " bytes via TCP socket"); 
        }
        void transport_receive(void* buffer, size_t size) { 
            log_info("TCP: Received ", size, " bytes via TCP socket"); 
        }
        const char* substrate_name() const { return "TCP"; }
        bool is_zero_copy() const { return false; }
        bool is_cross_process() const { return true; }
    };
    
    struct UDPSubstrate {
        void* allocate_memory_slab(size_t size) { return std::malloc(size); }
        void deallocate_memory_slab(void* ptr) { std::free(ptr); }
        void transport_send(void* data, size_t size) { 
            log_info("UDP: Sent ", size, " bytes via UDP datagram"); 
        }
        void transport_receive(void* buffer, size_t size) { 
            log_info("UDP: Received ", size, " bytes via UDP datagram"); 
        }
        const char* substrate_name() const { return "UDP"; }
        bool is_zero_copy() const { return false; }
        bool is_cross_process() const { return true; }
    };
    
    struct SharedMemSubstrate {
        void* allocate_memory_slab(size_t size) { return std::aligned_alloc(64, size); }
        void deallocate_memory_slab(void* ptr) { std::free(ptr); }
        void transport_send(void* data, size_t size) { 
            log_info("SHM: Sent ", size, " bytes via shared memory"); 
        }
        void transport_receive(void* buffer, size_t size) { 
            log_info("SHM: Received ", size, " bytes via shared memory"); 
        }
        const char* substrate_name() const { return "SharedMemory"; }
        bool is_zero_copy() const { return true; }
        bool is_cross_process() const { return true; }
    };
    
    // Same protocol, different substrates!
    ProtocolChannel<TensorMessage, TCPSubstrate, TDTCompressionProtocol> tcp_channel;
    ProtocolChannel<TensorMessage, UDPSubstrate, TDTCompressionProtocol> udp_channel;  
    ProtocolChannel<TensorMessage, SharedMemSubstrate, TDTCompressionProtocol> shm_channel;
    
    // Create test tensor
    auto create_test_tensor = [](auto& channel) {
        auto* tensor = channel.create_message(50 * 50 * 4 * sizeof(float));
        tensor->tensor_id = 42;
        tensor->width = 50;
        tensor->height = 50; 
        tensor->channels = 4;
        tensor->fill_with_random_data();
        return tensor;
    };
    
    // Force compression for demo
    tcp_channel.update_network_conditions(30.0, 10.0, 0.4);
    udp_channel.update_network_conditions(30.0, 10.0, 0.4);
    shm_channel.update_network_conditions(30.0, 10.0, 0.4);
    
    log_info("Sending same tensor data via different substrates:");
    
    auto* tcp_tensor = create_test_tensor(tcp_channel);
    tcp_channel.send_message(tcp_tensor);
    
    auto* udp_tensor = create_test_tensor(udp_channel);
    udp_channel.send_message(udp_tensor);
    
    auto* shm_tensor = create_test_tensor(shm_channel);
    shm_channel.send_message(shm_tensor);
    
    log_info("✅ Same TDT protocol worked with TCP, UDP, and SharedMemory!");
    log_info("🚀 Protocols are truly substrate-agnostic!");
}

int main() {
    log_info("Protocol as Intelligent Buffer Demonstration");
    log_info("==========================================");
    
    try {
        demonstrate_protocol_intelligence();
        demonstrate_substrate_composability();
        
        log_info("\n🎉 Protocol demonstration completed successfully!");
        log_info("Key Takeaways:");
        log_info("• Protocols provide semantic data transformation");
        log_info("• Substrates provide physical transport");
        log_info("• Clean separation of concerns enables composability");
        log_info("• TDT compression works with ANY substrate!");
        
    } catch (const std::exception& e) {
        log_error("Error: ", e.what());
        return 1;
    }
    
    return 0;
}