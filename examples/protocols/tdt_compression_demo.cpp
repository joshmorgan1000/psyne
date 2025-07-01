/**
 * @file tdt_compression_demo.cpp
 * @brief Demonstration of TDT compression substrate over IP networks
 *
 * This example shows how the TDT (Tensor Data Transform) compression substrate
 * automatically compresses floating-point tensor data and transmits it over
 * TCP or UDP networks with adaptive compression strategies.
 */

#include "psyne/channel/substrate/tdt_ip.hpp"
#include "psyne/concepts/substrate_concepts.hpp"
#include "logger.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <cstring>

using namespace psyne;
using namespace psyne::substrate;

/**
 * @brief Neural network tensor message for testing
 */
struct TensorMessage {
    enum class TensorType {
        WEIGHTS,
        GRADIENTS, 
        ACTIVATIONS
    };
    
    uint32_t tensor_id;
    uint32_t layer_id;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    float data[];  // Variable-sized tensor data
    
    size_t tensor_size() const {
        return width * height * channels * sizeof(float);
    }
    
    size_t total_size() const {
        return sizeof(TensorMessage) + tensor_size();
    }
    
    void fill_with_realistic_data(TensorType type = TensorType::WEIGHTS) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        size_t num_floats = width * height * channels;
        
        switch (type) {
            case TensorType::WEIGHTS:
                // Weights: normally distributed around 0
                {
                    std::normal_distribution<float> dist(0.0f, 0.1f);
                    for (size_t i = 0; i < num_floats; ++i) {
                        data[i] = dist(gen);
                    }
                }
                break;
                
            case TensorType::GRADIENTS:
                // Gradients: sparse with many small values
                {
                    std::uniform_real_distribution<float> sparse_dist(0.0f, 1.0f);
                    std::normal_distribution<float> grad_dist(0.0f, 0.01f);
                    
                    for (size_t i = 0; i < num_floats; ++i) {
                        if (sparse_dist(gen) < 0.7f) {  // 70% sparse
                            data[i] = 0.0f;
                        } else {
                            data[i] = grad_dist(gen);
                        }
                    }
                }
                break;
                
            case TensorType::ACTIVATIONS:
                // Activations: post-ReLU (many zeros, positive values)
                {
                    std::uniform_real_distribution<float> relu_dist(0.0f, 1.0f);
                    std::exponential_distribution<float> activation_dist(2.0f);
                    
                    for (size_t i = 0; i < num_floats; ++i) {
                        if (relu_dist(gen) < 0.4f) {  // 40% zeros (post-ReLU)
                            data[i] = 0.0f;
                        } else {
                            data[i] = activation_dist(gen);
                        }
                    }
                }
                break;
        }
    }
};

/**
 * @brief Simple SPSC pattern for demo
 */
struct SPSCPattern {
    void* coordinate_allocation(void* slab, size_t max_messages, size_t message_size) {
        size_t offset = allocation_count_ * message_size;
        allocation_count_++;
        return static_cast<char*>(slab) + offset;
    }
    
    void* coordinate_receive() {
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

/**
 * @brief Message lens for tensor data
 */
template<typename T>
struct TensorLens {
    explicit TensorLens(void* substrate_memory, size_t size) 
        : memory_view_(static_cast<T*>(substrate_memory)), size_(size) {
        // Placement new for the fixed part
        new (memory_view_) T{};
    }
    
    T* operator->() { return memory_view_; }
    const T* operator->() const { return memory_view_; }
    T& operator*() { return *memory_view_; }
    const T& operator*() const { return *memory_view_; }
    
    void* raw_memory() const { return memory_view_; }
    size_t size() const { return size_; }
    
private:
    T* memory_view_;
    size_t size_;
};

/**
 * @brief TDT Channel using concept-based design
 */
template<typename MessageType, typename SubstrateType, typename PatternType>
    requires psyne::concepts::ChannelConfiguration<MessageType, SubstrateType, PatternType>
struct TDTChannel {
    explicit TDTChannel(size_t slab_size = 1024 * 1024) : slab_size_(slab_size) {
        slab_memory_ = substrate_.allocate_memory_slab(slab_size);
        
        log_info("TDTChannel: Created with ", substrate_.substrate_name(), 
                 " substrate and ", pattern_.pattern_name(), " pattern");
        log_info("  Slab size: ", slab_size / 1024, " KB");
        log_info("  Zero-copy: ", substrate_.is_zero_copy() ? "Yes" : "No");
        log_info("  Cross-process: ", substrate_.is_cross_process() ? "Yes" : "No");
    }
    
    ~TDTChannel() {
        if (slab_memory_) {
            substrate_.deallocate_memory_slab(slab_memory_);
        }
    }
    
    TensorLens<MessageType> create_tensor_message(size_t tensor_size) {
        size_t total_size = sizeof(MessageType) + tensor_size;
        void* memory = pattern_.coordinate_allocation(slab_memory_, 
                                                     slab_size_ / total_size, 
                                                     total_size);
        return TensorLens<MessageType>(memory, total_size);
    }
    
    void send_tensor(const TensorLens<MessageType>& tensor) {
        substrate_.transport_send(tensor.raw_memory(), tensor.size());
    }
    
    void receive_tensor(TensorLens<MessageType>& tensor) {
        substrate_.transport_receive(tensor.raw_memory(), tensor.size());
    }
    
    // Access substrate for monitoring
    const SubstrateType& substrate() const { return substrate_; }
    
private:
    SubstrateType substrate_;
    PatternType pattern_;
    void* slab_memory_ = nullptr;
    size_t slab_size_;
};

/**
 * @brief Benchmark different tensor types
 */
void benchmark_tensor_compression() {
    log_info("=== TDT Compression Benchmark ===");
    
    // Test different tensor configurations
    struct TestConfig {
        const char* name;
        uint32_t width, height, channels;
        TensorMessage::TensorType type;
    };
    
    std::vector<TestConfig> configs = {
        {"Small Weights (64x64x3)", 64, 64, 3, TensorMessage::TensorType::WEIGHTS},
        {"Large Weights (512x512x64)", 512, 512, 64, TensorMessage::TensorType::WEIGHTS},
        {"Sparse Gradients (256x256x32)", 256, 256, 32, TensorMessage::TensorType::GRADIENTS},
        {"ReLU Activations (128x128x128)", 128, 128, 128, TensorMessage::TensorType::ACTIVATIONS},
    };
    
    for (const auto& config : configs) {
        log_info("Testing: ", config.name);
        
        // Calculate tensor size
        size_t tensor_size = config.width * config.height * config.channels * sizeof(float);
        size_t total_size = sizeof(TensorMessage) + tensor_size;
        
        // Create test data
        auto test_data = std::make_unique<uint8_t[]>(total_size);
        auto* tensor_msg = reinterpret_cast<TensorMessage*>(test_data.get());
        
        tensor_msg->tensor_id = 1;
        tensor_msg->layer_id = 1;
        tensor_msg->width = config.width;
        tensor_msg->height = config.height;
        tensor_msg->channels = config.channels;
        tensor_msg->fill_with_realistic_data(config.type);
        
        // Benchmark compression
        TDTCompressor compressor;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto compressed = compressor.compress(test_data.get(), total_size);
        auto compress_time = std::chrono::high_resolution_clock::now() - start_time;
        
        auto compress_ms = std::chrono::duration<double, std::milli>(compress_time).count();
        double compress_speed_mbps = (total_size / 1024.0 / 1024.0) / (compress_ms / 1000.0);
        
        // Benchmark decompression
        start_time = std::chrono::high_resolution_clock::now();
        auto decompressed = compressor.decompress(compressed);
        auto decompress_time = std::chrono::high_resolution_clock::now() - start_time;
        
        auto decompress_ms = std::chrono::duration<double, std::milli>(decompress_time).count();
        double decompress_speed_mbps = (total_size / 1024.0 / 1024.0) / (decompress_ms / 1000.0);
        
        // Verify correctness
        bool correct = (decompressed.size() == total_size) && 
                      (std::memcmp(test_data.get(), decompressed.data(), total_size) == 0);
        
        log_info("  Original size: ", total_size / 1024, " KB");
        log_info("  Compressed size: ", compressed.compressed_size() / 1024, " KB");
        log_info("  Compression ratio: ", std::fixed, std::setprecision(2), compressed.compression_ratio, "x");
        log_info("  Compress speed: ", std::fixed, std::setprecision(1), compress_speed_mbps, " MB/s");
        log_info("  Decompress speed: ", std::fixed, std::setprecision(1), decompress_speed_mbps, " MB/s");
        log_info("  Correctness: ", correct ? "✅ PASS" : "❌ FAIL");
        log_info("");
    }
}

/**
 * @brief Demonstrate TCP server mode
 */
void run_tdt_server() {
    log_info("=== TDT Server Demo ===");
    
    TDTConfig config;
    config.is_server = true;
    config.remote_port = 9876;
    config.use_udp = false;  // Use TCP
    config.enable_adaptive = true;
    
    using TDTTCPSubstrate = TDTIPSubstrate<TensorMessage>;
    TDTChannel<TensorMessage, TDTTCPSubstrate, SPSCPattern> server_channel;
    
    log_info("Server started, waiting for connections...");
    
    // Wait for connection
    while (!server_channel.substrate().is_connected()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    log_info("Client connected! Receiving tensors...");
    
    // Receive and process tensors
    for (int i = 0; i < 10; ++i) {
        size_t tensor_size = 256 * 256 * 3 * sizeof(float);  // RGB image
        auto tensor = server_channel.create_tensor_message(tensor_size);
        
        server_channel.receive_tensor(tensor);
        
        log_info("Received tensor ", tensor->tensor_id, " (", tensor->width, "x",
                 tensor->height, "x", tensor->channels, ") from layer ", tensor->layer_id);
        
        // Simulate processing
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    log_info("Server demo completed");
}

/**
 * @brief Demonstrate TCP client mode
 */
void run_tdt_client() {
    log_info("=== TDT Client Demo ===");
    
    TDTConfig config;
    config.is_server = false;
    config.remote_host = "localhost";
    config.remote_port = 9876;
    config.use_udp = false;  // Use TCP
    config.enable_adaptive = true;
    
    using TDTTCPSubstrate = TDTIPSubstrate<TensorMessage>;
    TDTChannel<TensorMessage, TDTTCPSubstrate, SPSCPattern> client_channel;
    
    // Wait for connection
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    log_info("Connected to server! Sending tensors...");
    
    // Send different types of tensors
    for (int i = 0; i < 10; ++i) {
        size_t tensor_size = 256 * 256 * 3 * sizeof(float);  // RGB image
        auto tensor = client_channel.create_tensor_message(tensor_size);
        
        // Fill tensor with data
        tensor->tensor_id = i + 1;
        tensor->layer_id = (i % 3) + 1;
        tensor->width = 256;
        tensor->height = 256;
        tensor->channels = 3;
        
        // Different data types for variety
        TensorMessage::TensorType type = static_cast<TensorMessage::TensorType>(i % 3);
        tensor->fill_with_realistic_data(type);
        
        client_channel.send_tensor(tensor);
        
        log_info("Sent tensor ", tensor->tensor_id, " (type: ", 
                 static_cast<int>(type), ") to server");
        
        // Simulate neural network timing
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    log_info("Client demo completed");
    
    // Show compression statistics
    log_info("Final compression ratio: ", 
             client_channel.substrate().get_compression_ratio(), "x");
    log_info("Network bandwidth: ", 
             client_channel.substrate().get_bandwidth_mbps(), " Mbps");
}

/**
 * @brief Demonstrate UDP mode for low-latency scenarios
 */
void run_udp_demo() {
    log_info("=== TDT UDP Demo ===");
    
    TDTConfig config;
    config.use_udp = true;
    config.remote_port = 9877;
    config.enable_adaptive = false;  // Always compress for UDP demo
    
    using TDTUDPSubstrate = TDTIPSubstrate<TensorMessage>;
    TDTChannel<TensorMessage, TDTUDPSubstrate, SPSCPattern> udp_channel;
    
    // Create small tensors for UDP (avoid fragmentation)
    for (int i = 0; i < 5; ++i) {
        size_t tensor_size = 64 * 64 * 1 * sizeof(float);  // Small grayscale
        auto tensor = udp_channel.create_tensor_message(tensor_size);
        
        tensor->tensor_id = i + 100;
        tensor->layer_id = 1;
        tensor->width = 64;
        tensor->height = 64;
        tensor->channels = 1;
        tensor->fill_with_realistic_data(TensorMessage::TensorType::ACTIVATIONS);
        
        udp_channel.send_tensor(tensor);
        
        log_info("Sent UDP tensor ", tensor->tensor_id, " (", 
                 tensor_size / 1024, " KB)");
    }
    
    log_info("UDP demo completed");
}

int main(int argc, char* argv[]) {
    log_info("TDT Compression Substrate Demonstration");
    log_info("======================================");
    
    try {
        // 1. Benchmark compression performance
        benchmark_tensor_compression();
        
        // 2. Choose demo mode
        if (argc > 1) {
            std::string mode = argv[1];
            
            if (mode == "server") {
                run_tdt_server();
            } else if (mode == "client") {
                run_tdt_client();
            } else if (mode == "udp") {
                run_udp_demo();
            } else {
                log_error("Unknown mode. Use: server, client, or udp");
                return 1;
            }
        } else {
            log_info("Demo modes:");
            log_info("  ", argv[0], " server  - Run TDT TCP server");
            log_info("  ", argv[0], " client  - Run TDT TCP client");
            log_info("  ", argv[0], " udp     - Run TDT UDP demo");
            log_info("");
            log_info("Running standalone compression benchmark only.");
        }
        
        log_info("TDT substrate demonstration completed successfully!");
        
    } catch (const std::exception& e) {
        log_error("Error: ", e.what());
        return 1;
    }
    
    return 0;
}