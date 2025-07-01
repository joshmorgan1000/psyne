# Protocols: The Intelligent Buffer Layer

Protocols in Psyne represent the intelligent transformation layer that sits between patterns and substrates. They provide semantic understanding of data and make smart decisions about how to optimize it for transport.

## What Are Protocols?

Protocols are the "intelligent buffer between substrates" that can:

1. **Analyze data semantically** - Understand what the data represents
2. **Make transformation decisions** - Decide whether to compress, encrypt, etc.
3. **Adapt to conditions** - Respond to network bandwidth, CPU usage, etc.
4. **Maintain data integrity** - Ensure lossless transformations

```cpp
Message → Pattern → Protocol → Substrate → (network) → Substrate → Protocol → Pattern → Message
                       ↑                                              ↑
                 Intelligent                                   Intelligent
               Transformation                                Reconstruction
```

## Protocol Concept

Any type that satisfies the Protocol concept can be used:

```cpp
template<typename P>
concept Protocol = requires(P protocol, void* data, size_t size) {
    // Data understanding
    { protocol.should_transform(data, size) } -> std::same_as<bool>;
    { protocol.analyze_data(data, size) } -> std::same_as<void>;
    
    // Transformation
    { protocol.encode(data, size) } -> std::convertible_to<std::vector<uint8_t>>;
    { protocol.decode(std::declval<const std::vector<uint8_t>&>()) } -> std::convertible_to<std::vector<uint8_t>>;
    
    // Adaptation
    { protocol.update_network_metrics(0.0, 0.0) } -> std::same_as<void>;
    { protocol.update_system_metrics(0.0) } -> std::same_as<void>;
    
    // Identity
    { protocol.protocol_name() } -> std::convertible_to<const char*>;
    { protocol.is_lossless() } -> std::same_as<bool>;
    { protocol.transformation_ratio() } -> std::same_as<double>;
};
```

## Built-in Protocols

### TDT Compression Protocol

Based on research from [arXiv:2506.18062](https://arxiv.org/html/2506.18062v1), optimized for IEEE 754 floating-point tensor data:

```cpp
#include "psyne/protocol/tdt_compression.hpp"

TDTCompressionProtocol protocol;

// Automatically adapts based on conditions
protocol.update_network_metrics(50.0, 10.0);  // 50 Mbps, 10ms latency
protocol.update_system_metrics(0.4);          // 40% CPU usage

// Smart transformation decisions
bool will_compress = protocol.should_transform(tensor_data, tensor_size);
if (will_compress) {
    auto compressed = protocol.encode(tensor_data, tensor_size);
    // ... transport compressed data
    auto original = protocol.decode(compressed);
}
```

**Performance Characteristics**:
- **Sparse gradients**: 1.25x compression ratio
- **ReLU activations**: 0.73x compression ratio  
- **Dense weights**: 0.53x compression ratio (better at larger scales)
- **Compression speed**: 11-16 MB/s
- **Decompression speed**: 29-35 MB/s

### Encryption Protocol (Future)

```cpp
struct AESEncryptionProtocol {
    bool should_transform(void* data, size_t size) { 
        return true;  // Always encrypt
    }
    
    std::vector<uint8_t> encode(void* data, size_t size) {
        return aes_encrypt(data, size, encryption_key_);
    }
    
    std::vector<uint8_t> decode(const std::vector<uint8_t>& encrypted) {
        return aes_decrypt(encrypted, encryption_key_);
    }
    
    const char* protocol_name() const { return "AES-256"; }
    bool is_lossless() const { return true; }
    double transformation_ratio() const { return 1.0; }  // No compression
};
```

## Using Protocols

### Basic Usage

```cpp
// Create a channel with protocol
ProtocolChannel<TensorMessage, TCPSubstrate, TDTCompressionProtocol> channel;

// Protocol automatically analyzes and transforms data
auto tensor = channel.create_message(tensor_size);
// ... fill tensor data
channel.send_message(tensor);  // Automatically compressed if beneficial
```

### Adaptive Behavior

```cpp
// Update network conditions
channel.update_network_conditions(bandwidth_mbps, latency_ms, cpu_usage);

// Protocol adapts automatically:
// - Fast network + low CPU = no compression (avoid overhead)
// - Slow network + low CPU = compression (save bandwidth)
// - Any network + high CPU = no compression (save CPU)
```

### Protocol Stacks (Future)

Compose multiple protocols for layered transformations:

```cpp
// Compression → Encryption → Checksum
ProtocolStack<TDTCompressionProtocol, AESEncryptionProtocol, ChecksumProtocol> stack;

// Data flows through each protocol in sequence
Channel<TensorMessage, TCPSubstrate, SPSCPattern, decltype(stack)> secure_channel;
```

## Creating Custom Protocols

### Simple Example: Checksum Protocol

```cpp
struct ChecksumProtocol {
    bool should_transform(void* data, size_t size) {
        return size > 1024;  // Only checksum larger messages
    }
    
    std::vector<uint8_t> encode(void* data, size_t size) {
        std::vector<uint8_t> result(size + 4);  // +4 for checksum
        
        // Copy original data
        std::memcpy(result.data(), data, size);
        
        // Calculate and append checksum
        uint32_t checksum = calculate_crc32(data, size);
        std::memcpy(result.data() + size, &checksum, 4);
        
        return result;
    }
    
    std::vector<uint8_t> decode(const std::vector<uint8_t>& encoded) {
        if (encoded.size() < 4) throw std::runtime_error("Invalid checksum data");
        
        size_t data_size = encoded.size() - 4;
        
        // Extract checksum
        uint32_t stored_checksum;
        std::memcpy(&stored_checksum, encoded.data() + data_size, 4);
        
        // Verify checksum
        uint32_t calculated_checksum = calculate_crc32(encoded.data(), data_size);
        if (stored_checksum != calculated_checksum) {
            throw std::runtime_error("Checksum mismatch");
        }
        
        // Return original data
        return std::vector<uint8_t>(encoded.begin(), encoded.begin() + data_size);
    }
    
    void analyze_data(void* data, size_t size) { /* No analysis needed */ }
    void update_network_metrics(double bandwidth, double latency) { /* Not used */ }
    void update_system_metrics(double cpu_usage) { /* Not used */ }
    
    const char* protocol_name() const { return "CRC32-Checksum"; }
    bool is_lossless() const { return true; }
    double transformation_ratio() const { return 1.0; }  // No compression
    double processing_overhead_ms() const { return 0.1; }  // Very fast
};
```

### Advanced Example: Custom Compression

```cpp
struct CustomCompressionProtocol {
private:
    std::atomic<double> bandwidth_threshold_{100.0};  // Mbps
    std::atomic<double> cpu_threshold_{0.8};          // 80%
    
public:
    bool should_transform(void* data, size_t size) {
        // Custom logic for when to compress
        if (size < 4096) return false;  // Skip small messages
        
        // Check system conditions
        double bandwidth = get_current_bandwidth();
        double cpu_usage = get_current_cpu_usage();
        
        return (bandwidth < bandwidth_threshold_) && (cpu_usage < cpu_threshold_);
    }
    
    std::vector<uint8_t> encode(void* data, size_t size) {
        // Your custom compression algorithm
        return my_custom_compress(static_cast<uint8_t*>(data), size);
    }
    
    std::vector<uint8_t> decode(const std::vector<uint8_t>& encoded) {
        return my_custom_decompress(encoded);
    }
    
    void analyze_data(void* data, size_t size) {
        // Analyze data characteristics to tune compression
        double entropy = calculate_entropy(static_cast<uint8_t*>(data), size);
        // Adjust compression parameters based on entropy
    }
    
    void update_network_metrics(double bandwidth, double latency) {
        bandwidth_threshold_.store(bandwidth * 0.8);  // Compress if below 80% of capacity
    }
    
    void update_system_metrics(double cpu_usage) {
        // Don't compress if CPU is busy
    }
    
    const char* protocol_name() const { return "CustomCompression"; }
    bool is_lossless() const { return true; }
    double transformation_ratio() const { return last_compression_ratio_; }
};
```

## Protocol Performance Considerations

### When to Use Protocols

✅ **Good Use Cases**:
- Network transport (compression saves bandwidth)
- Security requirements (encryption)
- Data integrity (checksums)
- Cross-platform serialization

❌ **Avoid When**:
- Ultra-low latency requirements (< 100ns)
- Small message sizes (< 1KB)
- High CPU usage scenarios
- Local in-process communication

### Performance Tips

1. **Profile transformation overhead**:
   ```cpp
   auto overhead = protocol.processing_overhead_ms();
   if (overhead > latency_budget) {
       // Consider disabling protocol
   }
   ```

2. **Monitor adaptation metrics**:
   ```cpp
   double ratio = protocol.transformation_ratio();
   if (ratio < 1.1) {  // Less than 10% improvement
       // Protocol may not be beneficial
   }
   ```

3. **Use appropriate thresholds**:
   ```cpp
   // Tune based on your specific requirements
   protocol.update_network_metrics(measured_bandwidth, measured_latency);
   protocol.update_system_metrics(measured_cpu_usage);
   ```

## Substrate Compatibility

Protocols work with **any** substrate:

```cpp
// Same protocol, different substrates
ProtocolChannel<TensorMessage, TCPSubstrate, TDTCompressionProtocol> tcp_channel;
ProtocolChannel<TensorMessage, UDPSubstrate, TDTCompressionProtocol> udp_channel;
ProtocolChannel<TensorMessage, IPCSubstrate, TDTCompressionProtocol> ipc_channel;
```

This separation allows protocols to focus on data transformation while substrates handle physical transport.

## Future Protocols

Planned protocol implementations:

- **Advanced Compression**: Integration with zstd, lz4, brotli
- **Encryption**: AES, ChaCha20, post-quantum algorithms  
- **Serialization**: Protocol Buffers, MessagePack, custom formats
- **Error Correction**: Reed-Solomon, LDPC codes
- **Delta Compression**: For streaming updates
- **Deduplication**: Content-based deduplication

The protocol system is designed to be easily extensible for these and other custom transformations.