# TDT (Tensor Data Transform) Compression Implementation Guide

## Overview

TDT is a lossless compression algorithm specifically designed for floating-point tensor data. It achieves better compression ratios than general-purpose algorithms by exploiting the internal byte structure of IEEE 754 floating-point numbers.

## Core Concept

Instead of treating a tensor as a stream of bytes, TDT recognizes that different byte positions within floating-point numbers have different statistical properties:

```
Float32: [byte0: sign+exponent] [byte1: exponent+mantissa] [byte2: mantissa] [byte3: mantissa]
         ↑ Often similar         ↑ Some correlation        ↑ More random    ↑ Most random
```

By separating and grouping bytes with similar properties, each group can be compressed more effectively.

## Algorithm Steps

### 1. Sample Analysis (30% of data)
```python
# Sample 30% of the tensor for feature extraction
sample_indices = random_sample(tensor_size, fraction=0.3)
samples = tensor[sample_indices]
```

### 2. Feature Extraction (per byte position)
For each byte position (0-3 for float32), extract:
- **Entropy**: How random is this byte position?
- **Unique values**: How many different byte values appear?
- **Autocorrelation**: Do consecutive values correlate?
- **Variance**: How spread out are the values?

### 3. Hierarchical Clustering
Group byte positions with similar features:
```
Example clusters:
- Cluster 0: [byte0, byte1] (high correlation, low entropy)
- Cluster 1: [byte2] (medium entropy)
- Cluster 2: [byte3] (high entropy, random)
```

### 4. Byte Stream Separation
```python
# Create separate streams for each cluster
streams = [[] for _ in clusters]

for word in tensor:
    for cluster_id, byte_positions in enumerate(clusters):
        for byte_pos in byte_positions:
            streams[cluster_id].append(word[byte_pos])
```

### 5. Per-Stream Compression
Apply appropriate compression to each stream:
- Low entropy streams → High compression ratio
- High entropy streams → Lower compression ratio

## Implementation Specifications

### C++ Structure
```cpp
class TDTCompressor {
public:
    struct Config {
        float sample_fraction = 0.3f;
        int word_size = 4;  // 4 bytes for float32, 8 for float64
        bool auto_detect_clusters = true;
        int max_clusters = 4;
    };
    
    struct CompressedData {
        std::vector<std::vector<uint8_t>> streams;
        std::vector<int> cluster_mapping;  // Which byte goes to which stream
        size_t original_size;
        int word_size;
    };
    
    CompressedData compress(const uint8_t* data, size_t size, Config config = {});
    std::vector<uint8_t> decompress(const CompressedData& compressed);
};
```

### Feature Extraction
```cpp
struct ByteFeatures {
    double entropy;           // Shannon entropy
    double autocorrelation;   // Correlation with previous value
    int unique_count;         // Number of unique byte values
    double mean;
    double variance;
    std::array<int, 256> histogram;  // Byte value distribution
};

ByteFeatures extract_features(const uint8_t* data, 
                            const std::vector<size_t>& sample_indices,
                            int byte_offset, 
                            int word_size);
```

### Clustering Algorithm
Use hierarchical clustering with distance metric:
```cpp
double feature_distance(const ByteFeatures& a, const ByteFeatures& b) {
    // Weighted combination of:
    // - Entropy difference
    // - Histogram similarity (e.g., Bhattacharyya distance)
    // - Autocorrelation difference
}
```

Choose optimal number of clusters using:
- Davies-Bouldin index
- Gap statistic
- Or simply use elbow method

## Optimizations

### 1. SIMD Byte Extraction
```cpp
// AVX2 example for fast byte shuffling
__m256i extract_bytes(const float* data, int byte_index) {
    __m256i mask = _mm256_set1_epi32(0xFF << (byte_index * 8));
    __m256i values = _mm256_loadu_si256((__m256i*)data);
    __m256i shifted = _mm256_srli_epi32(values, byte_index * 8);
    return _mm256_and_si256(shifted, _mm256_set1_epi32(0xFF));
}
```

### 2. Streaming Compression
For large tensors, process in chunks:
```cpp
class StreamingTDT {
    void compress_chunk(const float* chunk, size_t count) {
        // Maintain running statistics
        // Adapt clusters based on seen data
    }
};
```

### 3. Tensor-Specific Adaptations
```cpp
enum TensorType { WEIGHTS, GRADIENTS, ACTIVATIONS };

CompressedData compress_tensor(const Tensor& tensor) {
    Config config;
    
    switch (tensor.type()) {
        case WEIGHTS:
            // Weights often normally distributed
            config.sample_fraction = 0.1f;  // Need less sampling
            break;
        case GRADIENTS:
            // Gradients often sparse
            // Maybe add zero-run encoding before TDT
            break;
        case ACTIVATIONS:
            // Post-ReLU has many exact zeros
            // Pre-process to separate zero/non-zero
            break;
    }
    
    return compress(tensor.data(), tensor.size(), config);
}
```

## Streaming TDT

TDT can be adapted for streaming scenarios where tensors arrive continuously:

### Streaming Implementation
```cpp
class StreamingTDTCompressor {
private:
    // Sliding window for statistics
    struct SlidingWindow {
        static constexpr size_t WINDOW_SIZE = 1000;  // samples
        CircularBuffer<ByteFeatures> features[4];    // per byte position
        
        void update(const float* data, size_t count) {
            // Update features incrementally
        }
    } window;
    
    // Adaptive clustering that can evolve
    struct AdaptiveClusters {
        std::vector<int> mapping;
        int stability_counter = 0;
        static constexpr int RECLUSTER_THRESHOLD = 100;
        
        bool update(const ByteFeatures* features) {
            if (features_changed_significantly(features)) {
                mapping = recluster(features);
                stability_counter = 0;
                return true;  // Clustering changed
            }
            stability_counter++;
            return false;
        }
    } clusters;
    
    // Per-stream compression context
    std::vector<StreamContext> streams;

public:
    void process_chunk(const float* data, size_t count,
                      std::function<void(uint8_t*, size_t)> output) {
        // 1. Update sliding statistics
        window.update(data, count);
        
        // 2. Check if we need to recluster
        if (clusters.update(window.get_current_features())) {
            // Send new metadata when clustering changes
            send_cluster_metadata(clusters.mapping, output);
        }
        
        // 3. Process data with current clustering
        auto compressed = compress_with_clustering(data, count, clusters.mapping);
        output(compressed.data(), compressed.size());
    }
};
```

## Adaptive Compression

Adaptive compression monitors performance in real-time and adjusts strategy based on network conditions, CPU load, and data characteristics:

### Architecture
```cpp
class AdaptiveCompressionManager {
public:
    enum Strategy {
        NO_COMPRESSION,
        LZ4_FAST,        // ~10-50 MB/s compression speed
        ZSTD_BALANCED,   // ~5-20 MB/s compression speed  
        TDT_FULL        // ~2-10 MB/s but best ratio for tensors
    };
    
private:
    struct PerformanceMetrics {
        // Compression metrics
        double compression_ratio_ma = 1.0;     // Moving average
        double compression_speed_mbps = 0.0;
        double decompression_speed_mbps = 0.0;
        
        // Network metrics
        double bandwidth_mbps = 1000.0;
        double latency_ms = 0.1;
        double packet_loss = 0.0;
        
        // System metrics
        double cpu_usage = 0.0;
        size_t memory_available_mb = 0;
        
        void update(const CompressionResult& result) {
            // Exponential moving average updates
            const double alpha = 0.1;
            compression_ratio_ma = (1-alpha) * compression_ratio_ma + 
                                  alpha * result.ratio;
            compression_speed_mbps = (1-alpha) * compression_speed_mbps + 
                                    alpha * result.speed_mbps;
        }
    } metrics;
    
    Strategy current_strategy = ZSTD_BALANCED;
    
public:
    Strategy select_strategy(const Tensor& tensor) {
        // 1. Estimate compression benefit
        double compress_time = tensor.size() / (metrics.compression_speed_mbps * 1e6);
        double network_time_raw = tensor.size() / (metrics.bandwidth_mbps * 125000);
        double network_time_compressed = network_time_raw / metrics.compression_ratio_ma;
        
        // 2. Factor in CPU availability
        if (metrics.cpu_usage > 0.9) {
            return NO_COMPRESSION;  // CPU saturated
        }
        
        // 3. Small tensor optimization
        if (tensor.size() < 64 * 1024) {  // < 64KB
            return NO_COMPRESSION;  // Overhead not worth it
        }
        
        // 4. Network-aware decision
        double total_time_compressed = compress_time + network_time_compressed;
        double speedup = network_time_raw / total_time_compressed;
        
        if (speedup < 1.1) {
            return NO_COMPRESSION;
        } else if (speedup < 1.5 || metrics.bandwidth_mbps > 5000) {
            return LZ4_FAST;  // Fast network, use fast compression
        } else if (tensor.is_sparse() || tensor.dtype() == float32) {
            return TDT_FULL;  // Best compression for tensors
        } else {
            return ZSTD_BALANCED;  // Good general purpose
        }
    }
    
    CompressedData compress_adaptive(const Tensor& tensor) {
        auto strategy = select_strategy(tensor);
        auto start = std::chrono::steady_clock::now();
        
        CompressedData result;
        switch (strategy) {
            case NO_COMPRESSION:
                result = {tensor.data(), tensor.size(), 1.0};
                break;
            case LZ4_FAST:
                result = compress_lz4(tensor);
                break;
            case ZSTD_BALANCED:
                result = compress_zstd(tensor, /*level=*/3);
                break;
            case TDT_FULL:
                result = compress_tdt(tensor);
                break;
        }
        
        // Update metrics
        auto end = std::chrono::steady_clock::now();
        result.compression_time_ms = 
            std::chrono::duration<double, std::milli>(end - start).count();
        result.speed_mbps = (tensor.size() / 1e6) / (result.compression_time_ms / 1000);
        
        metrics.update(result);
        return result;
    }
};
```

### Adaptive Streaming Protocol

```cpp
class AdaptiveStreamingChannel {
private:
    AdaptiveCompressionManager compressor;
    StreamingTDTCompressor tdt_streamer;
    
    // Protocol negotiation
    struct StreamConfig {
        bool compression_supported = true;
        int max_chunk_size = 1024 * 1024;  // 1MB chunks
        CompressionStrategy current_strategy;
    } config;
    
public:
    asio::awaitable<void> send_tensor_adaptive(const Tensor& tensor) {
        // 1. Decide compression strategy
        auto compressed = compressor.compress_adaptive(tensor);
        
        // 2. Send header with metadata
        StreamHeader header{
            .uncompressed_size = tensor.size(),
            .compressed_size = compressed.size(),
            .compression_type = compressed.strategy,
            .tensor_dtype = tensor.dtype(),
            .shape = tensor.shape()
        };
        
        co_await async_write(socket, buffer(&header, sizeof(header)));
        
        // 3. Send compressed data in chunks for streaming
        size_t offset = 0;
        while (offset < compressed.size()) {
            size_t chunk_size = std::min(
                config.max_chunk_size, 
                compressed.size() - offset
            );
            
            co_await async_write(socket, 
                buffer(compressed.data() + offset, chunk_size));
            
            offset += chunk_size;
            
            // 4. Yield periodically for better responsiveness
            if (offset % (10 * config.max_chunk_size) == 0) {
                co_await asio::post(asio::use_awaitable);
            }
        }
    }
};
```

### Auto-Tuning System

```cpp
class CompressionAutoTuner {
    struct Trial {
        CompressionStrategy strategy;
        double total_time_ms;
        double compression_ratio;
        double cpu_usage;
    };
    
    std::vector<Trial> history;
    
public:
    void periodic_tune(AdaptiveCompressionManager& manager) {
        // Run A/B tests periodically
        static int counter = 0;
        if (++counter % 1000 == 0) {  // Every 1000 tensors
            run_benchmark(manager);
        }
    }
    
    void run_benchmark(AdaptiveCompressionManager& manager) {
        // Test each strategy on same data
        Tensor test_tensor = get_representative_tensor();
        
        for (auto strategy : {NO_COMPRESSION, LZ4_FAST, ZSTD_BALANCED, TDT_FULL}) {
            auto start = now();
            auto compressed = compress_with_strategy(test_tensor, strategy);
            auto compress_time = now() - start;
            
            // Simulate network transfer
            auto transfer_time = simulate_transfer(compressed.size());
            
            history.push_back({
                strategy,
                compress_time + transfer_time,
                compressed.ratio(),
                get_cpu_usage()
            });
        }
        
        // Update default strategy based on results
        manager.set_default_strategy(best_strategy_from_history());
    }
};
```

## Integration with Psyne

### Basic Integration
```cpp
// Custom Psyne message type for compressed tensors
class CompressedTensorMessage : public psyne::Message<CompressedTensorMessage> {
public:
    static constexpr uint32_t message_type = 200;
    
    void set_compressed_data(const TDTCompressor::CompressedData& data);
    TDTCompressor::CompressedData get_compressed_data() const;
    
    // Automatically compress/decompress when sending
    void set_tensor(const Tensor& tensor) {
        auto compressed = TDTCompressor().compress(
            tensor.data(), 
            tensor.size() * sizeof(float)
        );
        set_compressed_data(compressed);
    }
    
    Tensor get_tensor() const {
        auto compressed = get_compressed_data();
        auto decompressed = TDTCompressor().decompress(compressed);
        return Tensor::from_bytes(decompressed);
    }
};
```

### Adaptive Compression Integration
```cpp
// Enhanced Psyne channel with adaptive compression
class AdaptiveCompressedChannel {
private:
    psyne::Channel& channel_;
    AdaptiveCompressionManager compression_manager_;
    
public:
    explicit AdaptiveCompressedChannel(psyne::Channel& channel) 
        : channel_(channel) {}
    
    void send_tensor(const Tensor& tensor) {
        // Let adaptive manager decide best strategy
        auto compressed = compression_manager_.compress_adaptive(tensor);
        
        // Create appropriate message type based on strategy
        switch (compressed.strategy) {
            case NO_COMPRESSION:
                send_raw_tensor(tensor);
                break;
            case TDT_FULL:
                send_tdt_compressed(compressed);
                break;
            default:
                send_general_compressed(compressed);
        }
    }
    
    // Network-aware configuration
    void update_network_stats(double bandwidth_mbps, double latency_ms) {
        compression_manager_.update_network_metrics(bandwidth_mbps, latency_ms);
    }
};

// Usage in Psynetics
class PsyneticsLayerCell {
    AdaptiveCompressedChannel output_channel_;
    
    void forward() {
        auto output = compute_forward(input);
        
        // Automatically uses best compression strategy
        output_channel_.send_tensor(output);
        
        // Channel adapts based on conditions
        if (iteration % 100 == 0) {
            output_channel_.update_network_stats(
                measure_bandwidth(),
                measure_latency()
            );
        }
    }
};
```

## Testing Strategy

1. **Compression Ratio Tests**
   - Compare against zstd, lz4, gzip
   - Test on different tensor types (weights, gradients, activations)
   - Measure with different quantization levels

2. **Performance Benchmarks**
   - Compression/decompression speed
   - Memory usage
   - CPU utilization

3. **Correctness Tests**
   - Bit-perfect reconstruction
   - Edge cases (all zeros, denormalized floats, NaN/Inf)

4. **Statistical Validation**
   - Verify feature extraction accuracy
   - Validate clustering quality

## Expected Results

For typical neural network tensors:
- **Weights**: 2-4x compression (better than general-purpose)
- **Gradients**: 3-6x compression (due to small values)
- **Activations**: 4-10x compression (post-ReLU sparsity)

## References

- Original TDT paper: [Include actual paper reference]
- IEEE 754 format: https://en.wikipedia.org/wiki/IEEE_754
- Hierarchical clustering: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

## Key Takeaway

TDT works by recognizing that floating-point tensors aren't random data - they have structure. By understanding and exploiting this structure at the byte level, we can achieve better compression than treating them as generic binary data.