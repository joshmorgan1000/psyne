/**
 * @file tdt_compression.hpp
 * @brief TDT (Tensor Data Transform) compression protocol
 *
 * This protocol implements the TDT compression algorithm as an intelligent
 * buffer between substrates. It understands tensor data semantically and
 * applies compression when beneficial.
 * 
 * Based on research: https://arxiv.org/html/2506.18062v1
 */

#pragma once

#include "psyne/concepts/protocol_concepts.hpp"
#include "logger.hpp"
#include <vector>
#include <array>
#include <memory>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <atomic>

namespace psyne::protocol {

/**
 * @brief TDT compression configuration
 */
struct TDTConfig {
    float sample_fraction = 0.3f;    // Fraction of data to sample for analysis
    int word_size = 4;               // 4 bytes for float32, 8 for float64
    bool auto_detect_clusters = true;
    int max_clusters = 4;
    bool enable_simd = true;         // Use SIMD optimizations if available
    
    // Adaptive compression thresholds
    double bandwidth_threshold_mbps = 100.0;  // Switch strategies based on bandwidth
    double cpu_usage_threshold = 0.8;         // Disable compression if CPU busy
    size_t min_tensor_size = 1024;            // Don't compress small tensors
};

/**
 * @brief Feature extraction for byte positions in floating-point data
 */
struct ByteFeatures {
    double entropy = 0.0;           // Shannon entropy
    double autocorrelation = 0.0;   // Correlation with previous value
    int unique_count = 0;           // Number of unique byte values
    double mean = 0.0;
    double variance = 0.0;
    std::array<int, 256> histogram; // Byte value distribution
    
    ByteFeatures() {
        histogram.fill(0);
    }
};

/**
 * @brief Compressed data container with metadata
 */
struct TDTEncodedData {
    std::vector<std::vector<uint8_t>> streams;  // Separated byte streams
    std::vector<int> cluster_mapping;           // Which byte goes to which stream
    size_t original_size = 0;
    int word_size = 4;
    double compression_ratio = 1.0;
    
    size_t encoded_size() const {
        size_t total = 0;
        for (const auto& stream : streams) {
            total += stream.size();
        }
        return total + cluster_mapping.size() * sizeof(int) + sizeof(TDTEncodedData);
    }
    
    // Serialization for network transport
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> result;
        
        // Header
        uint32_t magic = 0x54445444; // "TDT"
        result.insert(result.end(), (uint8_t*)&magic, (uint8_t*)&magic + 4);
        
        uint32_t orig_size = static_cast<uint32_t>(original_size);
        result.insert(result.end(), (uint8_t*)&orig_size, (uint8_t*)&orig_size + 4);
        
        uint32_t num_streams = static_cast<uint32_t>(streams.size());
        result.insert(result.end(), (uint8_t*)&num_streams, (uint8_t*)&num_streams + 4);
        
        uint32_t word_sz = static_cast<uint32_t>(word_size);
        result.insert(result.end(), (uint8_t*)&word_sz, (uint8_t*)&word_sz + 4);
        
        // Cluster mapping
        uint32_t mapping_size = static_cast<uint32_t>(cluster_mapping.size());
        result.insert(result.end(), (uint8_t*)&mapping_size, (uint8_t*)&mapping_size + 4);
        result.insert(result.end(), (uint8_t*)cluster_mapping.data(), 
                     (uint8_t*)cluster_mapping.data() + cluster_mapping.size() * sizeof(int));
        
        // Streams
        for (const auto& stream : streams) {
            uint32_t stream_size = static_cast<uint32_t>(stream.size());
            result.insert(result.end(), (uint8_t*)&stream_size, (uint8_t*)&stream_size + 4);
            result.insert(result.end(), stream.begin(), stream.end());
        }
        
        return result;
    }
    
    static TDTEncodedData deserialize(const std::vector<uint8_t>& data) {
        TDTEncodedData result;
        size_t offset = 0;
        
        // Verify magic
        uint32_t magic = *reinterpret_cast<const uint32_t*>(data.data() + offset);
        offset += 4;
        if (magic != 0x54445444) {
            throw std::runtime_error("Invalid TDT magic number");
        }
        
        // Read header
        result.original_size = *reinterpret_cast<const uint32_t*>(data.data() + offset);
        offset += 4;
        
        uint32_t num_streams = *reinterpret_cast<const uint32_t*>(data.data() + offset);
        offset += 4;
        
        result.word_size = *reinterpret_cast<const uint32_t*>(data.data() + offset);
        offset += 4;
        
        // Read cluster mapping
        uint32_t mapping_size = *reinterpret_cast<const uint32_t*>(data.data() + offset);
        offset += 4;
        
        result.cluster_mapping.resize(mapping_size);
        std::memcpy(result.cluster_mapping.data(), data.data() + offset, mapping_size * sizeof(int));
        offset += mapping_size * sizeof(int);
        
        // Read streams
        result.streams.resize(num_streams);
        for (uint32_t i = 0; i < num_streams; ++i) {
            uint32_t stream_size = *reinterpret_cast<const uint32_t*>(data.data() + offset);
            offset += 4;
            
            result.streams[i].resize(stream_size);
            std::memcpy(result.streams[i].data(), data.data() + offset, stream_size);
            offset += stream_size;
        }
        
        result.compression_ratio = static_cast<double>(result.original_size) / result.encoded_size();
        return result;
    }
};

/**
 * @brief TDT Compression Protocol - The intelligent buffer between substrates
 */
class TDTCompressionProtocol {
public:
    explicit TDTCompressionProtocol(const TDTConfig& config = {}) : config_(config) {}
    
    // PROTOCOL CONCEPT IMPLEMENTATION
    
    /**
     * @brief Analyze if data should be transformed
     */
    bool should_transform(void* data, size_t size) {
        // Don't compress small data
        if (size < config_.min_tensor_size) return false;
        
        // Don't compress if CPU is busy
        if (cpu_usage_.load() > config_.cpu_usage_threshold) return false;
        
        // Only compress tensor-like data
        if (!is_tensor_data(data, size)) return false;
        
        // Use compression on slower networks
        return bandwidth_mbps_.load() < config_.bandwidth_threshold_mbps;
    }
    
    /**
     * @brief Analyze data characteristics for compression decisions
     */
    void analyze_data(void* data, size_t size) {
        if (!is_tensor_data(data, size)) return;
        
        // Quick entropy analysis to estimate compressibility
        auto sample_indices = generate_sample_indices(size / config_.word_size);
        
        double total_entropy = 0.0;
        for (int byte_pos = 0; byte_pos < config_.word_size; ++byte_pos) {
            auto features = extract_features(static_cast<uint8_t*>(data), sample_indices, byte_pos);
            total_entropy += features.entropy;
        }
        
        avg_entropy_ = total_entropy / config_.word_size;
        log_debug("TDT: Analyzed tensor data, avg entropy: ", avg_entropy_);
    }
    
    /**
     * @brief Encode data using TDT compression
     */
    std::vector<uint8_t> encode(void* data, size_t size) {
        auto start_time = std::chrono::steady_clock::now();
        
        if (!should_transform(data, size)) {
            // Pass through uncompressed with marker
            std::vector<uint8_t> result(size + 4);
            uint32_t marker = 0x554E4350; // "UNCP" - uncompressed
            std::memcpy(result.data(), &marker, 4);
            std::memcpy(result.data() + 4, data, size);
            return result;
        }
        
        try {
            auto compressed = compress_tdt(static_cast<uint8_t*>(data), size);
            auto result = compressed.serialize();
            
            // Update metrics
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
            last_encode_time_ms_ = duration.count();
            last_compression_ratio_ = compressed.compression_ratio;
            
            log_debug("TDT: Compressed ", size, " -> ", result.size(), " bytes (", 
                     compressed.compression_ratio, "x) in ", last_encode_time_ms_, "ms");
            
            return result;
            
        } catch (const std::exception& e) {
            log_warn("TDT: Compression failed, passing through: ", e.what());
            
            // Fallback to uncompressed
            std::vector<uint8_t> result(size + 4);
            uint32_t marker = 0x554E4350; // "UNCP"
            std::memcpy(result.data(), &marker, 4);
            std::memcpy(result.data() + 4, data, size);
            return result;
        }
    }
    
    /**
     * @brief Decode TDT compressed data
     */
    std::vector<uint8_t> decode(const std::vector<uint8_t>& encoded) {
        auto start_time = std::chrono::steady_clock::now();
        
        if (encoded.size() < 4) {
            throw std::runtime_error("TDT: Invalid encoded data size");
        }
        
        uint32_t magic = *reinterpret_cast<const uint32_t*>(encoded.data());
        
        if (magic == 0x554E4350) { // "UNCP" - uncompressed
            std::vector<uint8_t> result(encoded.begin() + 4, encoded.end());
            return result;
        }
        
        try {
            auto compressed = TDTEncodedData::deserialize(encoded);
            auto result = decompress_tdt(compressed);
            
            // Update metrics
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
            last_decode_time_ms_ = duration.count();
            
            log_debug("TDT: Decompressed ", encoded.size(), " -> ", result.size(), 
                     " bytes in ", last_decode_time_ms_, "ms");
            
            return result;
            
        } catch (const std::exception& e) {
            log_error("TDT: Decompression failed: ", e.what());
            throw;
        }
    }
    
    /**
     * @brief Update network performance metrics
     */
    void update_network_metrics(double bandwidth_mbps, double latency_ms) {
        bandwidth_mbps_.store(bandwidth_mbps);
        latency_ms_.store(latency_ms);
    }
    
    /**
     * @brief Update system performance metrics
     */
    void update_system_metrics(double cpu_usage) {
        cpu_usage_.store(cpu_usage);
    }
    
    // IDENTITY BEHAVIORS
    
    const char* protocol_name() const { return "TDT-Compression"; }
    bool is_lossless() const { return true; }
    double transformation_ratio() const { return last_compression_ratio_; }
    double processing_overhead_ms() const { 
        return (last_encode_time_ms_ + last_decode_time_ms_) / 2.0; 
    }
    
    // Additional TDT-specific metrics
    double get_average_entropy() const { return avg_entropy_; }
    double get_bandwidth_mbps() const { return bandwidth_mbps_.load(); }
    double get_cpu_usage() const { return cpu_usage_.load(); }

private:
    TDTConfig config_;
    std::mt19937 rng_{std::random_device{}()};
    
    // Performance metrics
    std::atomic<double> bandwidth_mbps_{100.0};
    std::atomic<double> latency_ms_{1.0};
    std::atomic<double> cpu_usage_{0.5};
    
    // Compression metrics
    double last_compression_ratio_ = 1.0;
    double last_encode_time_ms_ = 0.0;
    double last_decode_time_ms_ = 0.0;
    double avg_entropy_ = 0.0;
    
    // Core TDT compression implementation (moved from substrate)
    TDTEncodedData compress_tdt(const uint8_t* data, size_t size) {
        if (size == 0 || size % config_.word_size != 0) {
            throw std::invalid_argument("Data size must be multiple of word size");
        }
        
        size_t word_count = size / config_.word_size;
        
        // 1. Sample analysis
        auto sample_indices = generate_sample_indices(word_count);
        
        // 2. Feature extraction per byte position
        std::vector<ByteFeatures> features(config_.word_size);
        for (int byte_pos = 0; byte_pos < config_.word_size; ++byte_pos) {
            features[byte_pos] = extract_features(data, sample_indices, byte_pos);
        }
        
        // 3. Hierarchical clustering
        auto cluster_mapping = perform_clustering(features);
        
        // 4. Byte stream separation
        auto streams = separate_byte_streams(data, size, cluster_mapping);
        
        // 5. Per-stream compression
        compress_streams(streams);
        
        TDTEncodedData result;
        result.streams = std::move(streams);
        result.cluster_mapping = std::move(cluster_mapping);
        result.original_size = size;
        result.word_size = config_.word_size;
        result.compression_ratio = static_cast<double>(size) / result.encoded_size();
        
        return result;
    }
    
    std::vector<uint8_t> decompress_tdt(const TDTEncodedData& compressed) {
        // 1. Decompress individual streams
        auto decompressed_streams = decompress_streams(compressed.streams);
        
        // 2. Recombine byte streams
        return recombine_byte_streams(decompressed_streams, compressed);
    }
    
    bool is_tensor_data(void* data, size_t size) const {
        // Simple heuristic: data is tensor-like if size is multiple of float32/float64
        return (size % 4 == 0) && (size >= 64); // At least 16 floats
    }
    
    // [Include all the TDT algorithm methods from the original implementation]
    // generate_sample_indices, extract_features, perform_clustering, etc.
    // (Moving the implementation details here for brevity)
    
    std::vector<size_t> generate_sample_indices(size_t word_count) {
        size_t sample_count = static_cast<size_t>(word_count * config_.sample_fraction);
        sample_count = std::max(sample_count, size_t(100));
        sample_count = std::min(sample_count, word_count);
        
        std::vector<size_t> indices(word_count);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);
        
        indices.resize(sample_count);
        std::sort(indices.begin(), indices.end());
        return indices;
    }
    
    ByteFeatures extract_features(const uint8_t* data, 
                                 const std::vector<size_t>& sample_indices,
                                 int byte_offset) {
        ByteFeatures features;
        std::vector<uint8_t> byte_values;
        byte_values.reserve(sample_indices.size());
        
        for (size_t word_idx : sample_indices) {
            size_t byte_idx = word_idx * config_.word_size + byte_offset;
            uint8_t byte_val = data[byte_idx];
            byte_values.push_back(byte_val);
            features.histogram[byte_val]++;
        }
        
        features.unique_count = std::count_if(features.histogram.begin(),
                                            features.histogram.end(),
                                            [](int count) { return count > 0; });
        
        features.mean = std::accumulate(byte_values.begin(), byte_values.end(), 0.0) /
                       byte_values.size();
        
        double sum_sq_diff = 0.0;
        for (uint8_t val : byte_values) {
            double diff = val - features.mean;
            sum_sq_diff += diff * diff;
        }
        features.variance = sum_sq_diff / byte_values.size();
        
        features.entropy = calculate_entropy(features.histogram, byte_values.size());
        features.autocorrelation = calculate_autocorrelation(byte_values);
        
        return features;
    }
    
    double calculate_entropy(const std::array<int, 256>& histogram, size_t total_count) {
        double entropy = 0.0;
        for (int count : histogram) {
            if (count > 0) {
                double prob = static_cast<double>(count) / total_count;
                entropy -= prob * std::log2(prob);
            }
        }
        return entropy;
    }
    
    double calculate_autocorrelation(const std::vector<uint8_t>& values) {
        if (values.size() < 2) return 0.0;
        
        double sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0;
        double sum_x2 = 0.0, sum_y2 = 0.0;
        size_t n = values.size() - 1;
        
        for (size_t i = 0; i < n; ++i) {
            double x = values[i];
            double y = values[i + 1];
            sum_xy += x * y;
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }
        
        double numerator = n * sum_xy - sum_x * sum_y;
        double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * 
                                      (n * sum_y2 - sum_y * sum_y));
        
        return denominator > 0 ? numerator / denominator : 0.0;
    }
    
    std::vector<int> perform_clustering(const std::vector<ByteFeatures>& features) {
        std::vector<int> mapping(config_.word_size);
        std::vector<double> entropies;
        
        for (int i = 0; i < config_.word_size; ++i) {
            entropies.push_back(features[i].entropy);
        }
        
        double entropy_threshold = std::accumulate(entropies.begin(), entropies.end(), 0.0) /
                                  entropies.size();
        
        for (int i = 0; i < config_.word_size; ++i) {
            mapping[i] = (entropies[i] > entropy_threshold) ? 1 : 0;
        }
        
        return mapping;
    }
    
    std::vector<std::vector<uint8_t>> separate_byte_streams(const uint8_t* data,
                                                           size_t size,
                                                           const std::vector<int>& cluster_mapping) {
        int max_cluster = *std::max_element(cluster_mapping.begin(), cluster_mapping.end());
        std::vector<std::vector<uint8_t>> streams(max_cluster + 1);
        
        size_t word_count = size / config_.word_size;
        
        for (auto& stream : streams) {
            stream.reserve(word_count);
        }
        
        for (size_t word_idx = 0; word_idx < word_count; ++word_idx) {
            for (int byte_pos = 0; byte_pos < config_.word_size; ++byte_pos) {
                int cluster = cluster_mapping[byte_pos];
                size_t byte_idx = word_idx * config_.word_size + byte_pos;
                streams[cluster].push_back(data[byte_idx]);
            }
        }
        
        return streams;
    }
    
    void compress_streams(std::vector<std::vector<uint8_t>>& streams) {
        for (auto& stream : streams) {
            stream = simple_rle_compress(stream);
        }
    }
    
    std::vector<uint8_t> simple_rle_compress(const std::vector<uint8_t>& data) {
        if (data.empty()) return {};
        
        std::vector<uint8_t> compressed;
        compressed.reserve(data.size());
        
        uint8_t current = data[0];
        uint8_t count = 1;
        
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i] == current && count < 255) {
                count++;
            } else {
                compressed.push_back(count);
                compressed.push_back(current);
                current = data[i];
                count = 1;
            }
        }
        
        compressed.push_back(count);
        compressed.push_back(current);
        
        return compressed;
    }
    
    std::vector<std::vector<uint8_t>> decompress_streams(
        const std::vector<std::vector<uint8_t>>& compressed_streams) {
        std::vector<std::vector<uint8_t>> decompressed;
        decompressed.reserve(compressed_streams.size());
        
        for (const auto& stream : compressed_streams) {
            decompressed.push_back(simple_rle_decompress(stream));
        }
        
        return decompressed;
    }
    
    std::vector<uint8_t> simple_rle_decompress(const std::vector<uint8_t>& compressed) {
        std::vector<uint8_t> decompressed;
        
        for (size_t i = 0; i < compressed.size(); i += 2) {
            if (i + 1 < compressed.size()) {
                uint8_t count = compressed[i];
                uint8_t value = compressed[i + 1];
                
                for (uint8_t j = 0; j < count; ++j) {
                    decompressed.push_back(value);
                }
            }
        }
        
        return decompressed;
    }
    
    std::vector<uint8_t> recombine_byte_streams(
        const std::vector<std::vector<uint8_t>>& streams,
        const TDTEncodedData& compressed) {
        
        std::vector<uint8_t> result(compressed.original_size);
        size_t word_count = compressed.original_size / compressed.word_size;
        
        std::vector<size_t> stream_positions(streams.size(), 0);
        
        for (size_t word_idx = 0; word_idx < word_count; ++word_idx) {
            for (int byte_pos = 0; byte_pos < compressed.word_size; ++byte_pos) {
                int cluster = compressed.cluster_mapping[byte_pos];
                size_t byte_idx = word_idx * compressed.word_size + byte_pos;
                
                if (stream_positions[cluster] < streams[cluster].size()) {
                    result[byte_idx] = streams[cluster][stream_positions[cluster]];
                    stream_positions[cluster]++;
                }
            }
        }
        
        return result;
    }
};

} // namespace psyne::protocol