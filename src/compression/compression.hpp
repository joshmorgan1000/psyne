#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace psyne {
namespace compression {

// Use the compression types from psyne.hpp when available
#ifndef PSYNE_COMPRESSION_TYPES_DEFINED
#define PSYNE_COMPRESSION_TYPES_DEFINED

/**
 * @enum CompressionType
 * @brief Supported compression algorithms
 */
enum class CompressionType : uint8_t {
    None = 0,  ///< No compression
    LZ4 = 1,   ///< Fast compression/decompression
    Zstd = 2,  ///< Better compression ratio
    Snappy = 3 ///< Google Snappy - balanced speed/ratio
};

/**
 * @struct CompressionConfig
 * @brief Configuration for compression behavior
 */
struct CompressionConfig {
    CompressionType type = CompressionType::None;
    int level = 1; ///< Compression level (algorithm dependent)
    size_t min_size_threshold =
        128;                     ///< Don't compress messages smaller than this
    bool enable_checksum = true; ///< Add checksum for compressed data
};

#endif // PSYNE_COMPRESSION_TYPES_DEFINED

/**
 * @class Compressor
 * @brief Abstract base class for compression algorithms
 */
class Compressor {
public:
    virtual ~Compressor() = default;

    /**
     * @brief Compress data
     * @param src Source data to compress
     * @param src_size Size of source data
     * @param dst Destination buffer (must be pre-allocated)
     * @param dst_capacity Maximum size of destination buffer
     * @return Size of compressed data, or 0 on failure
     */
    virtual size_t compress(const void *src, size_t src_size, void *dst,
                            size_t dst_capacity) = 0;

    /**
     * @brief Decompress data
     * @param src Compressed source data
     * @param src_size Size of compressed data
     * @param dst Destination buffer for decompressed data
     * @param dst_capacity Maximum size of destination buffer
     * @return Size of decompressed data, or 0 on failure
     */
    virtual size_t decompress(const void *src, size_t src_size, void *dst,
                              size_t dst_capacity) = 0;

    /**
     * @brief Get maximum compressed size for given input size
     * @param src_size Size of uncompressed data
     * @return Maximum possible compressed size
     */
    virtual size_t max_compressed_size(size_t src_size) = 0;

    /**
     * @brief Get compression type
     */
    virtual CompressionType type() const = 0;
};

/**
 * @class SimpleCompressor
 * @brief Basic compression implementation using built-in algorithms
 *
 * Uses simple compression schemes that don't require external libraries.
 * Primarily for demonstration - real implementations would use LZ4/Zstd.
 */
class SimpleCompressor : public Compressor {
public:
    explicit SimpleCompressor(CompressionType type = CompressionType::None);

    size_t compress(const void *src, size_t src_size, void *dst,
                    size_t dst_capacity) override;

    size_t decompress(const void *src, size_t src_size, void *dst,
                      size_t dst_capacity) override;

    size_t max_compressed_size(size_t src_size) override;

    CompressionType type() const override {
        return type_;
    }

private:
    CompressionType type_;

    // Simple RLE compression for demonstration
    size_t compress_rle(const uint8_t *src, size_t src_size, uint8_t *dst,
                        size_t dst_capacity);
    size_t decompress_rle(const uint8_t *src, size_t src_size, uint8_t *dst,
                          size_t dst_capacity);
};

/**
 * @class CompressionManager
 * @brief Manages compression for channels
 */
class CompressionManager {
public:
    explicit CompressionManager(const CompressionConfig &config = {});

    /**
     * @brief Check if compression should be applied for given data size
     */
    bool should_compress(size_t data_size) const;

    /**
     * @brief Compress message data
     * @param src Source data
     * @param src_size Size of source data
     * @param compressed_buffer Output buffer for compressed data
     * @return Size of compressed data, or 0 if compression failed/not
     * beneficial
     */
    size_t compress_message(const void *src, size_t src_size,
                            std::vector<uint8_t> &compressed_buffer);

    /**
     * @brief Decompress message data
     * @param src Compressed data
     * @param src_size Size of compressed data
     * @param dst Destination buffer
     * @param dst_capacity Maximum destination size
     * @return Size of decompressed data, or 0 on failure
     */
    size_t decompress_message(const void *src, size_t src_size, void *dst,
                              size_t dst_capacity);

    /**
     * @brief Get current configuration
     */
    const CompressionConfig &config() const {
        return config_;
    }

    /**
     * @brief Update configuration
     */
    void set_config(const CompressionConfig &config);

private:
    CompressionConfig config_;
    std::unique_ptr<Compressor> compressor_;

    void create_compressor();
};

/**
 * @brief Create a compressor for the given type
 */
std::unique_ptr<Compressor> create_compressor(CompressionType type);

} // namespace compression
} // namespace psyne