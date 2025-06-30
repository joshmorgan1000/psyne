#include "compression.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <psyne/psyne.hpp>
#include "../utils/logger.hpp"

namespace psyne {
namespace compression {

// Simple Compressor Implementation
SimpleCompressor::SimpleCompressor(CompressionType type) : type_(type) {}

size_t SimpleCompressor::compress(const void *src, size_t src_size, void *dst,
                                  size_t dst_capacity) {
    log_debug("Compressing data: type=", static_cast<int>(type_), ", src_size=", src_size, ", dst_capacity=", dst_capacity);
    
    if (!src || !dst || src_size == 0 || dst_capacity == 0) {
        log_warn("Invalid compression parameters: src=", src, ", dst=", dst, ", src_size=", src_size, ", dst_capacity=", dst_capacity);
        return 0;
    }

    switch (type_) {
    case CompressionType::None:
        log_trace("No compression applied");
        if (dst_capacity < src_size) {
            log_warn("Insufficient destination capacity for uncompressed data");
            return 0;
        }
        std::memcpy(dst, src, src_size);
        return src_size;

    case CompressionType::LZ4: {
        log_trace("Applying RLE compression (LZ4 demo implementation)");
        // For demonstration, use simple RLE compression
        auto result = compress_rle(static_cast<const uint8_t *>(src), src_size,
                            static_cast<uint8_t *>(dst), dst_capacity);
        log_debug("RLE compression result: ", result, " bytes (ratio: ", 
                 result > 0 ? (double)result/src_size : 0.0, ")");
        return result;
    }

    default:
        // Fallback to no compression
        if (dst_capacity < src_size)
            return 0;
        std::memcpy(dst, src, src_size);
        return src_size;
    }
}

size_t SimpleCompressor::decompress(const void *src, size_t src_size, void *dst,
                                    size_t dst_capacity) {
    log_debug("Decompressing data: type=", static_cast<int>(type_), ", src_size=", src_size, ", dst_capacity=", dst_capacity);
    
    if (!src || !dst || src_size == 0 || dst_capacity == 0) {
        log_warn("Invalid decompression parameters: src=", src, ", dst=", dst, ", src_size=", src_size, ", dst_capacity=", dst_capacity);
        return 0;
    }

    switch (type_) {
    case CompressionType::None:
        log_trace("No decompression applied");
        if (dst_capacity < src_size) {
            log_warn("Insufficient destination capacity for uncompressed data");
            return 0;
        }
        std::memcpy(dst, src, src_size);
        return src_size;

    case CompressionType::LZ4: {
        log_trace("Applying RLE decompression (LZ4 demo implementation)");
        // For demonstration, use simple RLE decompression
        auto result = decompress_rle(static_cast<const uint8_t *>(src), src_size,
                              static_cast<uint8_t *>(dst), dst_capacity);
        log_debug("RLE decompression result: ", result, " bytes");
        return result;
    }

    default:
        // Fallback to no compression
        if (dst_capacity < src_size)
            return 0;
        std::memcpy(dst, src, src_size);
        return src_size;
    }
}

size_t SimpleCompressor::max_compressed_size(size_t src_size) {
    switch (type_) {
    case CompressionType::None:
        return src_size;
    case CompressionType::LZ4:
        // RLE worst case: every byte is unique, so 2x size + header
        return src_size * 2 + 16;
    default:
        return src_size;
    }
}

size_t SimpleCompressor::compress_rle(const uint8_t *src, size_t src_size,
                                      uint8_t *dst, size_t dst_capacity) {
    if (dst_capacity < 8)
        return 0; // Need space for header

    // Simple RLE: [count][byte] pairs
    const uint8_t *src_end = src + src_size;
    uint8_t *dst_start = dst;
    uint8_t *dst_end = dst + dst_capacity;

    // Write original size for decompression
    *reinterpret_cast<uint32_t *>(dst) = static_cast<uint32_t>(src_size);
    dst += 4;

    while (src < src_end && dst + 2 <= dst_end) {
        uint8_t byte = *src++;
        uint8_t count = 1;

        // Count consecutive identical bytes (max 255)
        while (src < src_end && *src == byte && count < 255) {
            src++;
            count++;
        }

        *dst++ = count;
        *dst++ = byte;
    }

    // Check if compression was beneficial
    size_t compressed_size = dst - dst_start;
    if (compressed_size >= src_size) {
        // Not beneficial, store uncompressed
        if (dst_capacity < src_size + 4)
            return 0;
        std::memcpy(dst_start + 4, src - (src_end - (src - src_size)),
                    src_size);
        *reinterpret_cast<uint32_t *>(dst_start) = 0; // Mark as uncompressed
        return src_size + 4;
    }

    return compressed_size;
}

size_t SimpleCompressor::decompress_rle(const uint8_t *src, size_t src_size,
                                        uint8_t *dst, size_t dst_capacity) {
    if (src_size < 4)
        return 0;

    uint32_t original_size = *reinterpret_cast<const uint32_t *>(src);
    src += 4;
    src_size -= 4;

    if (original_size == 0) {
        // Data was stored uncompressed
        if (dst_capacity < src_size)
            return 0;
        std::memcpy(dst, src, src_size);
        return src_size;
    }

    if (dst_capacity < original_size)
        return 0;

    const uint8_t *src_end = src + src_size;
    uint8_t *dst_start = dst;
    uint8_t *dst_end = dst + original_size;

    while (src + 2 <= src_end && dst < dst_end) {
        uint8_t count = *src++;
        uint8_t byte = *src++;

        if (dst + count > dst_end)
            break; // Prevent overflow

        for (uint8_t i = 0; i < count; ++i) {
            *dst++ = byte;
        }
    }

    return dst - dst_start;
}

// Compression Manager Implementation
CompressionManager::CompressionManager(const CompressionConfig &config)
    : config_(config) {
    create_compressor();
}

bool CompressionManager::should_compress(size_t data_size) const {
    return config_.type != CompressionType::None &&
           data_size >= config_.min_size_threshold;
}

size_t
CompressionManager::compress_message(const void *src, size_t src_size,
                                     std::vector<uint8_t> &compressed_buffer) {
    log_debug("Compression manager compress: src_size=", src_size, ", should_compress=", should_compress(src_size));
    
    if (!should_compress(src_size) || !compressor_) {
        if (!should_compress(src_size)) {
            log_trace("Skipping compression: size ", src_size, " below threshold ", config_.min_size_threshold);
        } else {
            log_warn("No compressor available");
        }
        return 0;
    }

    size_t max_size = compressor_->max_compressed_size(src_size);
    compressed_buffer.resize(max_size);

    size_t compressed_size = compressor_->compress(
        src, src_size, compressed_buffer.data(), compressed_buffer.size());

    if (compressed_size == 0 || compressed_size >= src_size) {
        // Compression failed or not beneficial
        log_debug("Compression not beneficial: compressed_size=", compressed_size, " >= src_size=", src_size);
        return 0;
    }
    
    log_info("Message compressed successfully: ", src_size, " -> ", compressed_size, " bytes (ratio: ", (double)compressed_size/src_size, ")");

    compressed_buffer.resize(compressed_size);
    return compressed_size;
}

size_t CompressionManager::decompress_message(const void *src, size_t src_size,
                                              void *dst, size_t dst_capacity) {
    log_debug("Compression manager decompress: src_size=", src_size, ", dst_capacity=", dst_capacity);
    
    if (!compressor_) {
        log_warn("No compressor available for decompression");
        return 0;
    }

    return compressor_->decompress(src, src_size, dst, dst_capacity);
}

void CompressionManager::set_config(const CompressionConfig &config) {
    log_info("Setting compression config: type=", static_cast<int>(config.type), ", min_threshold=", config.min_size_threshold);
    
    if (config.type != config_.type) {
        log_debug("Compression type changed from ", static_cast<int>(config_.type), " to ", static_cast<int>(config.type), ", recreating compressor");
        config_ = config;
        create_compressor();
    } else {
        config_ = config;
    }
}

void CompressionManager::create_compressor() {
    log_debug("Creating compressor for type: ", static_cast<int>(config_.type));
    compressor_ = ::psyne::compression::create_compressor(config_.type);
    if (compressor_) {
        log_info("Compressor created successfully");
    } else {
        log_error("Failed to create compressor");
    }
}

// Factory function
std::unique_ptr<Compressor> create_compressor(CompressionType type) {
    log_debug("Factory creating compressor for type: ", static_cast<int>(type));
    return std::make_unique<SimpleCompressor>(type);
}

} // namespace compression
} // namespace psyne