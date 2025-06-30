#include "compression.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <psyne/psyne.hpp>

namespace psyne {
namespace compression {

// Simple Compressor Implementation
SimpleCompressor::SimpleCompressor(CompressionType type) : type_(type) {}

size_t SimpleCompressor::compress(const void *src, size_t src_size, void *dst,
                                  size_t dst_capacity) {
    if (!src || !dst || src_size == 0 || dst_capacity == 0) {
        return 0;
    }

    switch (type_) {
    case CompressionType::None:
        if (dst_capacity < src_size)
            return 0;
        std::memcpy(dst, src, src_size);
        return src_size;

    case CompressionType::LZ4:
        // For demonstration, use simple RLE compression
        return compress_rle(static_cast<const uint8_t *>(src), src_size,
                            static_cast<uint8_t *>(dst), dst_capacity);

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
    if (!src || !dst || src_size == 0 || dst_capacity == 0) {
        return 0;
    }

    switch (type_) {
    case CompressionType::None:
        if (dst_capacity < src_size)
            return 0;
        std::memcpy(dst, src, src_size);
        return src_size;

    case CompressionType::LZ4:
        // For demonstration, use simple RLE decompression
        return decompress_rle(static_cast<const uint8_t *>(src), src_size,
                              static_cast<uint8_t *>(dst), dst_capacity);

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
    if (!should_compress(src_size) || !compressor_) {
        return 0;
    }

    size_t max_size = compressor_->max_compressed_size(src_size);
    compressed_buffer.resize(max_size);

    size_t compressed_size = compressor_->compress(
        src, src_size, compressed_buffer.data(), compressed_buffer.size());

    if (compressed_size == 0 || compressed_size >= src_size) {
        // Compression failed or not beneficial
        return 0;
    }

    compressed_buffer.resize(compressed_size);
    return compressed_size;
}

size_t CompressionManager::decompress_message(const void *src, size_t src_size,
                                              void *dst, size_t dst_capacity) {
    if (!compressor_) {
        return 0;
    }

    return compressor_->decompress(src, src_size, dst, dst_capacity);
}

void CompressionManager::set_config(const CompressionConfig &config) {
    if (config.type != config_.type) {
        config_ = config;
        create_compressor();
    } else {
        config_ = config;
    }
}

void CompressionManager::create_compressor() {
    compressor_ = ::psyne::compression::create_compressor(config_.type);
}

// Factory function
std::unique_ptr<Compressor> create_compressor(CompressionType type) {
    return std::make_unique<SimpleCompressor>(type);
}

} // namespace compression
} // namespace psyne