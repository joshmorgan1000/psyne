/**
 * @file simd_ops.cpp
 * @brief SIMD operations implementation
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include "simd_ops.hpp"
#include <cstring>

#ifdef __x86_64__
#include <cpuid.h>
#endif

namespace psyne {
namespace simd {

SIMDCapabilities SIMDCapabilities::detect() {
    SIMDCapabilities caps;

#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;

    // Check for SSE2
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        caps.has_sse2 = (edx & (1 << 26)) != 0;

        // Check for AVX
        caps.has_avx = (ecx & (1 << 28)) != 0;
    }

    // Check for AVX2
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        caps.has_avx2 = (ebx & (1 << 5)) != 0;
        caps.has_avx512f = (ebx & (1 << 16)) != 0;
    }
#elif defined(__aarch64__) || defined(__arm__)
    caps.has_neon = true; // NEON is mandatory on AArch64
#endif

    return caps;
}


// Template instantiations for TensorOps
// multiply with different signature is now element-wise multiply in the header
/*
template <>
void TensorOps<float>::multiply(const float *a, const float *b, float *c,
                                size_t count) {
#ifdef __x86_64__
    const size_t simd_width = 16;
    const size_t simd_count = count / simd_width;

    for (size_t i = 0; i < simd_count; ++i) {
        __m512 va = _mm512_loadu_ps(a + i * simd_width);
        __m512 vb = _mm512_loadu_ps(b + i * simd_width);
        __m512 vc = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(c + i * simd_width, vc);
    }

    // Handle remainder
    for (size_t i = simd_count * simd_width; i < count; ++i) {
        c[i] = a[i] * b[i];
    }
#elif defined(__aarch64__)
    const size_t simd_width = 4;
    const size_t simd_count = count / simd_width;

    for (size_t i = 0; i < simd_count; ++i) {
        float32x4_t va = vld1q_f32(a + i * simd_width);
        float32x4_t vb = vld1q_f32(b + i * simd_width);
        float32x4_t vc = vmulq_f32(va, vb);
        vst1q_f32(c + i * simd_width, vc);
    }

    // Handle remainder
    for (size_t i = simd_count * simd_width; i < count; ++i) {
        c[i] = a[i] * b[i];
    }
#else
    // Fallback scalar implementation
    for (size_t i = 0; i < count; ++i) {
        c[i] = a[i] * b[i];
    }
#endif
}
*/

// Additional implementations that don't match header declarations
// These are commented out until header is updated with proper declarations
/*
// scale() implementation - commented out as it's not declared in header
template <>
void TensorOps<float>::scale(const float *a, float alpha, float *b,
                             size_t count) {
#ifdef __x86_64__
    const size_t simd_width = 16;
    const size_t simd_count = count / simd_width;
    __m512 valpha = _mm512_set1_ps(alpha);

    for (size_t i = 0; i < simd_count; ++i) {
        __m512 va = _mm512_loadu_ps(a + i * simd_width);
        __m512 vb = _mm512_mul_ps(va, valpha);
        _mm512_storeu_ps(b + i * simd_width, vb);
    }

    // Handle remainder
    for (size_t i = simd_count * simd_width; i < count; ++i) {
        b[i] = a[i] * alpha;
    }
#elif defined(__aarch64__)
    const size_t simd_width = 4;
    const size_t simd_count = count / simd_width;
    float32x4_t valpha = vdupq_n_f32(alpha);

    for (size_t i = 0; i < simd_count; ++i) {
        float32x4_t va = vld1q_f32(a + i * simd_width);
        float32x4_t vb = vmulq_f32(va, valpha);
        vst1q_f32(b + i * simd_width, vb);
    }

    // Handle remainder
    for (size_t i = simd_count * simd_width; i < count; ++i) {
        b[i] = a[i] * alpha;
    }
#else
    // Fallback scalar implementation
    for (size_t i = 0; i < count; ++i) {
        b[i] = a[i] * alpha;
    }
#endif
}
*/

// matmul() implementation - commented out as it's not declared in header
/*
template <>
void TensorOps<float>::matmul(const float *a, const float *b, float *c,
                              size_t m, size_t k, size_t n) {
    // Simple tiled matrix multiplication with SIMD
    const size_t tile_size = 64; // Cache-friendly tile size

    // Initialize output to zero
    std::memset(c, 0, m * n * sizeof(float));

    // Tiled multiplication
    for (size_t i = 0; i < m; i += tile_size) {
        for (size_t j = 0; j < n; j += tile_size) {
            for (size_t l = 0; l < k; l += tile_size) {
                // Process tile
                size_t tile_m = std::min(tile_size, m - i);
                size_t tile_n = std::min(tile_size, n - j);
                size_t tile_k = std::min(tile_size, k - l);

                for (size_t ti = 0; ti < tile_m; ++ti) {
                    for (size_t tj = 0; tj < tile_n; ++tj) {
                        float sum = c[(i + ti) * n + (j + tj)];

                        // SIMD dot product for this element
                        const float *a_row = a + (i + ti) * k + l;
                        const float *b_col = b + l * n + (j + tj);

                        for (size_t tk = 0; tk < tile_k; ++tk) {
                            sum += a_row[tk] * b_col[tk * n];
                        }

                        c[(i + ti) * n + (j + tj)] = sum;
                    }
                }
            }
        }
    }
}
*/

// transpose_layout() implementation - commented out as it's not declared in header
/*
template <>
void TensorOps<float>::transpose_layout(const float *src, float *dst, size_t n,
                                        size_t c, size_t h, size_t w,
                                        bool nchw_to_nhwc) {
    if (nchw_to_nhwc) {
// NCHW -> NHWC
#pragma omp parallel for collapse(4)
        for (size_t in = 0; in < n; ++in) {
            for (size_t ic = 0; ic < c; ++ic) {
                for (size_t ih = 0; ih < h; ++ih) {
                    for (size_t iw = 0; iw < w; ++iw) {
                        size_t src_idx = ((in * c + ic) * h + ih) * w + iw;
                        size_t dst_idx = ((in * h + ih) * w + iw) * c + ic;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    } else {
// NHWC -> NCHW
#pragma omp parallel for collapse(4)
        for (size_t in = 0; in < n; ++in) {
            for (size_t ih = 0; ih < h; ++ih) {
                for (size_t iw = 0; iw < w; ++iw) {
                    for (size_t ic = 0; ic < c; ++ic) {
                        size_t src_idx = ((in * h + ih) * w + iw) * c + ic;
                        size_t dst_idx = ((in * c + ic) * h + ih) * w + iw;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}
*/

// Compression implementations
size_t SIMDCompression::compress_rle(const uint8_t *src, size_t src_size,
                                     uint8_t *dst, size_t dst_capacity) {
    if (src_size == 0 || dst_capacity < 2)
        return 0;

    size_t dst_pos = 0;
    size_t src_pos = 0;

    while (src_pos < src_size && dst_pos + 2 <= dst_capacity) {
        uint8_t value = src[src_pos];
        size_t count = 1;

        // Count consecutive values
        while (src_pos + count < src_size && src[src_pos + count] == value &&
               count < 255) {
            ++count;
        }

        // Write count and value
        dst[dst_pos++] = static_cast<uint8_t>(count);
        dst[dst_pos++] = value;
        src_pos += count;
    }

    return (src_pos == src_size) ? dst_pos : 0;
}

void SIMDCompression::delta_encode(const float *src, float *dst, size_t count) {
    if (count == 0)
        return;

    dst[0] = src[0]; // First value unchanged

#ifdef __x86_64__
    static SIMDCapabilities caps = SIMDCapabilities::detect();
    size_t i = 1;
    
#ifdef __AVX512F__
    if (caps.has_avx512f) {
        // Use AVX-512 if both compile-time and runtime support available
        const size_t simd_width = 16;
        for (; i + simd_width <= count; i += simd_width) {
            __m512 current = _mm512_loadu_ps(src + i);
            __m512 prev = _mm512_loadu_ps(src + i - 1);
            __m512 delta = _mm512_sub_ps(current, prev);
            _mm512_storeu_ps(dst + i, delta);
        }
    } else
#endif
    if (caps.has_avx2) {
        // Fallback to AVX2
        const size_t simd_width = 8;
        for (; i + simd_width <= count; i += simd_width) {
            __m256 current = _mm256_loadu_ps(src + i);
            __m256 prev = _mm256_loadu_ps(src + i - 1);
            __m256 delta = _mm256_sub_ps(current, prev);
            _mm256_storeu_ps(dst + i, delta);
        }
    }

    // Handle remainder with scalar code
    for (; i < count; ++i) {
        dst[i] = src[i] - src[i - 1];
    }
#else
    // Scalar fallback for non-x86_64
    for (size_t i = 1; i < count; ++i) {
        dst[i] = src[i] - src[i - 1];
    }
#endif
}

void SIMDCompression::quantize_int8(const float *src, int8_t *dst, size_t count,
                                    float scale) {
#ifdef __x86_64__
    static SIMDCapabilities caps = SIMDCapabilities::detect();
    size_t i = 0;
    
#ifdef __AVX512F__
    if (caps.has_avx512f) {
        // Use AVX-512 if both compile-time and runtime support available
        const size_t simd_width = 16;
        const size_t simd_count = count / simd_width;
        __m512 vscale = _mm512_set1_ps(scale);

        for (; i < simd_count * simd_width; i += simd_width) {
            __m512 vsrc = _mm512_loadu_ps(src + i);
            __m512 scaled = _mm512_mul_ps(vsrc, vscale);
            __m512i quantized = _mm512_cvtps_epi32(scaled);

            // Pack to int8
            __m256i low = _mm512_extracti32x8_epi32(quantized, 0);
            __m256i high = _mm512_extracti32x8_epi32(quantized, 1);
            __m256i packed = _mm256_packs_epi32(low, high);
            __m128i result =
                _mm_packs_epi16(_mm256_extracti128_si256(packed, 0),
                                _mm256_extracti128_si256(packed, 1));

            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), result);
        }
    } else
#endif
    if (caps.has_avx2) {
        // Fallback to AVX2
        const size_t simd_width = 8;
        __m256 vscale = _mm256_set1_ps(scale);

        for (; i + simd_width <= count; i += simd_width) {
            __m256 vsrc = _mm256_loadu_ps(src + i);
            __m256 scaled = _mm256_mul_ps(vsrc, vscale);
            __m256i quantized = _mm256_cvtps_epi32(scaled);

            // Pack to int8
            __m128i low = _mm256_extracti128_si256(quantized, 0);
            __m128i high = _mm256_extracti128_si256(quantized, 1);
            __m128i packed = _mm_packs_epi32(low, high);
            packed = _mm_packs_epi16(packed, packed);

            // Store lower 8 bytes
            *reinterpret_cast<int64_t *>(dst + i) = _mm_cvtsi128_si64(packed);
        }
    }

    // Handle remainder with scalar code
    for (; i < count; ++i) {
        int32_t quantized = static_cast<int32_t>(src[i] * scale + 0.5f);
        dst[i] = static_cast<int8_t>(std::max(-128, std::min(127, quantized)));
    }
#else
    // Scalar fallback for non-x86_64
    for (size_t i = 0; i < count; ++i) {
        int32_t quantized = static_cast<int32_t>(src[i] * scale + 0.5f);
        dst[i] = static_cast<int8_t>(std::max(-128, std::min(127, quantized)));
    }
#endif
}

// Checksum implementations
uint32_t SIMDChecksum::crc32(const uint8_t *data, size_t size) {
    uint32_t crc = 0xFFFFFFFF;

#ifdef __x86_64__
    // Use hardware CRC32 instruction if available
    const size_t chunk_size = sizeof(uint64_t);
    const size_t chunks = size / chunk_size;

    for (size_t i = 0; i < chunks; ++i) {
        uint64_t chunk;
        std::memcpy(&chunk, data + i * chunk_size, chunk_size);
        crc = __builtin_ia32_crc32di(crc, chunk);
    }

    // Handle remainder byte by byte
    for (size_t i = chunks * chunk_size; i < size; ++i) {
        crc = __builtin_ia32_crc32qi(crc, data[i]);
    }
#else
    // Software CRC32 fallback
    static const uint32_t crc_table[256] = {
        // CRC32 polynomial table (omitted for brevity)
        // Would include full table in production code
    };

    for (size_t i = 0; i < size; ++i) {
        crc = (crc >> 8) ^ crc_table[(crc ^ data[i]) & 0xFF];
    }
#endif

    return ~crc;
}

uint64_t SIMDChecksum::xxhash64(const uint8_t *data, size_t size) {
    // Simplified xxHash64 implementation
    // In production, would use optimized xxHash library
    const uint64_t prime1 = 11400714785074694791ULL;
    const uint64_t prime2 = 14029467366897019727ULL;
    const uint64_t prime3 = 1609587929392839161ULL;
    const uint64_t prime4 = 9650029242287828579ULL;
    const uint64_t prime5 = 2870177450012600261ULL;

    uint64_t hash = size + prime5;

    // Process 32-byte blocks
    const size_t blocks = size / 32;
    for (size_t i = 0; i < blocks; ++i) {
        const uint64_t *block =
            reinterpret_cast<const uint64_t *>(data + i * 32);
        hash ^= (block[0] * prime1) ^ (block[1] * prime2) ^
                (block[2] * prime3) ^ (block[3] * prime4);
        hash = (hash << 31) | (hash >> 33);
        hash *= prime1;
    }

    // Process remaining bytes
    const size_t offset = blocks * 32;
    for (size_t i = offset; i < size; ++i) {
        hash ^= data[i] * prime5;
        hash = (hash << 11) | (hash >> 53);
        hash *= prime1;
    }

    // Final mixing
    hash ^= hash >> 33;
    hash *= prime2;
    hash ^= hash >> 29;
    hash *= prime3;
    hash ^= hash >> 32;

    return hash;
}

} // namespace simd
} // namespace psyne