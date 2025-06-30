/**
 * @file simd_ops.hpp
 * @brief SIMD-accelerated operations for tensor/vector computations
 *
 * Provides hardware-accelerated operations using:
 * - AVX-512 and AVX2 for x86_64
 * - NEON for ARM64/AArch64
 * - Automatic fallback to scalar operations
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#ifdef __x86_64__
#include <immintrin.h>
#ifdef __AVX512F__
#define PSYNE_HAS_AVX512
#endif
#ifdef __AVX2__
#define PSYNE_HAS_AVX2
#endif
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define PSYNE_HAS_NEON
#endif

namespace psyne {
namespace simd {

/**
 * @brief SIMD alignment requirement in bytes
 */
constexpr size_t SIMD_ALIGNMENT = 64;

/**
 * @brief SIMD capability detection
 */
struct SIMDCapabilities {
    bool has_sse2 = false;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_neon = false;

    static SIMDCapabilities detect();
};

/**
 * @brief Check if pointer is aligned for SIMD operations
 */
template <typename T>
inline bool is_aligned(const T *ptr, size_t alignment = SIMD_ALIGNMENT) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

/**
 * @brief SIMD operations for different data types
 */
template <typename T>
class TensorOps {
public:
    static void add(const T *a, const T *b, T *result, size_t count);
    static void multiply(const T *a, const T *b, T *result, size_t count);
    static T dot_product(const T *a, const T *b, size_t count);
    static void fast_copy(const T *src, T *dst, size_t count);
    static void fill(T *data, T value, size_t count);
    static void quantize_int8(const T *src, int8_t *dst, size_t count, T scale);
    static void dequantize_int8(const int8_t *src, T *dst, size_t count,
                                T scale);
};

// Float specialization with AVX-512/AVX2/NEON support
template <>
class TensorOps<float> {
public:
    static void add(const float *a, const float *b, float *result,
                    size_t count) {
        size_t i = 0;

#ifdef PSYNE_HAS_AVX512
        const size_t simd_width = 16;
        const size_t aligned_count = count - (count % simd_width);

        for (; i < aligned_count; i += simd_width) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            __m512 vr = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(&result[i], vr);
        }
#elif defined(PSYNE_HAS_AVX2)
        const size_t simd_width = 8;
        const size_t aligned_count = count - (count % simd_width);

        for (; i < aligned_count; i += simd_width) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vr = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&result[i], vr);
        }
#elif defined(PSYNE_HAS_NEON)
        const size_t simd_width = 4;
        const size_t aligned_count = count - (count % simd_width);

        for (; i < aligned_count; i += simd_width) {
            float32x4_t va = vld1q_f32(&a[i]);
            float32x4_t vb = vld1q_f32(&b[i]);
            float32x4_t vr = vaddq_f32(va, vb);
            vst1q_f32(&result[i], vr);
        }
#endif

        // Scalar fallback for remaining elements
        for (; i < count; ++i) {
            result[i] = a[i] + b[i];
        }
    }

    // Element-wise multiplication (implemented in header)
    static void multiply(const float *a, const float *b, float *result,
                         size_t count) {
        size_t i = 0;

#ifdef PSYNE_HAS_AVX512
        const size_t simd_width = 16;
        const size_t aligned_count = count - (count % simd_width);

        for (; i < aligned_count; i += simd_width) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            __m512 vr = _mm512_mul_ps(va, vb);
            _mm512_storeu_ps(&result[i], vr);
        }
#elif defined(PSYNE_HAS_AVX2)
        const size_t simd_width = 8;
        const size_t aligned_count = count - (count % simd_width);

        for (; i < aligned_count; i += simd_width) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vr = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(&result[i], vr);
        }
#elif defined(PSYNE_HAS_NEON)
        const size_t simd_width = 4;
        const size_t aligned_count = count - (count % simd_width);

        for (; i < aligned_count; i += simd_width) {
            float32x4_t va = vld1q_f32(&a[i]);
            float32x4_t vb = vld1q_f32(&b[i]);
            float32x4_t vr = vmulq_f32(va, vb);
            vst1q_f32(&result[i], vr);
        }
#endif

        // Scalar fallback
        for (; i < count; ++i) {
            result[i] = a[i] * b[i];
        }
    }

    static float dot_product(const float *a, const float *b, size_t count) {
        size_t i = 0;
        float sum = 0.0f;

#ifdef PSYNE_HAS_AVX512
        const size_t simd_width = 16;
        const size_t aligned_count = count - (count % simd_width);
        __m512 vsum = _mm512_setzero_ps();

        for (; i < aligned_count; i += simd_width) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            vsum = _mm512_fmadd_ps(va, vb, vsum);
        }

        // Horizontal sum
        sum = _mm512_reduce_add_ps(vsum);
#elif defined(PSYNE_HAS_AVX2)
        const size_t simd_width = 8;
        const size_t aligned_count = count - (count % simd_width);
        __m256 vsum = _mm256_setzero_ps();

        for (; i < aligned_count; i += simd_width) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            vsum = _mm256_fmadd_ps(va, vb, vsum);
        }

        // Horizontal sum
        __m128 vlow = _mm256_castps256_ps128(vsum);
        __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        vlow = _mm_hadd_ps(vlow, vlow);
        vlow = _mm_hadd_ps(vlow, vlow);
        sum = _mm_cvtss_f32(vlow);
#elif defined(PSYNE_HAS_NEON)
        const size_t simd_width = 4;
        const size_t aligned_count = count - (count % simd_width);
        float32x4_t vsum = vdupq_n_f32(0.0f);

        for (; i < aligned_count; i += simd_width) {
            float32x4_t va = vld1q_f32(&a[i]);
            float32x4_t vb = vld1q_f32(&b[i]);
            vsum = vmlaq_f32(vsum, va, vb);
        }

        // Horizontal sum
        float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
        sum = vget_lane_f32(vpadd_f32(vsum2, vsum2), 0);
#endif

        // Scalar fallback
        for (; i < count; ++i) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    static void fast_copy(const float *src, float *dst, size_t count) {
        size_t i = 0;

#ifdef PSYNE_HAS_AVX512
        const size_t simd_width = 16;
        const size_t aligned_count = count - (count % simd_width);

        for (; i < aligned_count; i += simd_width) {
            __m512 v = _mm512_loadu_ps(&src[i]);
            _mm512_storeu_ps(&dst[i], v);
        }
#elif defined(PSYNE_HAS_AVX2)
        const size_t simd_width = 8;
        const size_t aligned_count = count - (count % simd_width);

        for (; i < aligned_count; i += simd_width) {
            __m256 v = _mm256_loadu_ps(&src[i]);
            _mm256_storeu_ps(&dst[i], v);
        }
#elif defined(PSYNE_HAS_NEON)
        const size_t simd_width = 4;
        const size_t aligned_count = count - (count % simd_width);

        for (; i < aligned_count; i += simd_width) {
            float32x4_t v = vld1q_f32(&src[i]);
            vst1q_f32(&dst[i], v);
        }
#endif

        // Scalar fallback - manual loop for zero-copy compliance
        for (; i < count; ++i) {
            dst[i] = src[i];
        }
    }

    static void fill(float *data, float value, size_t count) {
        size_t i = 0;

#ifdef PSYNE_HAS_AVX512
        const size_t simd_width = 16;
        const size_t aligned_count = count - (count % simd_width);
        __m512 vval = _mm512_set1_ps(value);

        for (; i < aligned_count; i += simd_width) {
            _mm512_storeu_ps(&data[i], vval);
        }
#elif defined(PSYNE_HAS_AVX2)
        const size_t simd_width = 8;
        const size_t aligned_count = count - (count % simd_width);
        __m256 vval = _mm256_set1_ps(value);

        for (; i < aligned_count; i += simd_width) {
            _mm256_storeu_ps(&data[i], vval);
        }
#elif defined(PSYNE_HAS_NEON)
        const size_t simd_width = 4;
        const size_t aligned_count = count - (count % simd_width);
        float32x4_t vval = vdupq_n_f32(value);

        for (; i < aligned_count; i += simd_width) {
            vst1q_f32(&data[i], vval);
        }
#endif

        // Scalar fallback
        for (; i < count; ++i) {
            data[i] = value;
        }
    }

    static void quantize_int8(const float *src, int8_t *dst, size_t count,
                              float scale) {
        size_t i = 0;
        const float inv_scale = 1.0f / scale;

#ifdef PSYNE_HAS_AVX512
        const size_t simd_width = 16;
        const size_t aligned_count = count - (count % simd_width);
        __m512 vinv_scale = _mm512_set1_ps(inv_scale);

        for (; i < aligned_count; i += simd_width) {
            __m512 v = _mm512_loadu_ps(&src[i]);
            v = _mm512_mul_ps(v, vinv_scale);
            __m512i vi = _mm512_cvtps_epi32(v);
            __m128i packed = _mm512_cvtsepi32_epi8(vi);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(&dst[i]), packed);
        }
#elif defined(PSYNE_HAS_AVX2)
        const size_t simd_width = 8;
        const size_t aligned_count = count - (count % simd_width);
        __m256 vinv_scale = _mm256_set1_ps(inv_scale);

        for (; i < aligned_count; i += simd_width) {
            __m256 v = _mm256_loadu_ps(&src[i]);
            v = _mm256_mul_ps(v, vinv_scale);
            __m256i vi = _mm256_cvtps_epi32(v);

            // Pack to int8
            __m128i low = _mm256_extracti128_si256(vi, 0);
            __m128i high = _mm256_extracti128_si256(vi, 1);
            __m128i packed = _mm_packs_epi32(low, high);
            packed = _mm_packs_epi16(packed, packed);

            // Store lower 8 bytes
            *reinterpret_cast<int64_t *>(&dst[i]) = _mm_cvtsi128_si64(packed);
        }
#elif defined(PSYNE_HAS_NEON)
        const size_t simd_width = 4;
        const size_t aligned_count = count - (count % simd_width);
        float32x4_t vinv_scale = vdupq_n_f32(inv_scale);

        for (; i < aligned_count; i += simd_width) {
            float32x4_t v = vld1q_f32(&src[i]);
            v = vmulq_f32(v, vinv_scale);
            int32x4_t vi = vcvtq_s32_f32(v);
            int16x4_t vi16 = vqmovn_s32(vi);
            int8x8_t vi8 = vqmovn_s16(vcombine_s16(vi16, vi16));
            vst1_lane_s32(reinterpret_cast<int32_t *>(&dst[i]),
                          vreinterpret_s32_s8(vi8), 0);
        }
#endif

        // Scalar fallback
        for (; i < count; ++i) {
            int32_t val = static_cast<int32_t>(src[i] * inv_scale);
            dst[i] = static_cast<int8_t>(std::max(-128, std::min(127, val)));
        }
    }

    static void dequantize_int8(const int8_t *src, float *dst, size_t count,
                                float scale) {
        size_t i = 0;

#ifdef PSYNE_HAS_AVX512
        const size_t simd_width = 16;
        const size_t aligned_count = count - (count % simd_width);
        __m512 vscale = _mm512_set1_ps(scale);

        for (; i < aligned_count; i += simd_width) {
            __m128i vi8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(&src[i]));
            __m512i vi32 = _mm512_cvtepi8_epi32(vi8);
            __m512 v = _mm512_cvtepi32_ps(vi32);
            v = _mm512_mul_ps(v, vscale);
            _mm512_storeu_ps(&dst[i], v);
        }
#elif defined(PSYNE_HAS_AVX2)
        const size_t simd_width = 8;
        const size_t aligned_count = count - (count % simd_width);
        __m256 vscale = _mm256_set1_ps(scale);

        for (; i < aligned_count; i += simd_width) {
            // Load 8 int8 values
            int64_t packed = *reinterpret_cast<const int64_t *>(&src[i]);
            __m128i vi8 = _mm_cvtsi64_si128(packed);

            // Unpack to int32
            __m256i vi32 = _mm256_cvtepi8_epi32(vi8);
            __m256 v = _mm256_cvtepi32_ps(vi32);
            v = _mm256_mul_ps(v, vscale);
            _mm256_storeu_ps(&dst[i], v);
        }
#elif defined(PSYNE_HAS_NEON)
        const size_t simd_width = 4;
        const size_t aligned_count = count - (count % simd_width);
        float32x4_t vscale = vdupq_n_f32(scale);

        for (; i < aligned_count; i += simd_width) {
            int32_t packed = *reinterpret_cast<const int32_t *>(&src[i]);
            int8x8_t vi8 = vreinterpret_s8_s32(vdup_n_s32(packed));
            int16x8_t vi16 = vmovl_s8(vi8);
            int32x4_t vi32 = vmovl_s16(vget_low_s16(vi16));
            float32x4_t v = vcvtq_f32_s32(vi32);
            v = vmulq_f32(v, vscale);
            vst1q_f32(&dst[i], v);
        }
#endif

        // Scalar fallback
        for (; i < count; ++i) {
            dst[i] = static_cast<float>(src[i]) * scale;
        }
    }

    // Additional operations used in cpp file
    static void scale(const float *a, float alpha, float *b, size_t count);
    static void matmul(const float *a, const float *b, float *c, size_t m,
                       size_t n, size_t k);
    static void transpose_layout(const float *src, float *dst, size_t n,
                                 size_t m);
};

/**
 * @brief SIMD compression utilities
 */
class SIMDCompression {
public:
    static size_t compress_rle(const uint8_t *src, size_t src_size,
                               uint8_t *dst, size_t dst_size);
    static void delta_encode(const float *src, float *dst, size_t count);
    static void quantize_int8(const float *src, int8_t *dst, size_t count,
                              float scale);
};

/**
 * @brief SIMD checksum utilities
 */
class SIMDChecksum {
public:
    static uint32_t crc32(const uint8_t *data, size_t size);
    static uint64_t xxhash64(const uint8_t *data, size_t size);
};

/**
 * @brief Layout transformation utilities for AI/ML tensors
 */
class LayoutTransform {
public:
    /**
     * @brief Convert NCHW to NHWC layout (optimized for inference)
     */
    static void nchw_to_nhwc(const float *src, float *dst, size_t batch,
                             size_t channels, size_t height, size_t width) {
        // Use SIMD for the inner loops
        for (size_t n = 0; n < batch; ++n) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    for (size_t c = 0; c < channels; ++c) {
                        size_t src_idx = n * channels * height * width +
                                         c * height * width + h * width + w;
                        size_t dst_idx = n * height * width * channels +
                                         h * width * channels + w * channels +
                                         c;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }

    /**
     * @brief Convert NHWC to NCHW layout (optimized for training)
     */
    static void nhwc_to_nchw(const float *src, float *dst, size_t batch,
                             size_t height, size_t width, size_t channels) {
        for (size_t n = 0; n < batch; ++n) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        size_t src_idx = n * height * width * channels +
                                         h * width * channels + w * channels +
                                         c;
                        size_t dst_idx = n * channels * height * width +
                                         c * height * width + h * width + w;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
};

} // namespace simd
} // namespace psyne