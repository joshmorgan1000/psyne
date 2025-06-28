#include "../../include/psyne/perf/simd_operations.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

#ifdef __x86_64__
    #include <cpuid.h>
#endif

namespace psyne {
namespace perf {

// ============================================================================
// CPU Feature Detection
// ============================================================================

CPUFeatures::CPUFeatures() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    
    // Check for AVX support
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        has_sse4_1 = (ecx & bit_SSE4_1) != 0;
        has_avx = (ecx & bit_AVX) != 0;
    }
    
    // Check for AVX2 support
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        has_avx2 = (ebx & bit_AVX2) != 0;
    }
#elif defined(__aarch64__)
    has_neon = true; // ARM64 always has NEON
#endif
}

const CPUFeatures& get_cpu_features() {
    static CPUFeatures features;
    return features;
}

// ============================================================================
// Vector Operations - Float32
// ============================================================================

void vector_add_f32(const float* a, const float* b, float* result, size_t count) {
    const auto& features = get_cpu_features();
    
#ifdef PSYNE_HAS_AVX2
    if (features.has_avx2) {
        const size_t simd_count = count / 8;
        const size_t remainder = count % 8;
        
        for (size_t i = 0; i < simd_count; ++i) {
            __m256 va = _mm256_loadu_ps(&a[i * 8]);
            __m256 vb = _mm256_loadu_ps(&b[i * 8]);
            __m256 vr = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&result[i * 8], vr);
        }
        
        // Handle remainder
        for (size_t i = simd_count * 8; i < count; ++i) {
            result[i] = a[i] + b[i];
        }
        return;
    }
#endif

#ifdef PSYNE_HAS_SSE4_1
    if (features.has_sse4_1) {
        const size_t simd_count = count / 4;
        const size_t remainder = count % 4;
        
        for (size_t i = 0; i < simd_count; ++i) {
            __m128 va = _mm_loadu_ps(&a[i * 4]);
            __m128 vb = _mm_loadu_ps(&b[i * 4]);
            __m128 vr = _mm_add_ps(va, vb);
            _mm_storeu_ps(&result[i * 4], vr);
        }
        
        // Handle remainder
        for (size_t i = simd_count * 4; i < count; ++i) {
            result[i] = a[i] + b[i];
        }
        return;
    }
#endif

#ifdef PSYNE_HAS_NEON
    if (features.has_neon) {
        const size_t simd_count = count / 4;
        const size_t remainder = count % 4;
        
        for (size_t i = 0; i < simd_count; ++i) {
            float32x4_t va = vld1q_f32(&a[i * 4]);
            float32x4_t vb = vld1q_f32(&b[i * 4]);
            float32x4_t vr = vaddq_f32(va, vb);
            vst1q_f32(&result[i * 4], vr);
        }
        
        // Handle remainder
        for (size_t i = simd_count * 4; i < count; ++i) {
            result[i] = a[i] + b[i];
        }
        return;
    }
#endif

    // Fallback scalar implementation
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void vector_mul_f32(const float* a, const float* b, float* result, size_t count) {
    const auto& features = get_cpu_features();
    
#ifdef PSYNE_HAS_AVX2
    if (features.has_avx2) {
        const size_t simd_count = count / 8;
        
        for (size_t i = 0; i < simd_count; ++i) {
            __m256 va = _mm256_loadu_ps(&a[i * 8]);
            __m256 vb = _mm256_loadu_ps(&b[i * 8]);
            __m256 vr = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(&result[i * 8], vr);
        }
        
        // Handle remainder
        for (size_t i = simd_count * 8; i < count; ++i) {
            result[i] = a[i] * b[i];
        }
        return;
    }
#endif

#ifdef PSYNE_HAS_SSE4_1
    if (features.has_sse4_1) {
        const size_t simd_count = count / 4;
        
        for (size_t i = 0; i < simd_count; ++i) {
            __m128 va = _mm_loadu_ps(&a[i * 4]);
            __m128 vb = _mm_loadu_ps(&b[i * 4]);
            __m128 vr = _mm_mul_ps(va, vb);
            _mm_storeu_ps(&result[i * 4], vr);
        }
        
        // Handle remainder
        for (size_t i = simd_count * 4; i < count; ++i) {
            result[i] = a[i] * b[i];
        }
        return;
    }
#endif

#ifdef PSYNE_HAS_NEON
    if (features.has_neon) {
        const size_t simd_count = count / 4;
        
        for (size_t i = 0; i < simd_count; ++i) {
            float32x4_t va = vld1q_f32(&a[i * 4]);
            float32x4_t vb = vld1q_f32(&b[i * 4]);
            float32x4_t vr = vmulq_f32(va, vb);
            vst1q_f32(&result[i * 4], vr);
        }
        
        // Handle remainder
        for (size_t i = simd_count * 4; i < count; ++i) {
            result[i] = a[i] * b[i];
        }
        return;
    }
#endif

    // Fallback scalar implementation
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

void vector_mul_scalar_f32(const float* a, float scalar, float* result, size_t count) {
    const auto& features = get_cpu_features();
    
#ifdef PSYNE_HAS_AVX2
    if (features.has_avx2) {
        const size_t simd_count = count / 8;
        __m256 vscalar = _mm256_set1_ps(scalar);
        
        for (size_t i = 0; i < simd_count; ++i) {
            __m256 va = _mm256_loadu_ps(&a[i * 8]);
            __m256 vr = _mm256_mul_ps(va, vscalar);
            _mm256_storeu_ps(&result[i * 8], vr);
        }
        
        // Handle remainder
        for (size_t i = simd_count * 8; i < count; ++i) {
            result[i] = a[i] * scalar;
        }
        return;
    }
#endif

#ifdef PSYNE_HAS_SSE4_1
    if (features.has_sse4_1) {
        const size_t simd_count = count / 4;
        __m128 vscalar = _mm_set1_ps(scalar);
        
        for (size_t i = 0; i < simd_count; ++i) {
            __m128 va = _mm_loadu_ps(&a[i * 4]);
            __m128 vr = _mm_mul_ps(va, vscalar);
            _mm_storeu_ps(&result[i * 4], vr);
        }
        
        // Handle remainder
        for (size_t i = simd_count * 4; i < count; ++i) {
            result[i] = a[i] * scalar;
        }
        return;
    }
#endif

#ifdef PSYNE_HAS_NEON
    if (features.has_neon) {
        const size_t simd_count = count / 4;
        float32x4_t vscalar = vdupq_n_f32(scalar);
        
        for (size_t i = 0; i < simd_count; ++i) {
            float32x4_t va = vld1q_f32(&a[i * 4]);
            float32x4_t vr = vmulq_f32(va, vscalar);
            vst1q_f32(&result[i * 4], vr);
        }
        
        // Handle remainder
        for (size_t i = simd_count * 4; i < count; ++i) {
            result[i] = a[i] * scalar;
        }
        return;
    }
#endif

    // Fallback scalar implementation
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * scalar;
    }
}

float vector_dot_f32(const float* a, const float* b, size_t count) {
    const auto& features = get_cpu_features();
    float result = 0.0f;
    
#ifdef PSYNE_HAS_AVX2
    if (features.has_avx2) {
        const size_t simd_count = count / 8;
        __m256 vsum = _mm256_setzero_ps();
        
        for (size_t i = 0; i < simd_count; ++i) {
            __m256 va = _mm256_loadu_ps(&a[i * 8]);
            __m256 vb = _mm256_loadu_ps(&b[i * 8]);
            __m256 vmul = _mm256_mul_ps(va, vb);
            vsum = _mm256_add_ps(vsum, vmul);
        }
        
        // Sum the 8 elements in vsum
        float sum_array[8];
        _mm256_storeu_ps(sum_array, vsum);
        for (int i = 0; i < 8; ++i) {
            result += sum_array[i];
        }
        
        // Handle remainder
        for (size_t i = simd_count * 8; i < count; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
#endif

#ifdef PSYNE_HAS_SSE4_1
    if (features.has_sse4_1) {
        const size_t simd_count = count / 4;
        __m128 vsum = _mm_setzero_ps();
        
        for (size_t i = 0; i < simd_count; ++i) {
            __m128 va = _mm_loadu_ps(&a[i * 4]);
            __m128 vb = _mm_loadu_ps(&b[i * 4]);
            __m128 vmul = _mm_mul_ps(va, vb);
            vsum = _mm_add_ps(vsum, vmul);
        }
        
        // Sum the 4 elements in vsum
        float sum_array[4];
        _mm_storeu_ps(sum_array, vsum);
        for (int i = 0; i < 4; ++i) {
            result += sum_array[i];
        }
        
        // Handle remainder
        for (size_t i = simd_count * 4; i < count; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
#endif

#ifdef PSYNE_HAS_NEON
    if (features.has_neon) {
        const size_t simd_count = count / 4;
        float32x4_t vsum = vdupq_n_f32(0.0f);
        
        for (size_t i = 0; i < simd_count; ++i) {
            float32x4_t va = vld1q_f32(&a[i * 4]);
            float32x4_t vb = vld1q_f32(&b[i * 4]);
            float32x4_t vmul = vmulq_f32(va, vb);
            vsum = vaddq_f32(vsum, vmul);
        }
        
        // Sum the 4 elements in vsum
        float sum_array[4];
        vst1q_f32(sum_array, vsum);
        for (int i = 0; i < 4; ++i) {
            result += sum_array[i];
        }
        
        // Handle remainder
        for (size_t i = simd_count * 4; i < count; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
#endif

    // Fallback scalar implementation
    for (size_t i = 0; i < count; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// ============================================================================
// Stub implementations for other functions (to keep compilation working)
// ============================================================================

void vector_sub_f32(const float* a, const float* b, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] - b[i];
    }
}

void vector_add_f64(const double* a, const double* b, double* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void vector_sub_f64(const double* a, const double* b, double* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] - b[i];
    }
}

void vector_mul_f64(const double* a, const double* b, double* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

void vector_add_scalar_f32(const float* a, float scalar, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + scalar;
    }
}

void vector_add_scalar_f64(const double* a, double scalar, double* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + scalar;
    }
}

void vector_mul_scalar_f64(const double* a, double scalar, double* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * scalar;
    }
}

double vector_dot_f64(const double* a, const double* b, size_t count) {
    double result = 0.0;
    for (size_t i = 0; i < count; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

float vector_magnitude_f32(const float* a, size_t count) {
    return std::sqrt(vector_dot_f32(a, a, count));
}

double vector_magnitude_f64(const double* a, size_t count) {
    return std::sqrt(vector_dot_f64(a, a, count));
}

void vector_normalize_f32(const float* a, float* result, size_t count) {
    float mag = vector_magnitude_f32(a, count);
    if (mag > 0.0f) {
        vector_mul_scalar_f32(a, 1.0f / mag, result, count);
    } else {
        std::memcpy(result, a, count * sizeof(float));
    }
}

void vector_normalize_f64(const double* a, double* result, size_t count) {
    double mag = vector_magnitude_f64(a, count);
    if (mag > 0.0) {
        vector_mul_scalar_f64(a, 1.0 / mag, result, count);
    } else {
        std::memcpy(result, a, count * sizeof(double));
    }
}

// Matrix operations (basic implementations)
void matrix_mul_f32(const float* a, const float* b, float* c, 
                    size_t rows_a, size_t cols_a, size_t cols_b) {
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < cols_b; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_a; ++k) {
                sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            c[i * cols_b + j] = sum;
        }
    }
}

void matrix_mul_f64(const double* a, const double* b, double* c,
                    size_t rows_a, size_t cols_a, size_t cols_b) {
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < cols_b; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols_a; ++k) {
                sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            c[i * cols_b + j] = sum;
        }
    }
}

void matrix_vector_mul_f32(const float* matrix, const float* vector, float* result,
                          size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        result[i] = vector_dot_f32(&matrix[i * cols], vector, cols);
    }
}

void matrix_vector_mul_f64(const double* matrix, const double* vector, double* result,
                          size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        result[i] = vector_dot_f64(&matrix[i * cols], vector, cols);
    }
}

void matrix_transpose_f32(const float* matrix, float* result, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
}

void matrix_transpose_f64(const double* matrix, double* result, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
}

// Memory operations
void simd_memcpy(void* dest, const void* src, size_t size) {
    std::memcpy(dest, src, size);
}

void simd_memset_f32(float* dest, float value, size_t count) {
    std::fill(dest, dest + count, value);
}

void simd_memset_f64(double* dest, double value, size_t count) {
    std::fill(dest, dest + count, value);
}

bool simd_memcmp(const void* a, const void* b, size_t size) {
    return std::memcmp(a, b, size) == 0;
}

// Quantization operations (stub implementations)
void quantize_f32_to_i8(const float* input, int8_t* output, size_t count, 
                       float scale, int8_t zero_point) {
    for (size_t i = 0; i < count; ++i) {
        int32_t quantized = static_cast<int32_t>(std::round(input[i] / scale)) + zero_point;
        output[i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
    }
}

void dequantize_i8_to_f32(const int8_t* input, float* output, size_t count,
                         float scale, int8_t zero_point) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = scale * (input[i] - zero_point);
    }
}

void quantize_f32_to_u8(const float* input, uint8_t* output, size_t count,
                       float scale, uint8_t zero_point) {
    for (size_t i = 0; i < count; ++i) {
        int32_t quantized = static_cast<int32_t>(std::round(input[i] / scale)) + zero_point;
        output[i] = static_cast<uint8_t>(std::clamp(quantized, 0, 255));
    }
}

void dequantize_u8_to_f32(const uint8_t* input, float* output, size_t count,
                         float scale, uint8_t zero_point) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = scale * (input[i] - zero_point);
    }
}

// Complex operations (stub implementations)
void complex_mul_f32(const float* a_real, const float* a_imag,
                     const float* b_real, const float* b_imag,
                     float* result_real, float* result_imag, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result_real[i] = a_real[i] * b_real[i] - a_imag[i] * b_imag[i];
        result_imag[i] = a_real[i] * b_imag[i] + a_imag[i] * b_real[i];
    }
}

void complex_mul_f64(const double* a_real, const double* a_imag,
                     const double* b_real, const double* b_imag,
                     double* result_real, double* result_imag, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result_real[i] = a_real[i] * b_real[i] - a_imag[i] * b_imag[i];
        result_imag[i] = a_real[i] * b_imag[i] + a_imag[i] * b_real[i];
    }
}

void complex_magnitude_f32(const float* real, const float* imag, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = std::sqrt(real[i] * real[i] + imag[i] * imag[i]);
    }
}

void complex_magnitude_f64(const double* real, const double* imag, double* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = std::sqrt(real[i] * real[i] + imag[i] * imag[i]);
    }
}

// Activation functions
void relu_f32(const float* input, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void relu_f64(const double* input, double* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = std::max(0.0, input[i]);
    }
}

void sigmoid_f32(const float* input, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

void sigmoid_f64(const double* input, double* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = 1.0 / (1.0 + std::exp(-input[i]));
    }
}

void tanh_f32(const float* input, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

void tanh_f64(const double* input, double* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

void softmax_f32(const float* input, float* output, size_t count) {
    // Find max for numerical stability
    float max_val = *std::max_element(input, input + count);
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    vector_mul_scalar_f32(output, 1.0f / sum, output, count);
}

void softmax_f64(const double* input, double* output, size_t count) {
    // Find max for numerical stability
    double max_val = *std::max_element(input, input + count);
    
    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    vector_mul_scalar_f64(output, 1.0 / sum, output, count);
}

// Utility functions
bool is_aligned(const void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

size_t get_simd_alignment() {
    const auto& features = get_cpu_features();
    
    if (features.has_avx2) return 32;
    if (features.has_avx || features.has_neon) return 16;
    if (features.has_sse4_1) return 16;
    return 8; // Default alignment
}

void prefetch_memory(const void* ptr, size_t size) {
#ifdef __builtin_prefetch
    const char* p = static_cast<const char*>(ptr);
    const size_t cache_line_size = 64;
    
    for (size_t offset = 0; offset < size; offset += cache_line_size) {
        __builtin_prefetch(p + offset, 0, 3); // Read prefetch, high temporal locality
    }
#else
    (void)ptr;
    (void)size;
#endif
}

size_t get_simd_width_f32() {
    const auto& features = get_cpu_features();
    
    if (features.has_avx2) return 8;
    if (features.has_avx || features.has_sse4_1 || features.has_neon) return 4;
    return 1;
}

size_t get_simd_width_f64() {
    const auto& features = get_cpu_features();
    
    if (features.has_avx2) return 4;
    if (features.has_avx) return 4;
    if (features.has_sse4_1) return 2;
    if (features.has_neon) return 2;
    return 1;
}

} // namespace perf
} // namespace psyne