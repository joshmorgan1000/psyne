#pragma once

// SIMD-optimized operations for psyne message types
// Provides vectorized implementations for common mathematical operations

#include <cstdint>
#include <cstddef>

// CPU feature detection
#ifdef __x86_64__
    #include <immintrin.h>
    #ifdef __AVX2__
        #define PSYNE_HAS_AVX2 1
    #endif
    #ifdef __AVX__
        #define PSYNE_HAS_AVX 1
    #endif
    #ifdef __SSE4_1__
        #define PSYNE_HAS_SSE4_1 1
    #endif
#elif defined(__aarch64__)
    #include <arm_neon.h>
    #define PSYNE_HAS_NEON 1
#endif

namespace psyne {
namespace perf {

// CPU feature detection at runtime
struct CPUFeatures {
    bool has_avx2 = false;
    bool has_avx = false;
    bool has_sse4_1 = false;
    bool has_neon = false;
    
    CPUFeatures();
};

// Get global CPU features instance
const CPUFeatures& get_cpu_features();

// ============================================================================
// Vector Operations
// ============================================================================

// Vector addition: result[i] = a[i] + b[i]
void vector_add_f32(const float* a, const float* b, float* result, size_t count);
void vector_add_f64(const double* a, const double* b, double* result, size_t count);

// Vector subtraction: result[i] = a[i] - b[i]
void vector_sub_f32(const float* a, const float* b, float* result, size_t count);
void vector_sub_f64(const double* a, const double* b, double* result, size_t count);

// Vector multiplication: result[i] = a[i] * b[i]
void vector_mul_f32(const float* a, const float* b, float* result, size_t count);
void vector_mul_f64(const double* a, const double* b, double* result, size_t count);

// Scalar operations: result[i] = a[i] + scalar
void vector_add_scalar_f32(const float* a, float scalar, float* result, size_t count);
void vector_add_scalar_f64(const double* a, double scalar, double* result, size_t count);

// Scalar multiplication: result[i] = a[i] * scalar
void vector_mul_scalar_f32(const float* a, float scalar, float* result, size_t count);
void vector_mul_scalar_f64(const double* a, double scalar, double* result, size_t count);

// Dot product: sum(a[i] * b[i])
float vector_dot_f32(const float* a, const float* b, size_t count);
double vector_dot_f64(const double* a, const double* b, size_t count);

// Vector magnitude: sqrt(sum(a[i] * a[i]))
float vector_magnitude_f32(const float* a, size_t count);
double vector_magnitude_f64(const double* a, size_t count);

// Vector normalize: result[i] = a[i] / magnitude(a)
void vector_normalize_f32(const float* a, float* result, size_t count);
void vector_normalize_f64(const double* a, double* result, size_t count);

// ============================================================================
// Matrix Operations  
// ============================================================================

// Matrix multiplication: C = A * B (row-major order)
void matrix_mul_f32(const float* a, const float* b, float* c, 
                    size_t rows_a, size_t cols_a, size_t cols_b);
void matrix_mul_f64(const double* a, const double* b, double* c,
                    size_t rows_a, size_t cols_a, size_t cols_b);

// Matrix-vector multiplication: result = matrix * vector
void matrix_vector_mul_f32(const float* matrix, const float* vector, float* result,
                          size_t rows, size_t cols);
void matrix_vector_mul_f64(const double* matrix, const double* vector, double* result,
                          size_t rows, size_t cols);

// Matrix transpose: result[j*rows + i] = matrix[i*cols + j]
void matrix_transpose_f32(const float* matrix, float* result, size_t rows, size_t cols);
void matrix_transpose_f64(const double* matrix, double* result, size_t rows, size_t cols);

// ============================================================================
// Memory Operations
// ============================================================================

// Fast memory copy with SIMD alignment
void simd_memcpy(void* dest, const void* src, size_t size);

// Fast memory set with SIMD
void simd_memset_f32(float* dest, float value, size_t count);
void simd_memset_f64(double* dest, double value, size_t count);

// Memory compare with early exit
bool simd_memcmp(const void* a, const void* b, size_t size);

// ============================================================================
// Quantization Operations
// ============================================================================

// Quantize float32 to int8
void quantize_f32_to_i8(const float* input, int8_t* output, size_t count, 
                       float scale, int8_t zero_point);

// Dequantize int8 to float32
void dequantize_i8_to_f32(const int8_t* input, float* output, size_t count,
                         float scale, int8_t zero_point);

// Quantize float32 to uint8
void quantize_f32_to_u8(const float* input, uint8_t* output, size_t count,
                       float scale, uint8_t zero_point);

// Dequantize uint8 to float32
void dequantize_u8_to_f32(const uint8_t* input, float* output, size_t count,
                         float scale, uint8_t zero_point);

// ============================================================================
// Complex Number Operations
// ============================================================================

// Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
void complex_mul_f32(const float* a_real, const float* a_imag,
                     const float* b_real, const float* b_imag,
                     float* result_real, float* result_imag, size_t count);

void complex_mul_f64(const double* a_real, const double* a_imag,
                     const double* b_real, const double* b_imag,
                     double* result_real, double* result_imag, size_t count);

// Complex magnitude: sqrt(real^2 + imag^2)
void complex_magnitude_f32(const float* real, const float* imag, float* result, size_t count);
void complex_magnitude_f64(const double* real, const double* imag, double* result, size_t count);

// ============================================================================
// Activation Functions (for ML tensors)
// ============================================================================

// ReLU: max(0, x)
void relu_f32(const float* input, float* output, size_t count);
void relu_f64(const double* input, double* output, size_t count);

// Sigmoid: 1 / (1 + exp(-x))
void sigmoid_f32(const float* input, float* output, size_t count);
void sigmoid_f64(const double* input, double* output, size_t count);

// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
void tanh_f32(const float* input, float* output, size_t count);
void tanh_f64(const double* input, double* output, size_t count);

// Softmax: exp(x[i]) / sum(exp(x[j]))
void softmax_f32(const float* input, float* output, size_t count);
void softmax_f64(const double* input, double* output, size_t count);

// ============================================================================
// Utility Functions
// ============================================================================

// Check if memory is aligned for SIMD operations
bool is_aligned(const void* ptr, size_t alignment);

// Get optimal SIMD alignment for current CPU
size_t get_simd_alignment();

// Prefetch memory for better cache performance
void prefetch_memory(const void* ptr, size_t size);

// Get number of elements that can be processed in a single SIMD operation
size_t get_simd_width_f32();
size_t get_simd_width_f64();

} // namespace perf
} // namespace psyne