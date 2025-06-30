/**
 * @file simd_demo.cpp
 * @brief Demonstrates SIMD-accelerated tensor operations
 */

#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <random>
#include <vector>

using namespace psyne;
using namespace std::chrono;

// Helper to measure execution time
template <typename F>
double measure_time(F &&func, const std::string &name) {
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    std::cout << name << ": " << duration << " μs" << std::endl;
    return duration;
}

int main() {
    const size_t SIZE = 1'000'000; // 1M elements

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Allocate aligned memory for SIMD operations
    std::vector<float> a(SIZE), b(SIZE), c(SIZE);

    // Fill with random data
    for (size_t i = 0; i < SIZE; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    std::cout << "SIMD Operations Demo\n";
    std::cout << "===================\n";
    std::cout << "Vector size: " << SIZE << " elements\n\n";

    // 1. Vector addition
    std::cout << "1. Vector Addition (C = A + B)\n";
    auto simd_time = measure_time(
        [&]() {
            simd::TensorOps<float>::add(a.data(), b.data(), c.data(), SIZE);
        },
        "  SIMD");

    // Compare with scalar version
    auto scalar_time = measure_time(
        [&]() {
            for (size_t i = 0; i < SIZE; ++i) {
                c[i] = a[i] + b[i];
            }
        },
        "  Scalar");

    std::cout << "  Speedup: " << scalar_time / simd_time << "x\n\n";

    // 2. Vector multiplication
    std::cout << "2. Vector Multiplication (C = A * B)\n";
    simd_time = measure_time(
        [&]() {
            simd::TensorOps<float>::multiply(a.data(), b.data(), c.data(),
                                             SIZE);
        },
        "  SIMD");

    scalar_time = measure_time(
        [&]() {
            for (size_t i = 0; i < SIZE; ++i) {
                c[i] = a[i] * b[i];
            }
        },
        "  Scalar");

    std::cout << "  Speedup: " << scalar_time / simd_time << "x\n\n";

    // 3. Dot product
    std::cout << "3. Dot Product (sum(A[i] * B[i]))\n";
    float dot_result = 0.0f;

    simd_time = measure_time(
        [&]() {
            dot_result =
                simd::TensorOps<float>::dot_product(a.data(), b.data(), SIZE);
        },
        "  SIMD");

    float scalar_dot = 0.0f;
    scalar_time = measure_time(
        [&]() {
            for (size_t i = 0; i < SIZE; ++i) {
                scalar_dot += a[i] * b[i];
            }
        },
        "  Scalar");

    std::cout << "  Result: " << dot_result
              << " (diff: " << std::abs(dot_result - scalar_dot) << ")\n";
    std::cout << "  Speedup: " << scalar_time / simd_time << "x\n\n";

    // 4. Fast copy
    std::cout << "4. Fast Memory Copy\n";
    simd_time = measure_time(
        [&]() { simd::TensorOps<float>::fast_copy(a.data(), c.data(), SIZE); },
        "  SIMD");

    scalar_time = measure_time(
        [&]() {
            for (size_t i = 0; i < SIZE; ++i) {
                c[i] = a[i];
            }
        },
        "  Scalar");

    std::cout << "  Speedup: " << scalar_time / simd_time << "x\n\n";

    // 5. Quantization demo
    std::cout << "5. INT8 Quantization (for neural network compression)\n";
    std::vector<int8_t> quantized(SIZE);
    std::vector<float> dequantized(SIZE);
    float scale = 127.0f; // Quantization scale

    simd_time = measure_time(
        [&]() {
            simd::TensorOps<float>::quantize_int8(a.data(), quantized.data(),
                                                  SIZE, scale);
        },
        "  Quantize (SIMD)");

    measure_time(
        [&]() {
            simd::TensorOps<float>::dequantize_int8(
                quantized.data(), dequantized.data(), SIZE, scale);
        },
        "  Dequantize (SIMD)");

    // Calculate quantization error
    float max_error = 0.0f;
    float avg_error = 0.0f;
    for (size_t i = 0; i < SIZE; ++i) {
        float error = std::abs(a[i] - dequantized[i]);
        max_error = std::max(max_error, error);
        avg_error += error;
    }
    avg_error /= SIZE;

    std::cout << "  Max error: " << max_error << "\n";
    std::cout << "  Avg error: " << avg_error << "\n\n";

    // 6. Tensor layout transformation demo
    std::cout << "6. Tensor Layout Transformation (NCHW <-> NHWC)\n";
    const size_t batch = 4, channels = 3, height = 224, width = 224;
    const size_t tensor_size = batch * channels * height * width;

    std::vector<float> nchw_tensor(tensor_size);
    std::vector<float> nhwc_tensor(tensor_size);

    // Fill NCHW tensor with pattern
    for (size_t n = 0; n < batch; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    size_t idx = n * channels * height * width +
                                 c * height * width + h * width + w;
                    nchw_tensor[idx] = static_cast<float>(c);
                }
            }
        }
    }

    measure_time(
        [&]() {
            simd::LayoutTransform::nchw_to_nhwc(nchw_tensor.data(),
                                                nhwc_tensor.data(), batch,
                                                channels, height, width);
        },
        "  NCHW to NHWC");

    measure_time(
        [&]() {
            simd::LayoutTransform::nhwc_to_nchw(nhwc_tensor.data(),
                                                nchw_tensor.data(), batch,
                                                height, width, channels);
        },
        "  NHWC to NCHW");

    std::cout << "\nSIMD Capabilities:\n";
#ifdef PSYNE_HAS_AVX512
    std::cout << "  ✓ AVX-512 support detected\n";
#elif defined(PSYNE_HAS_AVX2)
    std::cout << "  ✓ AVX2 support detected\n";
#elif defined(PSYNE_HAS_NEON)
    std::cout << "  ✓ NEON support detected\n";
#else
    std::cout << "  ✗ No SIMD support (using scalar fallback)\n";
#endif

    return 0;
}