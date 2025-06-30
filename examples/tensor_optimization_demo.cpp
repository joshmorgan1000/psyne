/**
 * @file tensor_optimization_demo.cpp
 * @brief Demonstrates AI/ML tensor optimization features
 */

#include "../src/tensor/tensor_ops.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <psyne/psyne.hpp>
#include <random>

using namespace psyne;
using namespace psyne::tensor;
using namespace std::chrono;

// Helper to measure time
template <typename F>
double measure_time_us(F &&func, const std::string &name) {
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    std::cout << name << ": " << duration << " μs" << std::endl;
    return duration;
}

// Helper to generate random tensor
void generate_random_tensor(float *data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

int main() {
    std::cout << "Tensor Optimization Demo\n";
    std::cout << "========================\n\n";

    // 1. Layout transformation benchmark
    std::cout << "1. Layout Transformation (NCHW <-> NHWC)\n";

    const size_t batch = 32;
    const size_t channels = 64;
    const size_t height = 224;
    const size_t width = 224;
    const size_t total_elements = batch * channels * height * width;

    // Allocate tensors using custom allocator
    TensorMemoryPool pool(256 * 1024 * 1024); // 256MB pool

    float *nchw_tensor =
        static_cast<float *>(pool.allocate(total_elements * sizeof(float)));
    float *nhwc_tensor =
        static_cast<float *>(pool.allocate(total_elements * sizeof(float)));

    generate_random_tensor(nchw_tensor, total_elements);

    // Create tensor views
    TensorShape shape_nchw{{batch, channels, height, width}, Layout::NCHW};
    TensorShape shape_nhwc{{batch, height, width, channels}, Layout::NHWC};

    TensorView<float> view_nchw(nchw_tensor, shape_nchw);
    TensorView<float> view_nhwc(nhwc_tensor, shape_nhwc);

    // Benchmark transformation
    auto nchw_to_nhwc_time = measure_time_us(
        [&]() {
            simd::LayoutTransform::nchw_to_nhwc(nchw_tensor, nhwc_tensor, batch,
                                                channels, height, width);
        },
        "  NCHW to NHWC");

    auto nhwc_to_nchw_time = measure_time_us(
        [&]() {
            simd::LayoutTransform::nhwc_to_nchw(nhwc_tensor, nchw_tensor, batch,
                                                height, width, channels);
        },
        "  NHWC to NCHW");

    double gb_processed =
        total_elements * sizeof(float) * 2.0 / (1024.0 * 1024.0 * 1024.0);
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
              << gb_processed / (nchw_to_nhwc_time / 1e6) << " GB/s\n\n";

    // 2. Quantization benchmark
    std::cout << "2. INT8 Quantization for Model Compression\n";

    const size_t weight_size = 1024 * 1024; // 1M weights
    float *weights =
        static_cast<float *>(pool.allocate(weight_size * sizeof(float)));
    int8_t *quantized = static_cast<int8_t *>(pool.allocate(weight_size));
    float *dequantized =
        static_cast<float *>(pool.allocate(weight_size * sizeof(float)));

    generate_random_tensor(weights, weight_size);

    // Compute scale
    float scale = Quantization::compute_scale(weights, weight_size, 8);
    std::cout << "  Quantization scale: " << scale << "\n";

    // Benchmark quantization
    auto quant_time = measure_time_us(
        [&]() {
            Quantization::quantize_symmetric(weights, quantized, weight_size,
                                             scale);
        },
        "  Quantization");

    auto dequant_time = measure_time_us(
        [&]() {
            simd::TensorOps<float>::dequantize_int8(quantized, dequantized,
                                                    weight_size, scale);
        },
        "  Dequantization");

    // Calculate error
    float max_error = 0.0f;
    double total_error = 0.0;
    for (size_t i = 0; i < weight_size; ++i) {
        float error = std::abs(weights[i] - dequantized[i]);
        max_error = std::max(max_error, error);
        total_error += error;
    }

    std::cout << "  Max error: " << max_error << "\n";
    std::cout << "  Avg error: " << total_error / weight_size << "\n";
    std::cout << "  Compression ratio: 4x (float32 -> int8)\n\n";

    // 3. Fused operations
    std::cout << "3. Fused Operations\n";

    const size_t vector_size = 1024 * 1024;
    float *a = static_cast<float *>(pool.allocate(vector_size * sizeof(float)));
    float *b = static_cast<float *>(pool.allocate(vector_size * sizeof(float)));
    float *c = static_cast<float *>(pool.allocate(vector_size * sizeof(float)));
    float *result =
        static_cast<float *>(pool.allocate(vector_size * sizeof(float)));

    generate_random_tensor(a, vector_size);
    generate_random_tensor(b, vector_size);
    generate_random_tensor(c, vector_size);

    // Benchmark fused multiply-add vs separate operations
    auto fused_time = measure_time_us(
        [&]() { FusedOps::fused_multiply_add(a, b, c, result, vector_size); },
        "  Fused multiply-add");

    auto separate_time = measure_time_us(
        [&]() {
            simd::TensorOps<float>::multiply(a, b, result, vector_size);
            simd::TensorOps<float>::add(result, c, result, vector_size);
        },
        "  Separate operations");

    std::cout << "  Speedup: "
              << separate_time / static_cast<double>(fused_time) << "x\n\n";

    // 4. Tensor compression
    std::cout << "4. Tensor Compression for Transport\n";

    TensorDescriptor desc;
    desc.shape = {{batch, channels, height, width}, Layout::NCHW};
    desc.dtype = DataType::Float32;

    // Test different compression methods
    auto test_compression = [&](TensorCompression::Method method,
                                const char *name) {
        auto compressed =
            TensorCompression::compress(nchw_tensor, desc, method);
        float ratio = static_cast<float>(desc.byte_size()) / compressed.size();

        std::cout << "  " << name << ":\n";
        std::cout << "    Original: " << desc.byte_size() / 1024 << " KB\n";
        std::cout << "    Compressed: " << compressed.size() / 1024 << " KB\n";
        std::cout << "    Ratio: " << std::fixed << std::setprecision(2)
                  << ratio << "x\n";

        // Test decompression
        float *decompressed =
            static_cast<float *>(pool.allocate(desc.byte_size()));
        bool success = TensorCompression::decompress(
            compressed.data(), compressed.size(), decompressed, desc, method);
        std::cout << "    Decompression: " << (success ? "Success" : "Failed")
                  << "\n";
    };

    test_compression(TensorCompression::Method::None, "No compression");
    test_compression(TensorCompression::Method::Quantization,
                     "INT8 Quantization");
    test_compression(TensorCompression::Method::DeltaEncoding,
                     "Delta Encoding");

    std::cout << "\n5. Tensor Transport Optimization\n";

    TensorTransport::Config transport_config;
    transport_config.enable_compression = true;
    transport_config.compression_method =
        TensorCompression::Method::Quantization;
    transport_config.enable_layout_optimization = true;

    auto transport_time = measure_time_us(
        [&]() {
            // Prepare for transport
            auto transport_data = TensorTransport::prepare_for_transport(
                nchw_tensor, desc, transport_config);

            // Receive and reconstruct
            float *received =
                static_cast<float *>(pool.allocate(desc.byte_size()));
            TensorDescriptor received_desc;
            TensorTransport::receive_tensor(transport_data.data(),
                                            transport_data.size(), received,
                                            received_desc, transport_config);
        },
        "  Full transport cycle");

    std::cout << "  Transport bandwidth: "
              << (desc.byte_size() / 1024.0 / 1024.0) / (transport_time / 1e6)
              << " MB/s\n\n";

    // 6. Memory pool statistics
    std::cout << "6. Tensor Memory Pool Statistics\n";
    auto stats = pool.get_stats();
    std::cout << "  Total size: " << stats.total_size / (1024 * 1024)
              << " MB\n";
    std::cout << "  Used size: " << stats.used_size / (1024 * 1024) << " MB\n";
    std::cout << "  Peak usage: " << stats.peak_usage / (1024 * 1024)
              << " MB\n";
    std::cout << "  Allocations: " << stats.allocation_count << "\n";
    std::cout << "  Efficiency: " << 100.0 * stats.used_size / stats.total_size
              << "%\n";

    // 7. Layer normalization
    std::cout << "\n7. Fused Layer Normalization\n";

    const size_t ln_batch = 32;
    const size_t ln_features = 768; // BERT-like dimensions

    float *ln_input = static_cast<float *>(
        pool.allocate(ln_batch * ln_features * sizeof(float)));
    float *ln_output = static_cast<float *>(
        pool.allocate(ln_batch * ln_features * sizeof(float)));
    float *gamma =
        static_cast<float *>(pool.allocate(ln_features * sizeof(float)));
    float *beta =
        static_cast<float *>(pool.allocate(ln_features * sizeof(float)));

    generate_random_tensor(ln_input, ln_batch * ln_features);
    std::fill(gamma, gamma + ln_features, 1.0f);
    std::fill(beta, beta + ln_features, 0.0f);

    auto ln_time = measure_time_us(
        [&]() {
            FusedOps::fused_layer_norm(ln_input, ln_output, gamma, beta,
                                       ln_batch, ln_features, 1e-5f);
        },
        "  Layer norm");

    double gflops = (ln_batch * ln_features * 4.0) / (ln_time * 1000.0);
    std::cout << "  Performance: " << std::fixed << std::setprecision(2)
              << gflops << " GFLOPS\n";

    return 0;
}