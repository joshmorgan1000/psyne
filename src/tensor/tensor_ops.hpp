/**
 * @file tensor_ops.hpp
 * @brief AI/ML tensor optimization utilities
 *
 * Provides optimized operations for tensor manipulation including:
 * - Layout transformations (NCHW <-> NHWC)
 * - Fused operations to reduce memory bandwidth
 * - Quantization-aware transport protocols
 * - Specialized compression for model weights
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include "../memory/custom_allocator.hpp"
#include "../simd/simd_ops.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace psyne {
namespace tensor {

/**
 * @brief Tensor memory layout
 */
enum class Layout {
    NCHW,  // Batch, Channels, Height, Width (GPU-friendly)
    NHWC,  // Batch, Height, Width, Channels (CPU/inference-friendly)
    NC,    // Batch, Channels (for 1D/FC layers)
    NCW,   // Batch, Channels, Width (for 1D convolutions)
    NWC,   // Batch, Width, Channels
    NDHWC, // Batch, Depth, Height, Width, Channels (3D)
    NCDHW  // Batch, Channels, Depth, Height, Width (3D)
};

/**
 * @brief Tensor data type
 */
enum class DataType { Float32, Float16, BFloat16, Int32, Int16, Int8, UInt8 };

/**
 * @brief Get size in bytes for data type
 */
constexpr size_t dtype_size(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
    case DataType::Int32:
        return 4;
    case DataType::Float16:
    case DataType::BFloat16:
    case DataType::Int16:
        return 2;
    case DataType::Int8:
    case DataType::UInt8:
        return 1;
    default:
        return 0;
    }
}

/**
 * @brief Tensor shape information
 */
struct TensorShape {
    std::vector<size_t> dims;
    Layout layout = Layout::NCHW;

    size_t total_elements() const {
        size_t total = 1;
        for (size_t dim : dims) {
            total *= dim;
        }
        return total;
    }

    size_t batch_size() const {
        return dims.empty() ? 0 : dims[0];
    }

    size_t channels() const {
        if (layout == Layout::NCHW || layout == Layout::NCDHW) {
            return dims.size() > 1 ? dims[1] : 0;
        } else if (layout == Layout::NHWC || layout == Layout::NDHWC) {
            return dims.size() > 0 ? dims.back() : 0;
        }
        return 0;
    }
};

/**
 * @brief Tensor descriptor for transport
 */
struct TensorDescriptor {
    TensorShape shape;
    DataType dtype = DataType::Float32;
    size_t offset = 0; // Offset in buffer
    size_t stride = 0; // Stride for sliced tensors
    bool is_contiguous = true;

    size_t byte_size() const {
        return shape.total_elements() * dtype_size(dtype);
    }
};

/**
 * @brief Fused tensor operations for reduced memory bandwidth
 */
class FusedOps {
public:
    /**
     * @brief Fused multiply-add: out = a * b + c
     */
    template <typename T>
    static void fused_multiply_add(const T *a, const T *b, const T *c, T *out,
                                   size_t count) {
        // Use SIMD operations from simd_ops.hpp
        simd::TensorOps<T>::multiply(a, b, out, count);
        simd::TensorOps<T>::add(out, c, out, count);
    }

    /**
     * @brief Fused activation with bias: out = activation(x + bias)
     */
    template <typename T>
    static void fused_bias_activation(const T *x, const T *bias, T *out,
                                      size_t batch, size_t channels,
                                      void (*activation)(T *, size_t)) {
        for (size_t b = 0; b < batch; ++b) {
            size_t offset = b * channels;
            // Add bias
            simd::TensorOps<T>::add(x + offset, bias, out + offset, channels);
            // Apply activation
            activation(out + offset, channels);
        }
    }

    /**
     * @brief Fused layer norm: out = (x - mean) / sqrt(var + eps) * gamma +
     * beta
     */
    static void fused_layer_norm(const float *x, float *out, const float *gamma,
                                 const float *beta, size_t batch,
                                 size_t features, float eps = 1e-5f);
};

/**
 * @brief Quantization utilities for neural network compression
 */
class Quantization {
public:
    /**
     * @brief Quantize float32 to int8 with scale and zero point
     */
    static void quantize_symmetric(const float *src, int8_t *dst, size_t count,
                                   float scale) {
        simd::TensorOps<float>::quantize_int8(src, dst, count, scale);
    }

    /**
     * @brief Quantize with asymmetric quantization
     */
    static void quantize_asymmetric(const float *src, uint8_t *dst,
                                    size_t count, float scale,
                                    uint8_t zero_point);

    /**
     * @brief Dynamic quantization - compute scale from data
     */
    static float compute_scale(const float *data, size_t count,
                               int num_bits = 8);

    /**
     * @brief Quantize weights with per-channel scales
     */
    static void quantize_per_channel(const float *weights, int8_t *quantized,
                                     float *scales, size_t channels,
                                     size_t elements_per_channel);
};

/**
 * @brief Tensor compression for transport
 */
class TensorCompression {
public:
    /**
     * @brief Compression method
     */
    enum class Method {
        None,
        Quantization,  // INT8/INT4 quantization
        Sparsity,      // Sparse tensor encoding
        DeltaEncoding, // Delta encoding for sequential data
        MixedPrecision // Different precision for different layers
    };

    /**
     * @brief Compress tensor for transport
     */
    static std::vector<uint8_t> compress(const void *data,
                                         const TensorDescriptor &desc,
                                         Method method = Method::Quantization);

    /**
     * @brief Decompress tensor after transport
     */
    static bool decompress(const uint8_t *compressed, size_t compressed_size,
                           void *output, const TensorDescriptor &desc,
                           Method method = Method::Quantization);

    /**
     * @brief Estimate compression ratio
     */
    static float estimate_compression_ratio(const TensorDescriptor &desc,
                                            Method method);
};

/**
 * @brief Optimized tensor transport
 */
class TensorTransport {
public:
    /**
     * @brief Transport configuration
     */
    struct Config {
        bool enable_compression = true;
        TensorCompression::Method compression_method =
            TensorCompression::Method::Quantization;
        bool enable_layout_optimization = true;
        bool use_huge_pages = true;
        size_t alignment = 64; // Cache line alignment
    };

    /**
     * @brief Prepare tensor for transport
     */
    static std::vector<uint8_t>
    prepare_for_transport(const void *tensor_data, const TensorDescriptor &desc,
                          const Config &config = {});

    /**
     * @brief Receive and reconstruct tensor
     */
    static bool receive_tensor(const uint8_t *transport_data,
                               size_t transport_size, void *output_buffer,
                               TensorDescriptor &desc,
                               const Config &config = {});

    /**
     * @brief Optimize layout for transport
     */
    static Layout optimal_transport_layout(const TensorShape &shape,
                                           bool is_inference = true);
};

/**
 * @brief Tensor memory pool for efficient allocation
 */
class TensorMemoryPool {
public:
    explicit TensorMemoryPool(size_t initial_size = 256 * 1024 * 1024); // 256MB
    ~TensorMemoryPool();

    /**
     * @brief Allocate tensor from pool
     */
    void *allocate(size_t size, size_t alignment = 64);

    /**
     * @brief Deallocate tensor
     */
    void deallocate(void *ptr);

    /**
     * @brief Reset pool (deallocate all)
     */
    void reset();

    /**
     * @brief Get pool statistics
     */
    struct Stats {
        size_t total_size;
        size_t used_size;
        size_t peak_usage;
        size_t allocation_count;
    };

    Stats get_stats() const;

private:
    struct Block {
        void *ptr;
        size_t size;
        bool in_use;
    };

    std::vector<Block> blocks_;
    void *pool_base_;
    size_t pool_size_;
    size_t used_size_;
    size_t peak_usage_;
    size_t allocation_count_;
};

/**
 * @brief Tensor view for zero-copy operations
 */
template <typename T>
class TensorView {
public:
    TensorView() = default;

    TensorView(T *data, const TensorShape &shape)
        : data_(data), shape_(shape) {}

    // Element access
    T &operator()(size_t n, size_t c, size_t h, size_t w) {
        return data_[calculate_offset(n, c, h, w)];
    }

    const T &operator()(size_t n, size_t c, size_t h, size_t w) const {
        return data_[calculate_offset(n, c, h, w)];
    }

    // Slicing
    TensorView<T> slice(size_t dim, size_t start, size_t end) const;

    // Layout transformation
    void transform_layout(Layout new_layout) {
        if (shape_.layout == new_layout)
            return;

        // Allocate temporary buffer
        std::vector<T> temp(shape_.total_elements());

        // Perform transformation
        if (shape_.layout == Layout::NCHW && new_layout == Layout::NHWC) {
            simd::LayoutTransform::nchw_to_nhwc(data_, temp.data(),
                                                shape_.dims[0], shape_.dims[1],
                                                shape_.dims[2], shape_.dims[3]);
        } else if (shape_.layout == Layout::NHWC &&
                   new_layout == Layout::NCHW) {
            simd::LayoutTransform::nhwc_to_nchw(data_, temp.data(),
                                                shape_.dims[0], shape_.dims[1],
                                                shape_.dims[2], shape_.dims[3]);
        }

        // Copy back
        simd::TensorOps<T>::fast_copy(temp.data(), data_, temp.size());
        shape_.layout = new_layout;
    }

    // Data access
    T *data() {
        return data_;
    }
    const T *data() const {
        return data_;
    }

    const TensorShape &shape() const {
        return shape_;
    }
    size_t size() const {
        return shape_.total_elements();
    }

private:
    size_t calculate_offset(size_t n, size_t c, size_t h, size_t w) const {
        if (shape_.layout == Layout::NCHW) {
            return n * shape_.dims[1] * shape_.dims[2] * shape_.dims[3] +
                   c * shape_.dims[2] * shape_.dims[3] + h * shape_.dims[3] + w;
        } else { // NHWC
            return n * shape_.dims[1] * shape_.dims[2] * shape_.dims[3] +
                   h * shape_.dims[2] * shape_.dims[3] + w * shape_.dims[3] + c;
        }
    }

    T *data_ = nullptr;
    TensorShape shape_;
};

} // namespace tensor
} // namespace psyne