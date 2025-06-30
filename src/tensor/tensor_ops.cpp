/**
 * @file tensor_ops.cpp
 * @brief Implementation of AI/ML tensor optimization utilities
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include "tensor_ops.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace psyne {
namespace tensor {

// FusedOps implementations

void FusedOps::fused_layer_norm(const float *x, float *out, const float *gamma,
                                const float *beta, size_t batch,
                                size_t features, float eps) {
    for (size_t b = 0; b < batch; ++b) {
        const float *batch_x = x + b * features;
        float *batch_out = out + b * features;

        // Compute mean
        float sum = 0.0f;
        for (size_t i = 0; i < features; ++i) {
            sum += batch_x[i];
        }
        float mean = sum / features;

        // Compute variance
        float var_sum = 0.0f;
        for (size_t i = 0; i < features; ++i) {
            float diff = batch_x[i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / features;

        // Normalize and scale
        float inv_std = 1.0f / std::sqrt(variance + eps);
        for (size_t i = 0; i < features; ++i) {
            float normalized = (batch_x[i] - mean) * inv_std;
            batch_out[i] = normalized * gamma[i] + beta[i];
        }
    }
}

// Quantization implementations

void Quantization::quantize_asymmetric(const float *src, uint8_t *dst,
                                       size_t count, float scale,
                                       uint8_t zero_point) {
    for (size_t i = 0; i < count; ++i) {
        int32_t quantized =
            static_cast<int32_t>(std::round(src[i] / scale)) + zero_point;
        dst[i] = static_cast<uint8_t>(std::max(0, std::min(255, quantized)));
    }
}

float Quantization::compute_scale(const float *data, size_t count,
                                  int num_bits) {
    // Find min and max
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < count; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    // Compute scale for symmetric quantization
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    int qmax = (1 << (num_bits - 1)) - 1;

    return abs_max / qmax;
}

void Quantization::quantize_per_channel(const float *weights, int8_t *quantized,
                                        float *scales, size_t channels,
                                        size_t elements_per_channel) {
    for (size_t c = 0; c < channels; ++c) {
        const float *channel_weights = weights + c * elements_per_channel;
        int8_t *channel_quantized = quantized + c * elements_per_channel;

        // Compute scale for this channel
        float scale = compute_scale(channel_weights, elements_per_channel, 8);
        scales[c] = scale;

        // Quantize channel
        quantize_symmetric(channel_weights, channel_quantized,
                           elements_per_channel, scale);
    }
}

// TensorCompression implementations

std::vector<uint8_t> TensorCompression::compress(const void *data,
                                                 const TensorDescriptor &desc,
                                                 Method method) {
    std::vector<uint8_t> compressed;

    switch (method) {
    case Method::None: {
        size_t byte_size = desc.byte_size();
        compressed.resize(byte_size);
        const uint8_t *src = static_cast<const uint8_t *>(data);
        // Zero-copy compliant: manual copy
        for (size_t i = 0; i < byte_size; ++i) {
            compressed[i] = src[i];
        }
    } break;

    case Method::Quantization:
        if (desc.dtype == DataType::Float32) {
            const float *float_data = static_cast<const float *>(data);
            size_t count = desc.shape.total_elements();

            // Compute scale
            float scale = Quantization::compute_scale(float_data, count, 8);

            // Reserve space for scale + quantized data
            compressed.resize(sizeof(float) + count);

            // Store scale
            *reinterpret_cast<float *>(compressed.data()) = scale;

            // Quantize data
            int8_t *quantized =
                reinterpret_cast<int8_t *>(compressed.data() + sizeof(float));
            Quantization::quantize_symmetric(float_data, quantized, count,
                                             scale);
        }
        break;

    case Method::DeltaEncoding:
        if (desc.dtype == DataType::Float32) {
            const float *float_data = static_cast<const float *>(data);
            size_t count = desc.shape.total_elements();

            compressed.resize(sizeof(float) + count * sizeof(float));
            float *output = reinterpret_cast<float *>(compressed.data());

            // Store first value
            output[0] = float_data[0];

            // Store deltas
            for (size_t i = 1; i < count; ++i) {
                output[i] = float_data[i] - float_data[i - 1];
            }
        }
        break;

    default:
        break;
    }

    return compressed;
}

bool TensorCompression::decompress(const uint8_t *compressed,
                                   size_t compressed_size, void *output,
                                   const TensorDescriptor &desc,
                                   Method method) {
    switch (method) {
    case Method::None: {
        size_t byte_size = desc.byte_size();
        if (compressed_size != byte_size)
            return false;

        uint8_t *dst = static_cast<uint8_t *>(output);
        // Zero-copy compliant: manual copy
        for (size_t i = 0; i < byte_size; ++i) {
            dst[i] = compressed[i];
        }
    }
        return true;

    case Method::Quantization:
        if (desc.dtype == DataType::Float32) {
            size_t count = desc.shape.total_elements();
            if (compressed_size < sizeof(float) + count)
                return false;

            // Read scale
            float scale = *reinterpret_cast<const float *>(compressed);

            // Dequantize
            const int8_t *quantized =
                reinterpret_cast<const int8_t *>(compressed + sizeof(float));
            float *float_output = static_cast<float *>(output);

            simd::TensorOps<float>::dequantize_int8(quantized, float_output,
                                                    count, scale);
        }
        return true;

    case Method::DeltaEncoding:
        if (desc.dtype == DataType::Float32) {
            size_t count = desc.shape.total_elements();
            if (compressed_size < count * sizeof(float))
                return false;

            const float *deltas = reinterpret_cast<const float *>(compressed);
            float *float_output = static_cast<float *>(output);

            // Reconstruct from deltas
            float_output[0] = deltas[0];
            for (size_t i = 1; i < count; ++i) {
                float_output[i] = float_output[i - 1] + deltas[i];
            }
        }
        return true;

    default:
        return false;
    }
}

float TensorCompression::estimate_compression_ratio(
    const TensorDescriptor &desc, Method method) {
    size_t original_size = desc.byte_size();
    size_t compressed_size = original_size;

    switch (method) {
    case Method::Quantization:
        if (desc.dtype == DataType::Float32) {
            // INT8 quantization: 4x compression + scale overhead
            compressed_size = desc.shape.total_elements() + sizeof(float);
            if (desc.shape.channels() > 1) {
                // Per-channel quantization
                compressed_size += desc.shape.channels() * sizeof(float);
            }
        }
        break;

    case Method::Sparsity:
        // Assume 90% sparsity for neural networks
        compressed_size = original_size * 0.1f;
        break;

    default:
        break;
    }

    return static_cast<float>(original_size) / compressed_size;
}

// TensorTransport implementations

std::vector<uint8_t>
TensorTransport::prepare_for_transport(const void *tensor_data,
                                       const TensorDescriptor &desc,
                                       const Config &config) {
    std::vector<uint8_t> transport_data;

    // Header: descriptor information
    transport_data.resize(sizeof(TensorDescriptor));
    TensorDescriptor *header =
        reinterpret_cast<TensorDescriptor *>(transport_data.data());
    *header = desc;

    // Optimize layout if needed
    if (config.enable_layout_optimization) {
        Layout optimal = optimal_transport_layout(desc.shape);
        if (optimal != desc.shape.layout) {
            // Transform layout for optimal transport
            transform_layout(tensor_data, desc.shape, optimal);
            header->shape.layout = optimal;
        }
    }

    // Compress if enabled
    if (config.enable_compression) {
        auto compressed = TensorCompression::compress(
            tensor_data, desc, config.compression_method);

        // Append compressed data
        transport_data.insert(transport_data.end(), compressed.begin(),
                              compressed.end());
    } else {
        // Raw copy
        size_t offset = transport_data.size();
        transport_data.resize(offset + desc.byte_size());

        const uint8_t *src = static_cast<const uint8_t *>(tensor_data);
        for (size_t i = 0; i < desc.byte_size(); ++i) {
            transport_data[offset + i] = src[i];
        }
    }

    return transport_data;
}

bool TensorTransport::receive_tensor(const uint8_t *transport_data,
                                     size_t transport_size, void *output_buffer,
                                     TensorDescriptor &desc,
                                     const Config &config) {
    if (transport_size < sizeof(TensorDescriptor)) {
        return false;
    }

    // Read header
    const TensorDescriptor *header =
        reinterpret_cast<const TensorDescriptor *>(transport_data);
    desc = *header;

    // Decompress if needed
    const uint8_t *data_start = transport_data + sizeof(TensorDescriptor);
    size_t data_size = transport_size - sizeof(TensorDescriptor);

    if (config.enable_compression) {
        return TensorCompression::decompress(data_start, data_size,
                                             output_buffer, desc,
                                             config.compression_method);
    } else {
        // Raw copy
        if (data_size != desc.byte_size()) {
            return false;
        }

        uint8_t *dst = static_cast<uint8_t *>(output_buffer);
        for (size_t i = 0; i < data_size; ++i) {
            dst[i] = data_start[i];
        }

        return true;
    }
}

Layout TensorTransport::optimal_transport_layout(const TensorShape &shape,
                                                 bool is_inference) {
    // For inference, NHWC is generally better (cache-friendly)
    // For training with GPUs, NCHW is often better

    if (shape.layout == Layout::NCHW || shape.layout == Layout::NHWC) {
        return is_inference ? Layout::NHWC : Layout::NCHW;
    }

    return shape.layout;
}

// TensorMemoryPool implementations

TensorMemoryPool::TensorMemoryPool(size_t initial_size)
    : pool_size_(initial_size), used_size_(0), peak_usage_(0),
      allocation_count_(0) {
    // Allocate pool with huge pages if possible
    memory::AllocationFlags flags;
    flags.use_huge_pages = true;
    flags.zero_memory = false;

    pool_base_ = memory::CustomAllocator::instance().allocate(
        pool_size_, flags, 4096 // Page alignment
    );

    if (!pool_base_) {
        throw std::bad_alloc();
    }
}

TensorMemoryPool::~TensorMemoryPool() {
    if (pool_base_) {
        memory::CustomAllocator::instance().deallocate(pool_base_);
    }
}

void *TensorMemoryPool::allocate(size_t size, size_t alignment) {
    // Align size
    size = (size + alignment - 1) & ~(alignment - 1);

    // Find free block
    for (auto &block : blocks_) {
        if (!block.in_use && block.size >= size) {
            block.in_use = true;
            allocation_count_++;
            used_size_ += size;
            peak_usage_ = std::max(peak_usage_, used_size_);
            return block.ptr;
        }
    }

    // Allocate new block from pool
    if (used_size_ + size > pool_size_) {
        return nullptr; // Pool exhausted
    }

    void *ptr = static_cast<uint8_t *>(pool_base_) + used_size_;
    blocks_.push_back({ptr, size, true});

    allocation_count_++;
    used_size_ += size;
    peak_usage_ = std::max(peak_usage_, used_size_);

    return ptr;
}

void TensorMemoryPool::deallocate(void *ptr) {
    for (auto &block : blocks_) {
        if (block.ptr == ptr && block.in_use) {
            block.in_use = false;
            used_size_ -= block.size;
            return;
        }
    }
}

void TensorMemoryPool::reset() {
    blocks_.clear();
    used_size_ = 0;
}

TensorMemoryPool::Stats TensorMemoryPool::get_stats() const {
    return {pool_size_, used_size_, peak_usage_, allocation_count_};
}

// TensorView implementations

template <typename T>
TensorView<T> TensorView<T>::slice(size_t dim, size_t start, size_t end) const {
    // Validate inputs
    if (dim >= shape.dims.size() || start >= end || end > shape.dims[dim]) {
        throw std::invalid_argument("Invalid slice parameters");
    }
    
    // Create new shape with sliced dimension
    TensorShape sliced_shape = shape;
    sliced_shape.dims[dim] = end - start;
    
    // Calculate offset for the slice
    size_t offset = 0;
    size_t stride = 1;
    for (size_t i = shape.dims.size(); i > dim; --i) {
        stride *= shape.dims[i - 1];
    }
    offset = start * stride * sizeof(T);
    
    // Return view of sliced data
    return TensorView<T>(
        reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(data) + offset),
        sliced_shape
    );
}

// Explicit instantiations
template class TensorView<float>;
template class TensorView<double>;
template class TensorView<int32_t>;
template class TensorView<int8_t>;
template class TensorView<uint8_t>;

} // namespace tensor
} // namespace psyne