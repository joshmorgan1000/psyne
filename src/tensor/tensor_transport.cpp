/**
 * @file tensor_transport.cpp
 * @brief Tensor transport implementation
 *
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include "tensor_transport.hpp"
#include "../utils/checksum.hpp"
#include "../simd/simd_ops.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>

namespace psyne {
namespace tensor {

// Helper functions
namespace {

void write_header(std::vector<uint8_t> &buffer, size_t &offset,
                  const void *data, size_t size) {
    if (offset + size > buffer.size()) {
        buffer.resize(offset + size);
    }
    std::memcpy(buffer.data() + offset, data, size);
    offset += size;
}

void read_header(const std::vector<uint8_t> &buffer, size_t &offset, void *data,
                 size_t size) {
    if (offset + size > buffer.size()) {
        throw std::runtime_error("Buffer underflow");
    }
    std::memcpy(data, buffer.data() + offset, size);
    offset += size;
}

} // anonymous namespace

// TensorMessage implementation

void TensorMessage::allocate() {
    size_t bytes = shape_.bytes(dtype_);
    if (bytes > 0) {
        // Use custom allocator with huge pages for large tensors
        bool use_huge_pages = bytes >= 2 * 1024 * 1024;
        data_ = memory::UniqueAlloc(
            bytes,
            use_huge_pages
                ? memory::AllocFlags::HugePage | memory::AllocFlags::Zeroed
                : memory::AllocFlags::Aligned64 | memory::AllocFlags::Zeroed);
    }
}

std::vector<uint8_t> TensorMessage::serialize() const {
    std::vector<uint8_t> buffer;
    size_t offset = 0;

    // Calculate total size
    size_t name_len = name_.size();
    size_t header_size =
        sizeof(uint32_t) * 4 + // magic, version, flags, name_len
        name_len + sizeof(TensorShape::MAX_DIMS) +
        sizeof(size_t) * shape_.rank() +
        sizeof(uint8_t) * 4 + // dtype, layout, compression, padding
        sizeof(QuantizationParams);

    size_t data_size = compression_ != CompressionMethod::None
                           ? compressed_data_.size()
                           : shape_.bytes(dtype_);

    buffer.reserve(header_size + data_size + sizeof(uint32_t)); // +checksum

    // Write header
    uint32_t magic = 0x54454E53; // "TENS"
    uint32_t version = 1;
    uint32_t flags = 0;
    if (compression_ != CompressionMethod::None)
        flags |= 1;
    if (quant_params_.target_type != DataType::Float32)
        flags |= 2;

    write_header(buffer, offset, &magic, sizeof(magic));
    write_header(buffer, offset, &version, sizeof(version));
    write_header(buffer, offset, &flags, sizeof(flags));
    write_header(buffer, offset, &name_len, sizeof(name_len));
    write_header(buffer, offset, name_.data(), name_len);

    // Write shape
    size_t rank = shape_.rank();
    write_header(buffer, offset, &rank, sizeof(rank));
    write_header(buffer, offset, shape_.dims().data(), sizeof(size_t) * rank);

    // Write type info
    write_header(buffer, offset, &dtype_, sizeof(dtype_));
    write_header(buffer, offset, &layout_, sizeof(layout_));
    write_header(buffer, offset, &compression_, sizeof(compression_));
    uint8_t padding = 0;
    write_header(buffer, offset, &padding, sizeof(padding));

    // Write quantization params if needed
    if (flags & 2) {
        write_header(buffer, offset, &quant_params_, sizeof(quant_params_));
    }

    // Write data
    if (compression_ != CompressionMethod::None) {
        uint32_t compressed_size =
            static_cast<uint32_t>(compressed_data_.size());
        write_header(buffer, offset, &compressed_size, sizeof(compressed_size));
        write_header(buffer, offset, compressed_data_.data(), compressed_size);
    } else {
        write_header(buffer, offset, data_.get(), data_size);
    }

    // Write checksum
    // Use SIMD CRC32 for checksum  
    uint32_t checksum = simd::SIMDChecksum::crc32(buffer.data(), offset);
    write_header(buffer, offset, &checksum, sizeof(checksum));

    buffer.resize(offset);
    return buffer;
}

bool TensorMessage::deserialize(const std::vector<uint8_t> &buffer) {
    try {
        size_t offset = 0;

        // Read and verify header
        uint32_t magic, version, flags, name_len;
        read_header(buffer, offset, &magic, sizeof(magic));
        read_header(buffer, offset, &version, sizeof(version));
        read_header(buffer, offset, &flags, sizeof(flags));
        read_header(buffer, offset, &name_len, sizeof(name_len));

        if (magic != 0x54454E53 || version != 1) {
            return false;
        }

        // Read name
        name_.resize(name_len);
        read_header(buffer, offset, name_.data(), name_len);

        // Read shape
        size_t rank;
        read_header(buffer, offset, &rank, sizeof(rank));
        if (rank > TensorShape::MAX_DIMS)
            return false;

        std::array<size_t, TensorShape::MAX_DIMS> dims{};
        read_header(buffer, offset, dims.data(), sizeof(size_t) * rank);

        std::vector<size_t> shape_vec(dims.begin(), dims.begin() + rank);
        shape_ = TensorShape(shape_vec);

        // Read type info
        read_header(buffer, offset, &dtype_, sizeof(dtype_));
        read_header(buffer, offset, &layout_, sizeof(layout_));
        read_header(buffer, offset, &compression_, sizeof(compression_));

        uint8_t padding;
        read_header(buffer, offset, &padding, sizeof(padding));

        // Read quantization params if present
        if (flags & 2) {
            read_header(buffer, offset, &quant_params_, sizeof(quant_params_));
        }

        // Allocate tensor
        allocate();

        // Read data
        if (compression_ != CompressionMethod::None) {
            uint32_t compressed_size;
            read_header(buffer, offset, &compressed_size,
                        sizeof(compressed_size));
            compressed_data_.resize(compressed_size);
            read_header(buffer, offset, compressed_data_.data(),
                        compressed_size);
            decompress();
        } else {
            size_t data_size = shape_.bytes(dtype_);
            read_header(buffer, offset, data_.get(), data_size);
        }

        // Verify checksum
        uint32_t expected_checksum;
        read_header(buffer, offset, &expected_checksum,
                    sizeof(expected_checksum));
        uint32_t actual_checksum =
            utils::xxhash32(buffer.data(), offset - sizeof(uint32_t));

        return expected_checksum == actual_checksum;

    } catch (const std::exception &) {
        return false;
    }
}

void TensorMessage::transform_layout(Layout new_layout) {
    if (layout_ == new_layout)
        return;

    // Currently only support NCHW <-> NHWC transformations
    if ((layout_ == Layout::NCHW && new_layout == Layout::NHWC) ||
        (layout_ == Layout::NHWC && new_layout == Layout::NCHW)) {
        if (shape_.rank() != 4) {
            throw std::runtime_error(
                "Layout transformation requires 4D tensor");
        }

        size_t n = shape_.dim(0);
        size_t c = layout_ == Layout::NCHW ? shape_.dim(1) : shape_.dim(3);
        size_t h = layout_ == Layout::NCHW ? shape_.dim(2) : shape_.dim(1);
        size_t w = layout_ == Layout::NCHW ? shape_.dim(3) : shape_.dim(2);

        // Allocate temporary buffer
        memory::UniqueAlloc temp(size_bytes(), memory::AllocFlags::Aligned64);

        // Perform transformation using SIMD
        if (dtype_ == DataType::Float32) {
            simd::TensorOps<float>::transpose_layout(
                static_cast<const float *>(data_.get()),
                static_cast<float *>(temp.get()), n, c, h, w,
                layout_ == Layout::NCHW);
        } else if (dtype_ == DataType::FLOAT64) {
            // Optimized version for double
            nchw_to_nhwc_optimized(
                static_cast<const double *>(data_.get()),
                static_cast<double *>(temp.get()), n, c, h, w,
                layout_ == Layout::NCHW);
        } else if (dtype_ == DataType::INT32) {
            // Optimized version for int32
            nchw_to_nhwc_optimized(
                static_cast<const int32_t *>(data_.get()),
                static_cast<int32_t *>(temp.get()), n, c, h, w,
                layout_ == Layout::NCHW);
        } else {
            // Generic transformation for remaining types
            size_t element_size = get_dtype_size(dtype_);
            generic_layout_transform(data_.get(), temp.get(), 
                                   shape_.dims, layout_, new_layout, element_size);
        }

        // Swap buffers
        data_ = std::move(temp);
        layout_ = new_layout;

        // Update shape dimensions
        if (new_layout == Layout::NHWC) {
            shape_ = TensorShape({n, h, w, c});
        } else {
            shape_ = TensorShape({n, c, h, w});
        }
    } else {
        throw std::runtime_error("Unsupported layout transformation");
    }
}

void TensorMessage::quantize(const QuantizationParams &params) {
    if (dtype_ != DataType::Float32) {
        throw std::runtime_error(
            "Quantization only supported for Float32 tensors");
    }

    size_t num_elements = shape_.total_elements();
    const float *src = static_cast<const float *>(data_.get());

    // Allocate quantized buffer
    memory::UniqueAlloc quantized(num_elements, memory::AllocFlags::Aligned64);
    int8_t *dst = static_cast<int8_t *>(quantized.get());

    // Perform quantization using SIMD
    simd::SIMDCompression::quantize_int8(src, dst, num_elements, params.scale);

    // Update tensor properties
    data_ = std::move(quantized);
    dtype_ = DataType::Int8;
    quant_params_ = params;
}

void TensorMessage::compress(CompressionMethod method) {
    switch (method) {
    case CompressionMethod::Sparse: {
        // Sparse compression for tensors with many zeros
        size_t num_elements = shape_.total_elements();
        std::vector<uint32_t> indices;
        std::vector<uint8_t> values;
        
        if (dtype_ == DataType::Float32) {
            const float* src = static_cast<const float*>(data_.get());
            for (size_t i = 0; i < num_elements; ++i) {
                if (src[i] != 0.0f) {
                    indices.push_back(static_cast<uint32_t>(i));
                    const uint8_t* value_ptr = reinterpret_cast<const uint8_t*>(&src[i]);
                    values.insert(values.end(), value_ptr, value_ptr + sizeof(float));
                }
            }
        } else if (dtype_ == DataType::Float64) {
            const double* src = static_cast<const double*>(data_.get());
            for (size_t i = 0; i < num_elements; ++i) {
                if (src[i] != 0.0) {
                    indices.push_back(static_cast<uint32_t>(i));
                    const uint8_t* value_ptr = reinterpret_cast<const uint8_t*>(&src[i]);
                    values.insert(values.end(), value_ptr, value_ptr + sizeof(double));
                }
            }
        }
        
        // Store compressed data: [num_nonzero][indices][values]
        compressed_data_.clear();
        uint32_t num_nonzero = static_cast<uint32_t>(indices.size());
        compressed_data_.resize(sizeof(uint32_t) + indices.size() * sizeof(uint32_t) + values.size());
        
        size_t offset = 0;
        std::memcpy(compressed_data_.data(), &num_nonzero, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        if (num_nonzero > 0) {
            std::memcpy(compressed_data_.data() + offset, indices.data(), indices.size() * sizeof(uint32_t));
            offset += indices.size() * sizeof(uint32_t);
            
            std::memcpy(compressed_data_.data() + offset, values.data(), values.size());
        }
        
        break;
    }

    case CompressionMethod::Quantized:
        if (dtype_ == DataType::Float32) {
            quantize(QuantizationParams{127.0f, 0, DataType::Int8, true});
        }
        break;

    case CompressionMethod::Delta:
        // Delta encoding handled separately
        break;

    default:
        break;
    }
}

void TensorMessage::decompress() {
    if (compression_ == CompressionMethod::None || compressed_data_.empty()) {
        return;
    }
    
    switch (compression_) {
    case CompressionMethod::Sparse: {
        // Decompress sparse format: [num_nonzero][indices][values]
        size_t offset = 0;
        uint32_t num_nonzero;
        std::memcpy(&num_nonzero, compressed_data_.data(), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Allocate output tensor and zero it
        allocate();
        
        if (num_nonzero > 0) {
            // Read indices
            std::vector<uint32_t> indices(num_nonzero);
            std::memcpy(indices.data(), compressed_data_.data() + offset, num_nonzero * sizeof(uint32_t));
            offset += num_nonzero * sizeof(uint32_t);
            
            // Read and place values
            if (dtype_ == DataType::Float32) {
                float* dst = static_cast<float*>(data_.get());
                for (uint32_t i = 0; i < num_nonzero; ++i) {
                    float value;
                    std::memcpy(&value, compressed_data_.data() + offset, sizeof(float));
                    offset += sizeof(float);
                    dst[indices[i]] = value;
                }
            } else if (dtype_ == DataType::Float64) {
                double* dst = static_cast<double*>(data_.get());
                for (uint32_t i = 0; i < num_nonzero; ++i) {
                    double value;
                    std::memcpy(&value, compressed_data_.data() + offset, sizeof(double));
                    offset += sizeof(double);
                    dst[indices[i]] = value;
                }
            }
        }
        break;
    }
    
    case CompressionMethod::Quantized:
        // Quantized data is handled during deserialization
        // The data_ already contains dequantized values
        break;
        
    case CompressionMethod::Delta:
        // Delta encoding is handled separately via apply_delta_encoding
        break;
        
    default:
        break;
    }
    
    compression_ = CompressionMethod::None;
    compressed_data_.clear();
}

void TensorMessage::apply_delta_encoding(const TensorMessage *base) {
    if (!base || shape_.total_elements() != base->shape_.total_elements() ||
        dtype_ != base->dtype_) {
        return;
    }

    if (dtype_ == DataType::Float32) {
        float *data = static_cast<float *>(data_.get());
        const float *base_data = static_cast<const float *>(base->data());
        size_t count = shape_.total_elements();

        // Apply delta encoding in-place
        for (size_t i = 0; i < count; ++i) {
            data[i] -= base_data[i];
        }
    }
}

// TensorBatch implementation

std::vector<uint8_t> TensorBatch::serialize() const {
    std::vector<uint8_t> buffer;

    // Header: count
    uint32_t count = static_cast<uint32_t>(tensors_.size());
    buffer.resize(sizeof(count));
    std::memcpy(buffer.data(), &count, sizeof(count));

    // Serialize each tensor
    for (const auto &tensor : tensors_) {
        auto tensor_data = tensor->serialize();
        uint32_t tensor_size = static_cast<uint32_t>(tensor_data.size());

        size_t offset = buffer.size();
        buffer.resize(offset + sizeof(tensor_size) + tensor_size);

        std::memcpy(buffer.data() + offset, &tensor_size, sizeof(tensor_size));
        std::memcpy(buffer.data() + offset + sizeof(tensor_size),
                    tensor_data.data(), tensor_size);
    }

    return buffer;
}

bool TensorBatch::deserialize(const std::vector<uint8_t> &buffer) {
    try {
        size_t offset = 0;

        // Read count
        uint32_t count;
        if (buffer.size() < sizeof(count))
            return false;
        std::memcpy(&count, buffer.data(), sizeof(count));
        offset += sizeof(count);

        tensors_.clear();
        tensors_.reserve(count);

        // Deserialize each tensor
        for (uint32_t i = 0; i < count; ++i) {
            if (offset + sizeof(uint32_t) > buffer.size())
                return false;

            uint32_t tensor_size;
            std::memcpy(&tensor_size, buffer.data() + offset,
                        sizeof(tensor_size));
            offset += sizeof(tensor_size);

            if (offset + tensor_size > buffer.size())
                return false;

            // Create tensor from serialized data
            auto tensor = std::make_unique<TensorMessage>(
                const_cast<uint8_t*>(buffer.data() + offset), tensor_size);
                
            // We already have the data in the tensor, no need to deserialize again

            tensors_.push_back(std::move(tensor));
            offset += tensor_size;
        }

        return true;

    } catch (const std::exception &) {
        return false;
    }
}

void TensorBatch::compress_batch() {
    if (tensors_.empty()) {
        return;
    }
    
    // Strategy 1: Fuse small tensors with same dtype
    std::unordered_map<DataType, std::vector<TensorMessage*>> tensors_by_type;
    size_t small_tensor_threshold = 1024; // 1KB
    
    for (auto& tensor : tensors_) {
        if (tensor->size_bytes() < small_tensor_threshold) {
            tensors_by_type[tensor->dtype()].push_back(tensor.get());
        }
    }
    
    // Strategy 2: Apply common quantization to float tensors
    for (auto& tensor : tensors_) {
        if (tensor->dtype() == DataType::Float32 && 
            tensor->size_bytes() > 10000) { // Only quantize larger tensors
            tensor->compress(CompressionMethod::Quantized);
        }
    }
    
    // Strategy 3: Delta encode consecutive tensors with same shape
    for (size_t i = 1; i < tensors_.size(); ++i) {
        auto& current = tensors_[i];
        auto& previous = tensors_[i-1];
        
        if (current->shape() == previous->shape() &&
            current->dtype() == previous->dtype()) {
            current->apply_delta_encoding(previous.get());
        }
    }
    
    // Strategy 4: Apply sparse compression to tensors with high sparsity
    for (auto& tensor : tensors_) {
        if (tensor->dtype() == DataType::Float32 || 
            tensor->dtype() == DataType::Float64) {
            // Estimate sparsity
            size_t zero_count = 0;
            size_t total_elements = tensor->shape().total_elements();
            
            if (tensor->dtype() == DataType::Float32) {
                const float* data = static_cast<const float*>(tensor->data());
                for (size_t i = 0; i < total_elements; ++i) {
                    if (data[i] == 0.0f) zero_count++;
                }
            } else {
                const double* data = static_cast<const double*>(tensor->data());
                for (size_t i = 0; i < total_elements; ++i) {
                    if (data[i] == 0.0) zero_count++;
                }
            }
            
            float sparsity = static_cast<float>(zero_count) / total_elements;
            if (sparsity > 0.9f) { // 90% sparse
                tensor->compress(CompressionMethod::Sparse);
            }
        }
    }
}

// GradientMessage implementation

void GradientMessage::compress_gradients(float sparsity) {
    if (dtype_ != DataType::Float32)
        return;

    size_t num_elements = shape_.total_elements();
    float *data = static_cast<float *>(data_.get());

    // Calculate threshold for top-k sparsification
    std::vector<float> abs_values(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        abs_values[i] = std::abs(data[i]);
    }

    size_t k = static_cast<size_t>((1.0f - sparsity) * num_elements);
    std::nth_element(abs_values.begin(), abs_values.begin() + k,
                     abs_values.end(), std::greater<float>());
    float threshold = abs_values[k];

    // Apply sparsification
    sparse_indices_.clear();
    sparse_indices_.reserve(k);

    for (size_t i = 0; i < num_elements; ++i) {
        if (std::abs(data[i]) >= threshold) {
            sparse_indices_.push_back(static_cast<uint32_t>(i));
        } else {
            data[i] = 0.0f;
        }
    }
}

void GradientMessage::merge(const GradientMessage &other) {
    if (shape_.total_elements() != other.shape_.total_elements() ||
        dtype_ != other.dtype_) {
        throw std::runtime_error("Cannot merge incompatible gradients");
    }

    if (dtype_ == DataType::Float32) {
        float *data = static_cast<float *>(data_.get());
        const float *other_data = static_cast<const float *>(other.data());
        size_t count = shape_.total_elements();

        // Element-wise addition
        simd::TensorOps<float>::add(data, other_data, data, count);
    }
}

// TensorChannel implementation

void TensorChannel::send_tensor(const TensorMessage &tensor, bool compress) {
    TensorMessage to_send = tensor; // Copy

    if (enable_delta_ && last_sent_) {
        to_send.apply_delta_encoding(last_sent_.get());
    }

    if (compress && enable_compression_) {
        // Choose compression based on tensor properties
        if (to_send.shape().total_elements() > 10000) {
            to_send.compress(CompressionMethod::Quantized);
        }
    }

    channel_->send(to_send);

    // Update last sent for delta encoding
    if (enable_delta_) {
        last_sent_ = std::make_unique<TensorMessage>(tensor);
    }
}

bool TensorChannel::receive_tensor(TensorMessage &tensor,
                                   std::chrono::milliseconds timeout) {
    return channel_->try_receive(tensor, timeout);
}

void TensorChannel::send_batch(const TensorBatch &batch) {
    TensorBatch optimized = batch; // Copy
    optimized.compress_batch();
    channel_->send(optimized);
}

void TensorChannel::pipeline_send(const std::vector<TensorMessage *> &tensors) {
    // Send tensors in pipeline fashion without waiting
    for (auto *tensor : tensors) {
        send_tensor(*tensor, true);
    }
}

// Helper functions implementation

namespace ops {

std::unique_ptr<TensorMessage>
fuse_tensors(const std::vector<TensorMessage *> &tensors) {
    if (tensors.empty())
        return nullptr;

    // Calculate total size
    size_t total_elements = 0;
    DataType common_dtype = tensors[0]->dtype();

    for (auto *tensor : tensors) {
        if (tensor->dtype() != common_dtype) {
            throw std::runtime_error(
                "Cannot fuse tensors with different dtypes");
        }
        total_elements += tensor->shape().total_elements();
    }

    // Create fused tensor
    auto fused = std::make_unique<TensorMessage>(
        "fused", TensorShape({total_elements}), common_dtype);

    // Copy data
    size_t offset = 0;
    size_t element_size = TensorShape::element_size(common_dtype);
    uint8_t *dst = static_cast<uint8_t *>(fused->data());

    for (auto *tensor : tensors) {
        size_t bytes = tensor->size_bytes();
        std::memcpy(dst + offset, tensor->data(), bytes);
        offset += bytes;
    }

    return fused;
}

std::vector<std::unique_ptr<TensorMessage>>
split_tensor(const TensorMessage &fused,
             const std::vector<TensorShape> &shapes) {
    std::vector<std::unique_ptr<TensorMessage>> result;
    result.reserve(shapes.size());

    size_t offset = 0;
    const uint8_t *src = static_cast<const uint8_t *>(fused.data());

    for (size_t i = 0; i < shapes.size(); ++i) {
        auto tensor = std::make_unique<TensorMessage>(
            "split_" + std::to_string(i), shapes[i], fused.dtype());

        size_t bytes = shapes[i].bytes(fused.dtype());
        std::memcpy(tensor->data(), src + offset, bytes);
        offset += bytes;

        result.push_back(std::move(tensor));
    }

    return result;
}

void optimize_for_transport(TensorMessage &tensor) {
    // Apply optimizations based on tensor properties
    size_t elements = tensor.shape().total_elements();

    // Small tensors: no optimization needed
    if (elements < 1000)
        return;

    // Medium tensors: consider quantization
    if (elements < 100000 && tensor.dtype() == DataType::Float32) {
        tensor.quantize(QuantizationParams{127.0f, 0, DataType::Int8, true});
    }

    // Large tensors: ensure optimal layout
    if (elements > 1000000 && tensor.layout() == Layout::NCHW) {
        // Some operations are faster with NHWC layout
        // tensor.transform_layout(Layout::NHWC);
    }
}

size_t optimal_batch_size(size_t tensor_bytes, size_t bandwidth_mbps) {
    // Calculate optimal batch size based on bandwidth and latency
    const size_t target_latency_us = 1000; // 1ms target
    const size_t bytes_per_us = (bandwidth_mbps * 1024 * 1024) / (8 * 1000000);
    const size_t bytes_per_batch = bytes_per_us * target_latency_us;

    return std::max(size_t(1), bytes_per_batch / tensor_bytes);
}

} // namespace ops

} // namespace tensor
} // namespace psyne