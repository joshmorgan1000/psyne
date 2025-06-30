/**
 * @file tensor_transport.hpp
 * @brief Optimized tensor transport for AI/ML workloads
 *
 * Provides specialized message types and optimizations for efficient
 * tensor transport between neural network layers, including layout
 * transformations, quantization, and compression.
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include "../core/message.hpp"
#include "../memory/custom_allocator.hpp"
#include "../simd/simd_ops.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace psyne {
namespace tensor {

/**
 * @brief Tensor data types
 */
enum class DataType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int32 = 3,
    Int16 = 4,
    Int8 = 5,
    UInt8 = 6,
    Bool = 7,
    Float64 = 8,
    Quantized = 9 // Custom quantization
};

/**
 * @brief Tensor layout formats
 */
enum class Layout : uint8_t {
    NCHW = 0,    // Batch, Channels, Height, Width (default for CNNs)
    NHWC = 1,    // Batch, Height, Width, Channels (optimized for some ops)
    NC = 2,      // Batch, Channels (for fully connected layers)
    CHW = 3,     // Channels, Height, Width (single image)
    HWC = 4,     // Height, Width, Channels (single image, channels last)
    Custom = 255 // User-defined layout
};

/**
 * @brief Quantization parameters
 */
struct QuantizationParams {
    float scale = 1.0f;
    int32_t zero_point = 0;
    DataType target_type = DataType::Int8;
    bool symmetric = true;
};

/**
 * @brief Tensor compression methods
 */
enum class CompressionMethod : uint8_t {
    None = 0,
    Sparse = 1,    // Sparse tensor compression
    Quantized = 2, // Quantization-based compression
    Delta = 3,     // Delta encoding for gradients
    Mixed = 4      // Mixed precision compression
};

/**
 * @brief Tensor shape information
 */
class TensorShape {
public:
    static constexpr size_t MAX_DIMS = 8;

    TensorShape() : dims_{}, ndims_(0) {}

    explicit TensorShape(std::initializer_list<size_t> dims)
        : dims_{}, ndims_(dims.size()) {
        std::copy(dims.begin(), dims.end(), dims_.begin());
    }
    
    explicit TensorShape(const std::vector<size_t> &dims)
        : dims_{}, ndims_(dims.size()) {
        std::copy(dims.begin(), dims.end(), dims_.begin());
    }

    size_t rank() const {
        return ndims_;
    }
    size_t dim(size_t i) const {
        return i < ndims_ ? dims_[i] : 1;
    }

    size_t total_elements() const {
        size_t total = 1;
        for (size_t i = 0; i < ndims_; ++i) {
            total *= dims_[i];
        }
        return total;
    }

    size_t bytes(DataType dtype) const {
        return total_elements() * element_size(dtype);
    }

    static size_t element_size(DataType dtype) {
        switch (dtype) {
        case DataType::Float64:
            return 8;
        case DataType::Float32:
        case DataType::Int32:
            return 4;
        case DataType::Float16:
        case DataType::BFloat16:
        case DataType::Int16:
            return 2;
        case DataType::Int8:
        case DataType::UInt8:
        case DataType::Bool:
            return 1;
        case DataType::Quantized:
            return 1; // Typically int8
        default:
            return 4;
        }
    }

    const std::array<size_t, MAX_DIMS> &dims() const {
        return dims_;
    }

private:
    std::array<size_t, MAX_DIMS> dims_;
    size_t ndims_;
};

/**
 * @brief Optimized tensor message for transport
 */
class TensorMessage : public Message<TensorMessage> {
public:
    /**
     * @brief Create tensor message for outgoing data
     */
    explicit TensorMessage(Channel &channel) : Message<TensorMessage>(channel) {}
    
    /**
     * @brief Create tensor message from incoming data
     */
    TensorMessage(void *data, size_t size) : Message<TensorMessage>(data, size) {}

    /**
     * @brief Create tensor message with data
     */
    TensorMessage(Channel &channel, const std::string &name, const TensorShape &shape,
                  DataType dtype, Layout layout = Layout::NCHW)
        : Message<TensorMessage>(channel), name_(name), shape_(shape), dtype_(dtype), layout_(layout) {
        allocate();
    }

    // Message interface
    static size_t calculate_size() {
        return 4096; // Reasonable default, will be adjusted during serialization
    }
    
    size_t type_hash() const {
        return std::hash<std::string>{}("TensorMessage");
    }

    std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t> &data);

    // Tensor-specific methods
    const std::string &name() const {
        return name_;
    }
    const TensorShape &shape() const {
        return shape_;
    }
    DataType dtype() const {
        return dtype_;
    }
    Layout layout() const {
        return layout_;
    }

    void *data() {
        return data_.get();
    }
    const void *data() const {
        return data_.get();
    }

    size_t size_bytes() const {
        return shape_.bytes(dtype_);
    }

    /**
     * @brief Transform tensor layout
     */
    void transform_layout(Layout new_layout);

    /**
     * @brief Quantize tensor data
     */
    void quantize(const QuantizationParams &params);

    /**
     * @brief Apply delta encoding (useful for gradients)
     */
    void apply_delta_encoding(const TensorMessage *base);

    /**
     * @brief Get compression ratio
     */
    float compression_ratio() const {
        if (compressed_data_.empty())
            return 1.0f;
        return static_cast<float>(size_bytes()) / compressed_data_.size();
    }

protected:
    void allocate();
    void compress(CompressionMethod method);
    void decompress();

    std::string name_;
    TensorShape shape_;
    DataType dtype_;
    Layout layout_;

    // Raw tensor data
    memory::UniqueAlloc data_;

    // Compressed representation
    std::vector<uint8_t> compressed_data_;
    CompressionMethod compression_ = CompressionMethod::None;

    // Quantization info
    QuantizationParams quant_params_;
};

/**
 * @brief Batch of tensors for efficient transport
 */
class TensorBatch : public Message<TensorBatch> {
public:
    /**
     * @brief Create tensor batch for outgoing data
     */
    explicit TensorBatch(Channel &channel) : Message<TensorBatch>(channel) {}
    
    /**
     * @brief Create tensor batch from incoming data
     */
    TensorBatch(void *data, size_t size) : Message<TensorBatch>(data, size) {}

    void add_tensor(std::unique_ptr<TensorMessage> tensor) {
        tensors_.push_back(std::move(tensor));
    }

    size_t count() const {
        return tensors_.size();
    }

    TensorMessage *tensor(size_t i) {
        return i < tensors_.size() ? tensors_[i].get() : nullptr;
    }

    const TensorMessage *tensor(size_t i) const {
        return i < tensors_.size() ? tensors_[i].get() : nullptr;
    }

    // Message interface
    static size_t calculate_size() {
        return 8192; // Reasonable default for batch messages
    }
    
    size_t type_hash() const {
        return std::hash<std::string>{}("TensorBatch");
    }

    std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t> &data);

    /**
     * @brief Apply batch compression
     */
    void compress_batch();

private:
    std::vector<std::unique_ptr<TensorMessage>> tensors_;
};

/**
 * @brief Gradient accumulation message for distributed training
 */
class GradientMessage : public TensorMessage {
public:
    /**
     * @brief Create gradient message for outgoing data
     */
    explicit GradientMessage(Channel &channel) : TensorMessage(channel) {}
    
    /**
     * @brief Create gradient message from incoming data
     */
    GradientMessage(void *data, size_t size) : TensorMessage(data, size) {}

    GradientMessage(Channel &channel, const std::string &name, const TensorShape &shape,
                    DataType dtype, int32_t iteration)
        : TensorMessage(channel, name, shape, dtype), iteration_(iteration) {}

    static size_t calculate_size() {
        return TensorMessage::calculate_size(); // Same as base tensor message
    }
    
    size_t type_hash() const {
        return std::hash<std::string>{}("GradientMessage");
    }

    int32_t iteration() const {
        return iteration_;
    }

    /**
     * @brief Apply gradient compression (top-k, random-k, etc.)
     */
    void compress_gradients(float sparsity = 0.99f);

    /**
     * @brief Merge with another gradient
     */
    void merge(const GradientMessage &other);

private:
    int32_t iteration_ = 0;
    std::vector<uint32_t> sparse_indices_;
};

/**
 * @brief Optimized channel wrapper for tensor transport
 */
class TensorChannel {
public:
    explicit TensorChannel(std::unique_ptr<Channel> channel)
        : channel_(std::move(channel)) {}

    /**
     * @brief Send tensor with optimizations
     */
    void send_tensor(const TensorMessage &tensor, bool compress = true);

    /**
     * @brief Receive tensor
     */
    bool receive_tensor(
        TensorMessage &tensor,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(-1));

    /**
     * @brief Send batch of tensors
     */
    void send_batch(const TensorBatch &batch);

    /**
     * @brief Pipeline multiple tensor sends
     */
    void pipeline_send(const std::vector<TensorMessage *> &tensors);

    /**
     * @brief Get underlying channel
     */
    Channel *channel() {
        return channel_.get();
    }

private:
    std::unique_ptr<Channel> channel_;

    // Optimization state
    std::unique_ptr<TensorMessage> last_sent_; // For delta encoding
    bool enable_delta_ = true;
    bool enable_compression_ = true;
};

/**
 * @brief Helper functions for tensor operations
 */
namespace ops {

/**
 * @brief Fuse multiple small tensors into one for transport
 */
std::unique_ptr<TensorMessage>
fuse_tensors(const std::vector<TensorMessage *> &tensors);

/**
 * @brief Split fused tensor back to components
 */
std::vector<std::unique_ptr<TensorMessage>>
split_tensor(const TensorMessage &fused,
             const std::vector<TensorShape> &shapes);

/**
 * @brief Optimize tensor for transport (in-place)
 */
void optimize_for_transport(TensorMessage &tensor);

/**
 * @brief Calculate optimal batch size for given bandwidth
 */
size_t optimal_batch_size(size_t tensor_bytes, size_t bandwidth_mbps);

} // namespace ops

} // namespace tensor
} // namespace psyne