#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace psyne {

/**
 * @brief Message types for high-performance communication
 * 
 * These are just examples - users can create any message type they want
 * and use it with Channel<MessageType, Substrate, Pattern>
 */
enum class MessageType : uint16_t {
    FLOAT32_VECTOR = 1,
    FLOAT32_MATRIX = 2,
    FLOAT32_TENSOR = 3,
    FLOAT16_VECTOR = 4,
    FLOAT16_MATRIX = 5,
    FLOAT16_TENSOR = 6,
    INT8_VECTOR = 7,
    INT8_MATRIX = 8,
    INT8_TENSOR = 9,
    INT16_VECTOR = 10,
    INT16_MATRIX = 11,
    INT16_TENSOR = 12,
    INT4_VECTOR = 13,
    INT4_MATRIX = 14,
    INT4_TENSOR = 15,
    INT3_VECTOR = 16,
    INT3_MATRIX = 17,
    INT3_TENSOR = 18,
    INT2_VECTOR = 19,
    INT2_MATRIX = 20,
    INT2_TENSOR = 21,
    DOUBLE_VECTOR = 16,
    DOUBLE_MATRIX = 17,
    DOUBLE_TENSOR = 18,
    GRADIENT = 100,
    ACTIVATION = 101,
    WEIGHT_UPDATE = 102
};

/**
 * @brief Fixed-size float32 vector message
 *
 * Optimized for layer outputs and embeddings. Can be directly
 * cast to Eigen::VectorXf for computation.
 *
 * @tparam N Dimension of the vector
 */
template <size_t N>
struct alignas(64) Float32VectorMessage {
    static constexpr size_t dimension = N;
    static constexpr MessageType type = MessageType::FLOAT32_VECTOR;

    // Data is first for direct Eigen mapping
    float data[N];

    // Metadata after data
    uint32_t batch_idx = 0;
    uint32_t layer_id = 0;

    /**
     * @brief Get as Eigen vector (zero-copy)
     */
    Eigen::Map<Eigen::VectorXf> as_eigen() {
        return Eigen::Map<Eigen::VectorXf>(data, N);
    }

    Eigen::Map<const Eigen::VectorXf> as_eigen() const {
        return Eigen::Map<const Eigen::VectorXf>(data, N);
    }

    /**
     * @brief Fill with zeros
     */
    void zero() {
        std::fill_n(data, N, 0.0f);
    }

    /**
     * @brief Get size in bytes
     */
    static constexpr size_t size_bytes() {
        return sizeof(Float32VectorMessage<N>);
    }
};

/**
 * @brief Fixed-size float32 matrix message
 *
 * Optimized for weight matrices and activations. Row-major storage
 * for compatibility with most ML frameworks.
 *
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 */
template <size_t Rows, size_t Cols>
struct alignas(64) Float32MatrixMessage {
    static constexpr size_t rows = Rows;
    static constexpr size_t cols = Cols;
    static constexpr size_t elements = Rows * Cols;
    static constexpr MessageType type = MessageType::FLOAT32_MATRIX;

    // Row-major storage
    float data[Rows * Cols];

    // Metadata
    uint32_t batch_idx = 0;
    uint32_t layer_id = 0;

    /**
     * @brief Get as Eigen matrix (zero-copy)
     */
    Eigen::Map<Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor>> as_eigen() {
        return Eigen::Map<Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor>>(
            data);
    }

    Eigen::Map<const Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor>>
    as_eigen() const {
        return Eigen::Map<
            const Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor>>(data);
    }

    /**
     * @brief Access element
     */
    float &operator()(size_t row, size_t col) {
        return data[row * Cols + col];
    }

    const float &operator()(size_t row, size_t col) const {
        return data[row * Cols + col];
    }

    /**
     * @brief Fill with zeros
     */
    void zero() {
        std::fill_n(data, elements, 0.0f);
    }

    /**
     * @brief Get size in bytes
     */
    static constexpr size_t size_bytes() {
        return sizeof(Float32MatrixMessage<Rows, Cols>);
    }
};

/**
 * @brief Dynamic tensor message with inline data
 *
 * For tensors with dynamic shapes. Data follows immediately after
 * the header in memory.
 */
struct alignas(64) DynamicTensorMessage {
    static constexpr MessageType type = MessageType::FLOAT32_TENSOR;

    // Shape information
    uint32_t ndims;
    uint32_t shape[4]; // Up to 4D tensors
    uint32_t strides[4];

    // Metadata
    uint32_t batch_idx = 0;
    uint32_t layer_id = 0;
    uint16_t dtype = 0;  // 0=float32, 1=float16, 2=int8
    uint16_t layout = 0; // 0=NCHW, 1=NHWC

    // Total number of elements
    size_t num_elements() const {
        size_t n = 1;
        for (uint32_t i = 0; i < ndims; ++i) {
            n *= shape[i];
        }
        return n;
    }

    // Get data pointer (data follows struct in memory)
    float *data() {
        return reinterpret_cast<float *>(this + 1);
    }

    const float *data() const {
        return reinterpret_cast<const float *>(this + 1);
    }

    // Total size including data
    size_t total_size() const {
        return sizeof(DynamicTensorMessage) + num_elements() * sizeof(float);
    }
};

/**
 * @brief Gradient message for backpropagation
 *
 * Specialized message for gradients with additional metadata
 * for optimization algorithms.
 */
template <size_t N>
struct alignas(64) GradientMessage {
    static constexpr MessageType type = MessageType::GRADIENT;

    float gradients[N];
    float momentum[N]; // For optimizers that need momentum

    uint32_t layer_id;
    uint32_t parameter_id;
    uint32_t iteration;
    float learning_rate;

    /**
     * @brief Apply gradient update with momentum
     */
    void apply_momentum(float beta = 0.9f) {
        for (size_t i = 0; i < N; ++i) {
            momentum[i] = beta * momentum[i] + (1 - beta) * gradients[i];
        }
    }
};

// Common sizes for neural networks
using Embedding64Message = Float32VectorMessage<64>;
using Embedding128Message = Float32VectorMessage<128>;
using Embedding256Message = Float32VectorMessage<256>;
using Embedding512Message = Float32VectorMessage<512>;
using Embedding768Message = Float32VectorMessage<768>;   // BERT-base
using Embedding1024Message = Float32VectorMessage<1024>; // BERT-large

// Batch messages for mini-batch processing
template <typename MessageType, size_t BatchSize>
struct BatchMessage {
    MessageType messages[BatchSize];
    uint32_t actual_batch_size = BatchSize;

    MessageType &operator[](size_t idx) {
        return messages[idx];
    }
    const MessageType &operator[](size_t idx) const {
        return messages[idx];
    }
};

} // namespace psyne