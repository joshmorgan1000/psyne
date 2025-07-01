#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace psyne::message {

/**
 * @brief Message types for numeric data
 *
 * These are examples - users can create any message type they want
 * and use it with Channel<MessageType, Substrate, Pattern>
 */
enum class NumericType : uint16_t {
    FLOAT32_VECTOR = 1,
    FLOAT32_MATRIX = 2,
    FLOAT64_VECTOR = 3,
    FLOAT64_MATRIX = 4,
    INT32_VECTOR = 5,
    INT32_MATRIX = 6,
    // Add more as needed
};

/**
 * @brief Fixed-size float32 vector message
 *
 * Optimized for numerical computations. Can be directly
 * cast to Eigen::VectorXf for computation.
 *
 * @tparam N Dimension of the vector
 */
template <size_t N>
struct alignas(64) Float32Vector {
    static constexpr size_t dimension = N;
    static constexpr NumericType type = NumericType::FLOAT32_VECTOR;

    // Data is first for direct Eigen mapping
    float data[N];

    // Metadata after data
    uint32_t batch_idx = 0;
    uint32_t sequence_id = 0;

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
        return sizeof(Float32Vector<N>);
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
struct alignas(64) Float32Matrix {
    static constexpr size_t rows = Rows;
    static constexpr size_t cols = Cols;
    static constexpr size_t elements = Rows * Cols;
    static constexpr NumericType type = NumericType::FLOAT32_MATRIX;

    // Row-major storage
    float data[Rows * Cols];

    // Metadata
    uint32_t batch_idx = 0;
    uint32_t sequence_id = 0;

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
        return sizeof(Float32Matrix<Rows, Cols>);
    }
};

// Common numeric message aliases
using Vector64 = Float32Vector<64>;
using Vector128 = Float32Vector<128>;
using Vector256 = Float32Vector<256>;
using Vector512 = Float32Vector<512>;
using Vector1024 = Float32Vector<1024>;

using Matrix4x4 = Float32Matrix<4, 4>;
using Matrix8x8 = Float32Matrix<8, 8>;
using Matrix16x16 = Float32Matrix<16, 16>;

} // namespace psyne::message