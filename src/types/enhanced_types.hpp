#pragma once

/**
 * @file enhanced_types.hpp
 * @brief Enhanced message types for Psyne - matrices, vectors, tensors, and
 * specialized data types
 *
 * This header provides advanced message types optimized for different use
 * cases:
 * - Fixed-size matrices and vectors for graphics/robotics
 * - Quantized types for efficient neural network inference
 * - Complex number vectors for signal processing
 * - ML tensors with layout support
 * - Sparse matrices for scientific computing
 *
 * All types maintain zero-copy semantics and are Eigen-compatible.
 */

#include "../core/message.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <vector>

#ifdef PSYNE_ENABLE_EIGEN
#include <Eigen/Dense>
#include <Eigen/Sparse>
#endif

namespace psyne {
namespace types {

// ============================================================================
// Fixed-size Matrix Types
// ============================================================================

/**
 * @class Matrix4x4f
 * @brief 4x4 single-precision float matrix for 3D transformations
 */
class Matrix4x4f : public Message<Matrix4x4f> {
public:
    static constexpr uint32_t message_type = 101;
    static constexpr size_t ROWS = 4;
    static constexpr size_t COLS = 4;
    static constexpr size_t SIZE = ROWS * COLS;

    using Message<Matrix4x4f>::Message;

    static size_t calculate_size() {
        return SIZE * sizeof(float);
    }

    void initialize() {
        // Initialize as identity matrix
        std::fill(data(), data() + SIZE, 0.0f);
        for (size_t i = 0; i < 4; ++i) {
            (*this)(i, i) = 1.0f;
        }
    }

    // Element access
    float &operator()(size_t row, size_t col) {
        assert(row < ROWS && col < COLS);
        return data()[row * COLS + col];
    }

    const float &operator()(size_t row, size_t col) const {
        assert(row < ROWS && col < COLS);
        return data()[row * COLS + col];
    }

    // Matrix operations
    float determinant() const {
        const float *m = data();
        return m[0] * (m[5] * (m[10] * m[15] - m[11] * m[14]) -
                       m[6] * (m[9] * m[15] - m[11] * m[13]) +
                       m[7] * (m[9] * m[14] - m[10] * m[13])) -
               m[1] * (m[4] * (m[10] * m[15] - m[11] * m[14]) -
                       m[6] * (m[8] * m[15] - m[11] * m[12]) +
                       m[7] * (m[8] * m[14] - m[10] * m[12])) +
               m[2] * (m[4] * (m[9] * m[15] - m[11] * m[13]) -
                       m[5] * (m[8] * m[15] - m[11] * m[12]) +
                       m[7] * (m[8] * m[13] - m[9] * m[12])) -
               m[3] * (m[4] * (m[9] * m[14] - m[10] * m[13]) -
                       m[5] * (m[8] * m[14] - m[10] * m[12]) +
                       m[6] * (m[8] * m[13] - m[9] * m[12]));
    }

    float trace() const {
        return (*this)(0, 0) + (*this)(1, 1) + (*this)(2, 2) + (*this)(3, 3);
    }

    // Data access
    float *data() {
        return reinterpret_cast<float *>(Message<Matrix4x4f>::data());
    }
    const float *data() const {
        return reinterpret_cast<const float *>(Message<Matrix4x4f>::data());
    }

#ifdef PSYNE_ENABLE_EIGEN
    auto as_eigen() {
        return Eigen::Map<Eigen::Matrix4f>(data());
    }

    auto as_eigen() const {
        return Eigen::Map<const Eigen::Matrix4f>(data());
    }
#endif
};

/**
 * @class Matrix3x3f
 * @brief 3x3 single-precision float matrix for 2D transformations and rotations
 */
class Matrix3x3f : public Message<Matrix3x3f> {
public:
    static constexpr uint32_t message_type = 102;
    static constexpr size_t ROWS = 3;
    static constexpr size_t COLS = 3;
    static constexpr size_t SIZE = ROWS * COLS;

    using Message<Matrix3x3f>::Message;

    static size_t calculate_size() {
        return SIZE * sizeof(float);
    }

    void initialize() {
        std::fill(data(), data() + SIZE, 0.0f);
        for (size_t i = 0; i < 3; ++i) {
            (*this)(i, i) = 1.0f;
        }
    }

    float &operator()(size_t row, size_t col) {
        assert(row < ROWS && col < COLS);
        return data()[row * COLS + col];
    }

    const float &operator()(size_t row, size_t col) const {
        assert(row < ROWS && col < COLS);
        return data()[row * COLS + col];
    }

    float determinant() const {
        const float *m = data();
        return m[0] * (m[4] * m[8] - m[5] * m[7]) -
               m[1] * (m[3] * m[8] - m[5] * m[6]) +
               m[2] * (m[3] * m[7] - m[4] * m[6]);
    }

    float trace() const {
        return (*this)(0, 0) + (*this)(1, 1) + (*this)(2, 2);
    }

    float *data() {
        return reinterpret_cast<float *>(Message<Matrix3x3f>::data());
    }
    const float *data() const {
        return reinterpret_cast<const float *>(Message<Matrix3x3f>::data());
    }

#ifdef PSYNE_ENABLE_EIGEN
    auto as_eigen() {
        return Eigen::Map<Eigen::Matrix3f>(data());
    }

    auto as_eigen() const {
        return Eigen::Map<const Eigen::Matrix3f>(data());
    }
#endif
};

// ============================================================================
// Fixed-size Vector Types
// ============================================================================

/**
 * @class Vector3f
 * @brief 3D single-precision float vector with named accessors
 */
class Vector3f : public Message<Vector3f> {
public:
    static constexpr uint32_t message_type = 103;
    static constexpr size_t SIZE = 3;

    using Message<Vector3f>::Message;

    static size_t calculate_size() {
        return SIZE * sizeof(float);
    }

    void initialize() {
        std::fill(data(), data() + SIZE, 0.0f);
    }

    // Named accessors
    float &x() {
        return data()[0];
    }
    float &y() {
        return data()[1];
    }
    float &z() {
        return data()[2];
    }

    const float &x() const {
        return data()[0];
    }
    const float &y() const {
        return data()[1];
    }
    const float &z() const {
        return data()[2];
    }

    // Array access
    float &operator[](size_t index) {
        assert(index < SIZE);
        return data()[index];
    }

    const float &operator[](size_t index) const {
        assert(index < SIZE);
        return data()[index];
    }

    // Vector operations
    float length() const {
        return std::sqrt(x() * x() + y() * y() + z() * z());
    }

    float length_squared() const {
        return x() * x() + y() * y() + z() * z();
    }

    void normalize() {
        float len = length();
        if (len > 1e-6f) {
            x() /= len;
            y() /= len;
            z() /= len;
        }
    }

    float dot(const Vector3f &other) const {
        return x() * other.x() + y() * other.y() + z() * other.z();
    }

    // In-place scalar operations
    Vector3f &operator*=(float scalar) {
        x() *= scalar;
        y() *= scalar;
        z() *= scalar;
        return *this;
    }

    Vector3f &operator+=(const Vector3f &other) {
        x() += other.x();
        y() += other.y();
        z() += other.z();
        return *this;
    }

    float *data() {
        return reinterpret_cast<float *>(Message<Vector3f>::data());
    }
    const float *data() const {
        return reinterpret_cast<const float *>(Message<Vector3f>::data());
    }

#ifdef PSYNE_ENABLE_EIGEN
    auto as_eigen() {
        return Eigen::Map<Eigen::Vector3f>(data());
    }

    auto as_eigen() const {
        return Eigen::Map<const Eigen::Vector3f>(data());
    }
#endif
};

/**
 * @class Vector4f
 * @brief 4D single-precision float vector (homogeneous coordinates)
 */
class Vector4f : public Message<Vector4f> {
public:
    static constexpr uint32_t message_type = 104;
    static constexpr size_t SIZE = 4;

    using Message<Vector4f>::Message;

    static size_t calculate_size() {
        return SIZE * sizeof(float);
    }

    void initialize() {
        std::fill(data(), data() + SIZE, 0.0f);
        w() = 1.0f; // Default to homogeneous coordinate
    }

    float &x() {
        return data()[0];
    }
    float &y() {
        return data()[1];
    }
    float &z() {
        return data()[2];
    }
    float &w() {
        return data()[3];
    }

    const float &x() const {
        return data()[0];
    }
    const float &y() const {
        return data()[1];
    }
    const float &z() const {
        return data()[2];
    }
    const float &w() const {
        return data()[3];
    }

    float &operator[](size_t index) {
        assert(index < SIZE);
        return data()[index];
    }

    const float &operator[](size_t index) const {
        assert(index < SIZE);
        return data()[index];
    }

    float length() const {
        return std::sqrt(x() * x() + y() * y() + z() * z() + w() * w());
    }

    float *data() {
        return reinterpret_cast<float *>(Message<Vector4f>::data());
    }
    const float *data() const {
        return reinterpret_cast<const float *>(Message<Vector4f>::data());
    }

#ifdef PSYNE_ENABLE_EIGEN
    auto as_eigen() {
        return Eigen::Map<Eigen::Vector4f>(data());
    }

    auto as_eigen() const {
        return Eigen::Map<const Eigen::Vector4f>(data());
    }
#endif
};

// ============================================================================
// Quantized Types for ML Inference
// ============================================================================

/**
 * @class Int8Vector
 * @brief Quantized 8-bit signed integer vector for neural network inference
 */
class Int8Vector : public Message<Int8Vector> {
public:
    static constexpr uint32_t message_type = 105;

    using Message<Int8Vector>::Message;

    static size_t calculate_size() {
        return 1024; // Default size, will be resized as needed
    }

    void initialize() {
        header().size = 0;
        header().scale = 1.0f;
        header().zero_point = 0;
    }

    // Quantization parameters
    struct Header {
        uint32_t size;
        float scale;
        int32_t zero_point;
    };

    Header &header() {
        return *reinterpret_cast<Header *>(Message<Int8Vector>::data());
    }
    const Header &header() const {
        return *reinterpret_cast<const Header *>(Message<Int8Vector>::data());
    }

    int8_t *data() {
        return reinterpret_cast<int8_t *>(Message<Int8Vector>::data()) +
               sizeof(Header);
    }
    const int8_t *data() const {
        return reinterpret_cast<const int8_t *>(Message<Int8Vector>::data()) +
               sizeof(Header);
    }

    size_t size() const {
        return header().size;
    }

    void resize(size_t new_size) {
        // Note: In real implementation, this would need to handle memory
        // allocation
        header().size = static_cast<uint32_t>(new_size);
    }

    void set_quantization_params(float scale, int32_t zero_point) {
        header().scale = scale;
        header().zero_point = zero_point;
    }

    // Quantize from float array
    void quantize_from(const float *values, size_t count) {
        resize(count);
        float scale = header().scale;
        int32_t zero_point = header().zero_point;

        for (size_t i = 0; i < count; ++i) {
            int32_t quantized = static_cast<int32_t>(
                std::round(values[i] / scale) + zero_point);
            data()[i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
        }
    }

    // Dequantize to float array
    void dequantize_to(float *values, size_t count) const {
        size_t copy_count = std::min(count, size());
        float scale = header().scale;
        int32_t zero_point = header().zero_point;

        for (size_t i = 0; i < copy_count; ++i) {
            values[i] = scale * (static_cast<int32_t>(data()[i]) - zero_point);
        }
    }

    int8_t &operator[](size_t index) {
        assert(index < size());
        return data()[index];
    }

    const int8_t &operator[](size_t index) const {
        assert(index < size());
        return data()[index];
    }
};

/**
 * @class UInt8Vector
 * @brief Quantized 8-bit unsigned integer vector for neural network inference
 */
class UInt8Vector : public Message<UInt8Vector> {
public:
    static constexpr uint32_t message_type = 106;

    using Message<UInt8Vector>::Message;

    static size_t calculate_size() {
        return 1024;
    }

    void initialize() {
        header().size = 0;
        header().scale = 1.0f;
        header().zero_point = 0;
    }

    struct Header {
        uint32_t size;
        float scale;
        uint32_t zero_point;
    };

    Header &header() {
        return *reinterpret_cast<Header *>(Message<UInt8Vector>::data());
    }
    const Header &header() const {
        return *reinterpret_cast<const Header *>(Message<UInt8Vector>::data());
    }

    uint8_t *data() {
        return reinterpret_cast<uint8_t *>(Message<UInt8Vector>::data()) +
               sizeof(Header);
    }
    const uint8_t *data() const {
        return reinterpret_cast<const uint8_t *>(Message<UInt8Vector>::data()) +
               sizeof(Header);
    }

    size_t size() const {
        return header().size;
    }

    void resize(size_t new_size) {
        header().size = static_cast<uint32_t>(new_size);
    }

    void set_quantization_params(float scale, uint32_t zero_point) {
        header().scale = scale;
        header().zero_point = zero_point;
    }

    void quantize_from(const float *values, size_t count) {
        resize(count);
        float scale = header().scale;
        uint32_t zero_point = header().zero_point;

        for (size_t i = 0; i < count; ++i) {
            int32_t quantized = static_cast<int32_t>(
                std::round(values[i] / scale) + zero_point);
            data()[i] = static_cast<uint8_t>(std::clamp(quantized, 0, 255));
        }
    }

    void dequantize_to(float *values, size_t count) const {
        size_t copy_count = std::min(count, size());
        float scale = header().scale;
        uint32_t zero_point = header().zero_point;

        for (size_t i = 0; i < copy_count; ++i) {
            values[i] = scale * (static_cast<int32_t>(data()[i]) -
                                 static_cast<int32_t>(zero_point));
        }
    }

    uint8_t &operator[](size_t index) {
        assert(index < size());
        return data()[index];
    }

    const uint8_t &operator[](size_t index) const {
        assert(index < size());
        return data()[index];
    }
};

} // namespace types
} // namespace psyne