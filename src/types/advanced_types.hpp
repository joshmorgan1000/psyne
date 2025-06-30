#pragma once

/**
 * @file advanced_types.hpp
 * @brief Advanced message types for Psyne - complex numbers, ML tensors, and
 * sparse matrices
 *
 * This header provides specialized message types for advanced use cases:
 * - Complex number vectors for signal processing and DSP
 * - ML tensors with layout support for neural networks
 * - Sparse matrices for scientific computing
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
// Complex Number Types
// ============================================================================

/**
 * @class ComplexVectorF
 * @brief Vector of single-precision complex numbers for signal processing
 */
class ComplexVectorF : public Message<ComplexVectorF> {
public:
    static constexpr uint32_t message_type = 107;
    using ComplexType = std::complex<float>;

    using Message<ComplexVectorF>::Message;

    static size_t calculate_size() {
        return 1024 * sizeof(ComplexType); // Default size
    }

    void initialize() {
        header().size = 0;
    }

    struct Header {
        uint32_t size;
        uint32_t padding; // Align to 8 bytes
    };

    Header &header() {
        return *reinterpret_cast<Header *>(Message<ComplexVectorF>::data());
    }
    const Header &header() const {
        return *reinterpret_cast<const Header *>(
            Message<ComplexVectorF>::data());
    }

    ComplexType *data() {
        return reinterpret_cast<ComplexType *>(
            reinterpret_cast<uint8_t *>(Message<ComplexVectorF>::data()) +
            sizeof(Header));
    }

    const ComplexType *data() const {
        return reinterpret_cast<const ComplexType *>(
            reinterpret_cast<const uint8_t *>(Message<ComplexVectorF>::data()) +
            sizeof(Header));
    }

    size_t size() const {
        return header().size;
    }

    void resize(size_t new_size) {
        header().size = static_cast<uint32_t>(new_size);
    }

    ComplexType &operator[](size_t index) {
        assert(index < size());
        return data()[index];
    }

    const ComplexType &operator[](size_t index) const {
        assert(index < size());
        return data()[index];
    }

    // Signal processing operations
    float power() const {
        float total = 0.0f;
        for (size_t i = 0; i < size(); ++i) {
            float real = data()[i].real();
            float imag = data()[i].imag();
            total += real * real + imag * imag;
        }
        return total;
    }

    void conjugate() {
        for (size_t i = 0; i < size(); ++i) {
            data()[i] = std::conj(data()[i]);
        }
    }

    void multiply_pointwise(const ComplexVectorF &other) {
        size_t min_size = std::min(size(), other.size());
        for (size_t i = 0; i < min_size; ++i) {
            data()[i] *= other.data()[i];
        }
    }

    // Copy from std::vector<std::complex<float>>
    void copy_from(const std::vector<ComplexType> &source) {
        resize(source.size());
        std::copy(source.begin(), source.end(), data());
    }

    // Copy to std::vector<std::complex<float>>
    std::vector<ComplexType> to_vector() const {
        return std::vector<ComplexType>(data(), data() + size());
    }
};

// ============================================================================
// ML Tensor Type
// ============================================================================

/**
 * @class MLTensorF
 * @brief Multi-dimensional tensor for machine learning with layout support
 */
class MLTensorF : public Message<MLTensorF> {
public:
    static constexpr uint32_t message_type = 108;
    static constexpr size_t MAX_DIMS = 8; // Support up to 8D tensors

    enum class Layout {
        NCHW,  // Batch, Channels, Height, Width (typical for computer vision)
        NHWC,  // Batch, Height, Width, Channels (TensorFlow default)
        CHW,   // Channels, Height, Width (single image)
        HWC,   // Height, Width, Channels (single image)
        Custom // User-defined layout
    };

    enum class Activation { None, ReLU, Sigmoid, Tanh, Softmax };

    using Message<MLTensorF>::Message;

    static size_t calculate_size() {
        return 1024 * 1024; // 1MB default
    }

    void initialize() {
        header().num_dims = 0;
        header().layout = Layout::Custom;
        header().activation = Activation::None;
        std::fill(header().shape, header().shape + MAX_DIMS, 0);
        std::fill(header().strides, header().strides + MAX_DIMS, 0);
        header().total_elements = 0;
    }

    struct Header {
        uint32_t num_dims;
        Layout layout;
        Activation activation;
        uint32_t shape[MAX_DIMS];
        uint32_t strides[MAX_DIMS];
        uint32_t total_elements;
        uint32_t padding[2]; // Align to 8 bytes
    };

    Header &header() {
        return *reinterpret_cast<Header *>(Message<MLTensorF>::data());
    }
    const Header &header() const {
        return *reinterpret_cast<const Header *>(Message<MLTensorF>::data());
    }

    float *data() {
        return reinterpret_cast<float *>(
            reinterpret_cast<uint8_t *>(Message<MLTensorF>::data()) +
            sizeof(Header));
    }

    const float *data() const {
        return reinterpret_cast<const float *>(
            reinterpret_cast<const uint8_t *>(Message<MLTensorF>::data()) +
            sizeof(Header));
    }

    // Shape management
    void set_shape(const std::vector<uint32_t> &new_shape,
                   Layout layout = Layout::Custom) {
        assert(new_shape.size() <= MAX_DIMS);

        header().num_dims = static_cast<uint32_t>(new_shape.size());
        header().layout = layout;

        // Copy shape
        std::copy(new_shape.begin(), new_shape.end(), header().shape);
        std::fill(header().shape + new_shape.size(), header().shape + MAX_DIMS,
                  0);

        // Calculate strides (row-major order)
        uint32_t stride = 1;
        for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
            header().strides[i] = stride;
            stride *= new_shape[i];
        }

        header().total_elements = stride;
    }

    std::vector<uint32_t> shape() const {
        return std::vector<uint32_t>(header().shape,
                                     header().shape + header().num_dims);
    }

    uint32_t total_elements() const {
        return header().total_elements;
    }
    Layout layout() const {
        return header().layout;
    }

    // Element access (supports up to 4D for convenience)
    float &operator()(uint32_t i0) {
        assert(header().num_dims >= 1);
        return data()[i0 * header().strides[0]];
    }

    float &operator()(uint32_t i0, uint32_t i1) {
        assert(header().num_dims >= 2);
        return data()[i0 * header().strides[0] + i1 * header().strides[1]];
    }

    float &operator()(uint32_t i0, uint32_t i1, uint32_t i2) {
        assert(header().num_dims >= 3);
        return data()[i0 * header().strides[0] + i1 * header().strides[1] +
                      i2 * header().strides[2]];
    }

    float &operator()(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
        assert(header().num_dims >= 4);
        return data()[i0 * header().strides[0] + i1 * header().strides[1] +
                      i2 * header().strides[2] + i3 * header().strides[3]];
    }

    // Const versions
    const float &operator()(uint32_t i0) const {
        assert(header().num_dims >= 1);
        return data()[i0 * header().strides[0]];
    }

    const float &operator()(uint32_t i0, uint32_t i1) const {
        assert(header().num_dims >= 2);
        return data()[i0 * header().strides[0] + i1 * header().strides[1]];
    }

    const float &operator()(uint32_t i0, uint32_t i1, uint32_t i2) const {
        assert(header().num_dims >= 3);
        return data()[i0 * header().strides[0] + i1 * header().strides[1] +
                      i2 * header().strides[2]];
    }

    const float &operator()(uint32_t i0, uint32_t i1, uint32_t i2,
                            uint32_t i3) const {
        assert(header().num_dims >= 4);
        return data()[i0 * header().strides[0] + i1 * header().strides[1] +
                      i2 * header().strides[2] + i3 * header().strides[3]];
    }

    // Activation functions (in-place)
    void apply_relu() {
        for (uint32_t i = 0; i < total_elements(); ++i) {
            data()[i] = std::max(0.0f, data()[i]);
        }
        header().activation = Activation::ReLU;
    }

    void apply_sigmoid() {
        for (uint32_t i = 0; i < total_elements(); ++i) {
            data()[i] = 1.0f / (1.0f + std::exp(-data()[i]));
        }
        header().activation = Activation::Sigmoid;
    }

    void apply_tanh() {
        for (uint32_t i = 0; i < total_elements(); ++i) {
            data()[i] = std::tanh(data()[i]);
        }
        header().activation = Activation::Tanh;
    }

    // Softmax along last dimension
    void apply_softmax() {
        if (header().num_dims == 0)
            return;

        uint32_t last_dim = header().shape[header().num_dims - 1];
        uint32_t batch_size = total_elements() / last_dim;

        for (uint32_t batch = 0; batch < batch_size; ++batch) {
            float *batch_data = data() + batch * last_dim;

            // Find max for numerical stability
            float max_val =
                *std::max_element(batch_data, batch_data + last_dim);

            // Compute exp and sum
            float sum = 0.0f;
            for (uint32_t i = 0; i < last_dim; ++i) {
                batch_data[i] = std::exp(batch_data[i] - max_val);
                sum += batch_data[i];
            }

            // Normalize
            for (uint32_t i = 0; i < last_dim; ++i) {
                batch_data[i] /= sum;
            }
        }

        header().activation = Activation::Softmax;
    }
};

// ============================================================================
// Sparse Matrix Type
// ============================================================================

/**
 * @class SparseMatrixF
 * @brief Sparse matrix in Compressed Sparse Row (CSR) format
 */
class SparseMatrixF : public Message<SparseMatrixF> {
public:
    static constexpr uint32_t message_type = 109;

    using Message<SparseMatrixF>::Message;

    static size_t calculate_size() {
        return 1024 * 1024; // 1MB default
    }

    void initialize() {
        header().rows = 0;
        header().cols = 0;
        header().nnz = 0;
    }

    struct Header {
        uint32_t rows;
        uint32_t cols;
        uint32_t nnz;     // Number of non-zero elements
        uint32_t padding; // Align to 8 bytes
    };

    Header &header() {
        return *reinterpret_cast<Header *>(Message<SparseMatrixF>::data());
    }
    const Header &header() const {
        return *reinterpret_cast<const Header *>(
            Message<SparseMatrixF>::data());
    }

    // CSR format: values, column_indices, row_pointers
    float *values() {
        return reinterpret_cast<float *>(
            reinterpret_cast<uint8_t *>(Message<SparseMatrixF>::data()) +
            sizeof(Header));
    }

    const float *values() const {
        return reinterpret_cast<const float *>(
            reinterpret_cast<const uint8_t *>(Message<SparseMatrixF>::data()) +
            sizeof(Header));
    }

    uint32_t *column_indices() {
        return reinterpret_cast<uint32_t *>(values() + header().nnz);
    }

    const uint32_t *column_indices() const {
        return reinterpret_cast<const uint32_t *>(values() + header().nnz);
    }

    uint32_t *row_pointers() {
        return column_indices() + header().nnz;
    }

    const uint32_t *row_pointers() const {
        return column_indices() + header().nnz;
    }

    uint32_t rows() const {
        return header().rows;
    }
    uint32_t cols() const {
        return header().cols;
    }
    uint32_t nnz() const {
        return header().nnz;
    }

    // Set matrix structure (must be called before adding values)
    void set_structure(uint32_t rows, uint32_t cols, uint32_t nnz) {
        header().rows = rows;
        header().cols = cols;
        header().nnz = nnz;

        // Initialize row pointers to zero
        std::fill(row_pointers(), row_pointers() + rows + 1, 0);
    }

    // Add a non-zero element (assumes elements are added in row-major order)
    void add_element(uint32_t row, uint32_t col, float value, uint32_t index) {
        assert(index < header().nnz);
        assert(row < header().rows);
        assert(col < header().cols);

        values()[index] = value;
        column_indices()[index] = col;

        // Update row pointers (simple approach - assumes sequential filling)
        for (uint32_t r = row + 1; r <= header().rows; ++r) {
            row_pointers()[r] = index + 1;
        }
    }

    // Matrix-vector multiplication: y = A * x
    void multiply_vector(const float *x, float *y) const {
        for (uint32_t row = 0; row < header().rows; ++row) {
            float sum = 0.0f;
            uint32_t start = row_pointers()[row];
            uint32_t end = row_pointers()[row + 1];

            for (uint32_t idx = start; idx < end; ++idx) {
                sum += values()[idx] * x[column_indices()[idx]];
            }

            y[row] = sum;
        }
    }

    // Get element value (slow for sparse matrices - for debugging only)
    float get_element(uint32_t row, uint32_t col) const {
        assert(row < header().rows && col < header().cols);

        uint32_t start = row_pointers()[row];
        uint32_t end = row_pointers()[row + 1];

        for (uint32_t idx = start; idx < end; ++idx) {
            if (column_indices()[idx] == col) {
                return values()[idx];
            }
        }

        return 0.0f; // Element not found (implicit zero)
    }

#ifdef PSYNE_ENABLE_EIGEN
    // Convert to Eigen sparse matrix (creates a copy)
    Eigen::SparseMatrix<float> to_eigen() const {
        Eigen::SparseMatrix<float> mat(header().rows, header().cols);
        mat.reserve(header().nnz);

        for (uint32_t row = 0; row < header().rows; ++row) {
            uint32_t start = row_pointers()[row];
            uint32_t end = row_pointers()[row + 1];

            for (uint32_t idx = start; idx < end; ++idx) {
                mat.insert(row, column_indices()[idx]) = values()[idx];
            }
        }

        mat.makeCompressed();
        return mat;
    }
#endif
};

} // namespace types
} // namespace psyne