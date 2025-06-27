#pragma once

#include "message.hpp"
#include <Eigen/Core>
#include <Eigen/LU>

namespace psyne {

// Fixed-size message types for ultra-high performance
// These have compile-time known sizes for zero overhead

template<size_t N>
class FixedFloatVector : public Message<FixedFloatVector<N>> {
public:
    static constexpr uint32_t message_type = 100 + N; // Unique type per size
    static constexpr size_t dimension = N;
    
    using Message<FixedFloatVector<N>>::Message;
    using EigenVector = Eigen::Matrix<float, N, 1>;
    using EigenMap = Eigen::Map<EigenVector>;
    using ConstEigenMap = Eigen::Map<const EigenVector>;
    
    // Direct array access
    float& operator[](size_t index) { return data_[index]; }
    const float& operator[](size_t index) const { return data_[index]; }
    
    // Assignment from initializer list
    FixedFloatVector& operator=(std::initializer_list<float> values) {
        size_t i = 0;
        for (float v : values) {
            if (i >= N) break;
            data_[i++] = v;
        }
        return *this;
    }
    
    // Eigen view - zero copy!
    EigenMap as_eigen() { 
        return EigenMap(data_); 
    }
    
    ConstEigenMap as_eigen() const { 
        return ConstEigenMap(data_); 
    }
    
    // GPU buffer support
    // The memory is already aligned and contiguous, perfect for GPU mapping
    // Platform-specific GPU code can directly use data() pointer
    
    // STL interface
    float* data() { return data_; }
    const float* data() const { return data_; }
    constexpr size_t size() const { return N; }
    
    float* begin() { return data_; }
    float* end() { return data_ + N; }
    const float* begin() const { return data_; }
    const float* end() const { return data_ + N; }
    
    static constexpr size_t calculate_size() {
        return sizeof(float) * N;
    }
    
private:
    friend class Message<FixedFloatVector<N>>;
    
    void initialize_storage(void* ptr) {
        data_ = static_cast<float*>(ptr);
        std::memset(data_, 0, sizeof(float) * N);
    }
    
    void initialize_view(void* ptr) {
        data_ = static_cast<float*>(ptr);
    }
    
    float* data_ = nullptr;
};

// Common sizes
using Vec3f = FixedFloatVector<3>;
using Vec4f = FixedFloatVector<4>;
using Vec64f = FixedFloatVector<64>;
using Vec128f = FixedFloatVector<128>;
using Vec256f = FixedFloatVector<256>;
using Vec512f = FixedFloatVector<512>;
using Vec1024f = FixedFloatVector<1024>;

// Fixed-size matrix with Eigen integration
template<size_t Rows, size_t Cols>
class FixedDoubleMatrix : public Message<FixedDoubleMatrix<Rows, Cols>> {
public:
    static constexpr uint32_t message_type = 2000 + Rows * 100 + Cols;
    static constexpr size_t rows = Rows;
    static constexpr size_t cols = Cols;
    
    using Message<FixedDoubleMatrix<Rows, Cols>>::Message;
    using EigenMatrix = Eigen::Matrix<double, Rows, Cols>;
    using EigenMap = Eigen::Map<EigenMatrix>;
    using ConstEigenMap = Eigen::Map<const EigenMatrix>;
    
    // Element access
    double& at(size_t row, size_t col) {
        return data_[row * Cols + col];
    }
    
    const double& at(size_t row, size_t col) const {
        return data_[row * Cols + col];
    }
    
    // Eigen view - zero copy!
    EigenMap as_eigen() {
        return EigenMap(data_);
    }
    
    ConstEigenMap as_eigen() const {
        return ConstEigenMap(data_);
    }
    
    // Fill from Eigen matrix
    void from_eigen(const EigenMatrix& mat) {
        std::memcpy(data_, mat.data(), sizeof(double) * Rows * Cols);
    }
    
    // GPU buffer support - memory is aligned and contiguous
    
    double* data() { return data_; }
    const double* data() const { return data_; }
    
    static constexpr size_t calculate_size() {
        return sizeof(double) * Rows * Cols;
    }
    
private:
    friend class Message<FixedDoubleMatrix<Rows, Cols>>;
    
    void initialize_storage(void* ptr) {
        data_ = static_cast<double*>(ptr);
        std::memset(data_, 0, sizeof(double) * Rows * Cols);
    }
    
    void initialize_view(void* ptr) {
        data_ = static_cast<double*>(ptr);
    }
    
    double* data_ = nullptr;
};

// Common matrix sizes
using Mat3d = FixedDoubleMatrix<3, 3>;
using Mat4d = FixedDoubleMatrix<4, 4>;
using Mat16d = FixedDoubleMatrix<16, 16>;
using Mat64d = FixedDoubleMatrix<64, 64>;
using Mat128d = FixedDoubleMatrix<128, 128>;

// GPU-optimized float matrix for ML workloads
template<size_t Rows, size_t Cols>
class FixedFloatMatrix : public Message<FixedFloatMatrix<Rows, Cols>> {
public:
    static constexpr uint32_t message_type = 3000 + Rows * 100 + Cols;
    static constexpr size_t rows = Rows;
    static constexpr size_t cols = Cols;
    
    using Message<FixedFloatMatrix<Rows, Cols>>::Message;
    using EigenMatrix = Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor>;
    using EigenMap = Eigen::Map<EigenMatrix>;
    using ConstEigenMap = Eigen::Map<const EigenMatrix>;
    
    float& at(size_t row, size_t col) {
        return data_[row * Cols + col];
    }
    
    const float& at(size_t row, size_t col) const {
        return data_[row * Cols + col];
    }
    
    EigenMap as_eigen() {
        return EigenMap(data_);
    }
    
    ConstEigenMap as_eigen() const {
        return ConstEigenMap(data_);
    }
    
    // GPU buffer support - memory is aligned and contiguous
    
    float* data() { return data_; }
    const float* data() const { return data_; }
    
    static constexpr size_t calculate_size() {
        return sizeof(float) * Rows * Cols;
    }
    
private:
    friend class Message<FixedFloatMatrix<Rows, Cols>>;
    
    void initialize_storage(void* ptr) {
        data_ = static_cast<float*>(ptr);
        std::memset(data_, 0, sizeof(float) * Rows * Cols);
    }
    
    void initialize_view(void* ptr) {
        data_ = static_cast<float*>(ptr);
    }
    
    float* data_ = nullptr;
};

// ML-specific sizes
using EmbeddingVec = FixedFloatVector<768>;    // BERT embeddings
using GPTEmbedding = FixedFloatVector<1024>;   // GPT embeddings
using AttentionMat = FixedFloatMatrix<64, 64>; // Attention heads

}  // namespace psyne