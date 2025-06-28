#pragma once

#include "gpu_buffer.hpp"
#include "../psyne.hpp"
#include <Eigen/Dense>

namespace psyne {
namespace gpu {

// GPU-aware message base class
template<typename Derived>
class GPUMessage : public Message<Derived> {
public:
    using Message<Derived>::Message;
    
    // GPU buffer management
    virtual std::unique_ptr<GPUBuffer> to_gpu_buffer(GPUContext& context) = 0;
    virtual void from_gpu_buffer(const GPUBuffer& buffer) = 0;
    
    // Check if data is currently on GPU
    virtual bool is_on_gpu() const { return gpu_buffer_ != nullptr; }
    
    // Get GPU buffer if available
    GPUBuffer* gpu_buffer() const { return gpu_buffer_.get(); }

protected:
    std::unique_ptr<GPUBuffer> gpu_buffer_;
};

// GPU-aware float vector
class GPUFloatVector : public GPUMessage<GPUFloatVector> {
public:
    static constexpr uint32_t message_type = 10;  // Different from regular FloatVector
    
    using GPUMessage<GPUFloatVector>::GPUMessage;
    
    // Calculate required size
    static size_t calculate_size() {
        return 1024;  // Default size
    }
    
    // CPU interface (same as FloatVector)
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    float* begin();
    float* end();
    const float* begin() const;
    const float* end() const;
    size_t size() const;
    size_t capacity() const;
    void resize(size_t new_size);
    GPUFloatVector& operator=(std::initializer_list<float> values);
    
    // GPU interface
    std::unique_ptr<GPUBuffer> to_gpu_buffer(GPUContext& context) override;
    void from_gpu_buffer(const GPUBuffer& buffer) override;
    
    // GPU compute operations
    void gpu_scale(GPUContext& context, float factor);
    void gpu_add(GPUContext& context, const GPUFloatVector& other);
    float gpu_dot_product(GPUContext& context, const GPUFloatVector& other);
    
    // Eigen integration with GPU awareness
    Eigen::Map<Eigen::VectorXf> as_eigen();
    Eigen::Map<const Eigen::VectorXf> as_eigen() const;
    
    void initialize();
};

// GPU-aware matrix for ML workloads
class GPUMatrix : public GPUMessage<GPUMatrix> {
public:
    static constexpr uint32_t message_type = 11;
    
    using GPUMessage<GPUMatrix>::GPUMessage;
    
    static size_t calculate_size() {
        return 8192;  // Default size
    }
    
    // Dimension management
    void set_dimensions(size_t rows, size_t cols);
    size_t rows() const;
    size_t cols() const;
    
    // Element access
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    float* data();
    const float* data() const;
    
    // GPU interface
    std::unique_ptr<GPUBuffer> to_gpu_buffer(GPUContext& context) override;
    void from_gpu_buffer(const GPUBuffer& buffer) override;
    
    // GPU matrix operations
    void gpu_multiply(GPUContext& context, const GPUMatrix& other, GPUMatrix& result);
    void gpu_transpose(GPUContext& context, GPUMatrix& result);
    void gpu_relu(GPUContext& context);  // In-place ReLU activation
    
    // Eigen integration
    Eigen::Map<Eigen::MatrixXf> as_eigen();
    Eigen::Map<const Eigen::MatrixXf> as_eigen() const;
    
    void initialize();
};

// GPU tensor for deep learning
class GPUTensor : public GPUMessage<GPUTensor> {
public:
    static constexpr uint32_t message_type = 12;
    
    using GPUMessage<GPUTensor>::GPUMessage;
    
    static size_t calculate_size() {
        return 32768;  // 32KB default
    }
    
    // Shape management
    void set_shape(const std::vector<size_t>& shape);
    const std::vector<size_t>& shape() const;
    size_t total_elements() const;
    size_t ndim() const;
    
    // Data access
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    float* data();
    const float* data() const;
    
    // Multi-dimensional indexing
    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;
    
    // GPU interface
    std::unique_ptr<GPUBuffer> to_gpu_buffer(GPUContext& context) override;
    void from_gpu_buffer(const GPUBuffer& buffer) override;
    
    // GPU tensor operations
    void gpu_convolution_2d(GPUContext& context, const GPUTensor& kernel, GPUTensor& result);
    void gpu_batch_norm(GPUContext& context, const GPUTensor& mean, const GPUTensor& variance);
    void gpu_softmax(GPUContext& context, size_t axis = -1);
    
    void initialize();

private:
    size_t calculate_index(const std::vector<size_t>& indices) const;
};

} // namespace gpu
} // namespace psyne