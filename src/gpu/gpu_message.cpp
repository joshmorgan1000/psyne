#include "../../include/psyne/gpu/gpu_message.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace psyne {
namespace gpu {

// GPUFloatVector implementation
void GPUFloatVector::initialize() {
    // Initialize size to 0
    if (this->data()) {
        *reinterpret_cast<size_t*>(this->data()) = 0;
    }
}

float& GPUFloatVector::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("GPUFloatVector index out of range");
    }
    return begin()[index];
}

const float& GPUFloatVector::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("GPUFloatVector index out of range");
    }
    return begin()[index];
}

float* GPUFloatVector::begin() {
    return reinterpret_cast<float*>(this->data() + sizeof(size_t));
}

float* GPUFloatVector::end() {
    return begin() + size();
}

const float* GPUFloatVector::begin() const {
    return reinterpret_cast<const float*>(this->data() + sizeof(size_t));
}

const float* GPUFloatVector::end() const {
    return begin() + size();
}

size_t GPUFloatVector::size() const {
    if (!this->data()) return 0;
    size_t stored_size = *reinterpret_cast<const size_t*>(this->data());
    if (stored_size > capacity()) {
        return 0;  // Return 0 for invalid size
    }
    return stored_size;
}

size_t GPUFloatVector::capacity() const {
    if (!this->data()) return 0;
    return (this->Message<GPUFloatVector>::size() - sizeof(size_t)) / sizeof(float);
}

void GPUFloatVector::resize(size_t new_size) {
    if (new_size > capacity()) {
        throw std::runtime_error("Cannot resize beyond capacity");
    }
    if (this->data()) {
        *reinterpret_cast<size_t*>(this->data()) = new_size;
    }
}

GPUFloatVector& GPUFloatVector::operator=(std::initializer_list<float> values) {
    resize(values.size());
    std::copy(values.begin(), values.end(), begin());
    return *this;
}

std::unique_ptr<GPUBuffer> GPUFloatVector::to_gpu_buffer(GPUContext& context) {
    auto factory = context.create_buffer_factory();
    if (!factory) {
        return nullptr;
    }
    
    size_t data_size = size() * sizeof(float);
    auto buffer = factory->create_buffer(data_size, BufferUsage::Dynamic, MemoryAccess::Shared);
    
    if (buffer) {
        // Copy data to GPU buffer
        void* gpu_ptr = buffer->map();
        if (gpu_ptr) {
            std::memcpy(gpu_ptr, begin(), data_size);
            buffer->unmap();
            buffer->flush();
        }
        
        // Cache the GPU buffer
        gpu_buffer_ = std::unique_ptr<GPUBuffer>(buffer.release());
    }
    
    return std::unique_ptr<GPUBuffer>(buffer.release());
}

void GPUFloatVector::from_gpu_buffer(const GPUBuffer& buffer) {
    const void* gpu_ptr = buffer.map();
    if (!gpu_ptr) {
        throw std::runtime_error("Failed to map GPU buffer");
    }
    
    size_t data_size = buffer.size();
    size_t element_count = data_size / sizeof(float);
    
    if (element_count > capacity()) {
        throw std::runtime_error("GPU buffer too large for vector capacity");
    }
    
    resize(element_count);
    std::memcpy(begin(), gpu_ptr, data_size);
    
    const_cast<GPUBuffer&>(buffer).unmap();
}

Eigen::Map<Eigen::VectorXf> GPUFloatVector::as_eigen() {
    return Eigen::Map<Eigen::VectorXf>(begin(), size());
}

Eigen::Map<const Eigen::VectorXf> GPUFloatVector::as_eigen() const {
    return Eigen::Map<const Eigen::VectorXf>(begin(), size());
}

void GPUFloatVector::gpu_scale(GPUContext& context, float factor) {
    // Simple compute shader to scale vector elements
    std::string shader = R"(
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void compute_main(device float* data [[buffer(0)]],
                               constant float& factor [[buffer(1)]],
                               uint index [[thread_position_in_grid]]) {
            data[index] *= factor;
        }
    )";
    
    if (!gpu_buffer_) {
        gpu_buffer_ = to_gpu_buffer(context);
    }
    
    if (gpu_buffer_) {
        // Create a buffer for the factor
        auto factory = context.create_buffer_factory();
        auto factor_buffer = factory->create_buffer(sizeof(float), BufferUsage::Static, MemoryAccess::Shared);
        
        if (factor_buffer) {
            float* factor_ptr = static_cast<float*>(factor_buffer->map());
            *factor_ptr = factor;
            factor_buffer->unmap();
            factor_buffer->flush();
            
            // Dispatch compute (simplified - real implementation would compile shader)
            context.dispatch_compute(shader.data(), shader.size(), *gpu_buffer_, *gpu_buffer_, size());
            context.sync();
            
            // Copy result back to CPU
            from_gpu_buffer(*gpu_buffer_);
        }
    }
}

void GPUFloatVector::gpu_add(GPUContext& context, const GPUFloatVector& other) {
    // Placeholder - would implement vector addition on GPU
    std::cout << "GPU vector addition not yet implemented\n";
}

float GPUFloatVector::gpu_dot_product(GPUContext& context, const GPUFloatVector& other) {
    // Placeholder - would implement dot product on GPU
    std::cout << "GPU dot product not yet implemented\n";
    return 0.0f;
}

// GPUMatrix implementation
void GPUMatrix::initialize() {
    // Initialize dimensions to 0x0
    if (this->Message<GPUMatrix>::data()) {
        size_t* header = reinterpret_cast<size_t*>(this->Message<GPUMatrix>::data());
        header[0] = 0;  // rows
        header[1] = 0;  // cols
    }
}

void GPUMatrix::set_dimensions(size_t rows, size_t cols) {
    if (!this->Message<GPUMatrix>::data()) return;
    
    size_t total_elements = rows * cols;
    size_t available_elements = (this->Message<GPUMatrix>::size() - 2 * sizeof(size_t)) / sizeof(float);
    
    if (total_elements > available_elements) {
        throw std::runtime_error("Matrix dimensions exceed capacity");
    }
    
    size_t* header = reinterpret_cast<size_t*>(this->Message<GPUMatrix>::data());
    header[0] = rows;
    header[1] = cols;
}

size_t GPUMatrix::rows() const {
    if (!this->Message<GPUMatrix>::data()) return 0;
    return reinterpret_cast<const size_t*>(this->Message<GPUMatrix>::data())[0];
}

size_t GPUMatrix::cols() const {
    if (!this->Message<GPUMatrix>::data()) return 0;
    return reinterpret_cast<const size_t*>(this->Message<GPUMatrix>::data())[1];
}

float& GPUMatrix::at(size_t row, size_t col) {
    if (row >= rows() || col >= cols()) {
        throw std::out_of_range("GPUMatrix index out of range");
    }
    float* matrix_data = reinterpret_cast<float*>(this->Message<GPUMatrix>::data() + 2 * sizeof(size_t));
    return matrix_data[row * cols() + col];
}

const float& GPUMatrix::at(size_t row, size_t col) const {
    if (row >= rows() || col >= cols()) {
        throw std::out_of_range("GPUMatrix index out of range");
    }
    const float* matrix_data = reinterpret_cast<const float*>(this->Message<GPUMatrix>::data() + 2 * sizeof(size_t));
    return matrix_data[row * cols() + col];
}

float* GPUMatrix::data() {
    return reinterpret_cast<float*>(this->Message<GPUMatrix>::data() + 2 * sizeof(size_t));
}

const float* GPUMatrix::data() const {
    return reinterpret_cast<const float*>(this->Message<GPUMatrix>::data() + 2 * sizeof(size_t));
}

std::unique_ptr<GPUBuffer> GPUMatrix::to_gpu_buffer(GPUContext& context) {
    auto factory = context.create_buffer_factory();
    if (!factory) {
        return nullptr;
    }
    
    size_t data_size = rows() * cols() * sizeof(float);
    auto buffer = factory->create_buffer(data_size, BufferUsage::Dynamic, MemoryAccess::Shared);
    
    if (buffer) {
        void* gpu_ptr = buffer->map();
        if (gpu_ptr) {
            std::memcpy(gpu_ptr, data(), data_size);
            buffer->unmap();
            buffer->flush();
        }
        
        gpu_buffer_ = std::unique_ptr<GPUBuffer>(buffer.release());
    }
    
    return std::unique_ptr<GPUBuffer>(buffer.release());
}

void GPUMatrix::from_gpu_buffer(const GPUBuffer& buffer) {
    const void* gpu_ptr = buffer.map();
    if (!gpu_ptr) {
        throw std::runtime_error("Failed to map GPU buffer");
    }
    
    size_t data_size = buffer.size();
    std::memcpy(data(), gpu_ptr, data_size);
    
    const_cast<GPUBuffer&>(buffer).unmap();
}

Eigen::Map<Eigen::MatrixXf> GPUMatrix::as_eigen() {
    return Eigen::Map<Eigen::MatrixXf>(data(), rows(), cols());
}

Eigen::Map<const Eigen::MatrixXf> GPUMatrix::as_eigen() const {
    return Eigen::Map<const Eigen::MatrixXf>(data(), rows(), cols());
}

void GPUMatrix::gpu_multiply(GPUContext& context, const GPUMatrix& other, GPUMatrix& result) {
    std::cout << "GPU matrix multiplication not yet implemented\n";
}

void GPUMatrix::gpu_transpose(GPUContext& context, GPUMatrix& result) {
    std::cout << "GPU matrix transpose not yet implemented\n";
}

void GPUMatrix::gpu_relu(GPUContext& context) {
    std::cout << "GPU ReLU activation not yet implemented\n";
}

// GPUTensor implementation
void GPUTensor::initialize() {
    // Initialize with empty shape
    if (this->Message<GPUTensor>::data()) {
        size_t* shape_size = reinterpret_cast<size_t*>(this->Message<GPUTensor>::data());
        *shape_size = 0;
    }
}

void GPUTensor::set_shape(const std::vector<size_t>& shape) {
    if (!this->Message<GPUTensor>::data()) return;
    
    // Calculate total elements
    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }
    
    // Check if it fits
    size_t header_size = sizeof(size_t) + shape.size() * sizeof(size_t);
    size_t available_elements = (this->Message<GPUTensor>::size() - header_size) / sizeof(float);
    
    if (total > available_elements) {
        throw std::runtime_error("Tensor shape exceeds capacity");
    }
    
    // Store shape
    uint8_t* ptr = this->Message<GPUTensor>::data();
    size_t* shape_size = reinterpret_cast<size_t*>(ptr);
    *shape_size = shape.size();
    ptr += sizeof(size_t);
    
    size_t* shape_data = reinterpret_cast<size_t*>(ptr);
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_data[i] = shape[i];
    }
}

const std::vector<size_t>& GPUTensor::shape() const {
    static thread_local std::vector<size_t> cached_shape;
    
    if (!this->Message<GPUTensor>::data()) {
        cached_shape.clear();
        return cached_shape;
    }
    
    const uint8_t* ptr = this->Message<GPUTensor>::data();
    size_t shape_size = *reinterpret_cast<const size_t*>(ptr);
    ptr += sizeof(size_t);
    
    const size_t* shape_data = reinterpret_cast<const size_t*>(ptr);
    
    cached_shape.assign(shape_data, shape_data + shape_size);
    return cached_shape;
}

size_t GPUTensor::total_elements() const {
    const auto& s = shape();
    size_t total = 1;
    for (size_t dim : s) {
        total *= dim;
    }
    return total;
}

size_t GPUTensor::ndim() const {
    const auto& s = shape();
    return s.size();
}

float* GPUTensor::data() {
    if (!this->Message<GPUTensor>::data()) return nullptr;
    
    const auto& s = shape();
    size_t header_size = sizeof(size_t) + s.size() * sizeof(size_t);
    return reinterpret_cast<float*>(this->Message<GPUTensor>::data() + header_size);
}

const float* GPUTensor::data() const {
    return const_cast<GPUTensor*>(this)->data();
}

float& GPUTensor::operator[](size_t index) {
    if (index >= total_elements()) {
        throw std::out_of_range("GPUTensor index out of range");
    }
    return data()[index];
}

const float& GPUTensor::operator[](size_t index) const {
    if (index >= total_elements()) {
        throw std::out_of_range("GPUTensor index out of range");
    }
    return data()[index];
}

size_t GPUTensor::calculate_index(const std::vector<size_t>& indices) const {
    const auto& s = shape();
    if (indices.size() != s.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    
    size_t index = 0;
    size_t multiplier = 1;
    
    for (int i = s.size() - 1; i >= 0; --i) {
        if (indices[i] >= s[i]) {
            throw std::out_of_range("Tensor index out of range");
        }
        index += indices[i] * multiplier;
        multiplier *= s[i];
    }
    
    return index;
}

float& GPUTensor::at(const std::vector<size_t>& indices) {
    return data()[calculate_index(indices)];
}

const float& GPUTensor::at(const std::vector<size_t>& indices) const {
    return data()[calculate_index(indices)];
}

std::unique_ptr<GPUBuffer> GPUTensor::to_gpu_buffer(GPUContext& context) {
    auto factory = context.create_buffer_factory();
    if (!factory) {
        return nullptr;
    }
    
    size_t data_size = total_elements() * sizeof(float);
    auto buffer = factory->create_buffer(data_size, BufferUsage::Dynamic, MemoryAccess::Shared);
    
    if (buffer) {
        void* gpu_ptr = buffer->map();
        if (gpu_ptr) {
            std::memcpy(gpu_ptr, data(), data_size);
            buffer->unmap();
            buffer->flush();
        }
        
        gpu_buffer_ = std::unique_ptr<GPUBuffer>(buffer.release());
    }
    
    return std::unique_ptr<GPUBuffer>(buffer.release());
}

void GPUTensor::from_gpu_buffer(const GPUBuffer& buffer) {
    const void* gpu_ptr = buffer.map();
    if (!gpu_ptr) {
        throw std::runtime_error("Failed to map GPU buffer");
    }
    
    size_t data_size = buffer.size();
    std::memcpy(data(), gpu_ptr, data_size);
    
    const_cast<GPUBuffer&>(buffer).unmap();
}

void GPUTensor::gpu_convolution_2d(GPUContext& context, const GPUTensor& kernel, GPUTensor& result) {
    std::cout << "GPU 2D convolution not yet implemented\n";
}

void GPUTensor::gpu_batch_norm(GPUContext& context, const GPUTensor& mean, const GPUTensor& variance) {
    std::cout << "GPU batch normalization not yet implemented\n";
}

void GPUTensor::gpu_softmax(GPUContext& context, size_t axis) {
    std::cout << "GPU softmax not yet implemented\n";
}

} // namespace gpu
} // namespace psyne