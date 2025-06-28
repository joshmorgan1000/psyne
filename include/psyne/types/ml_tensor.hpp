#pragma once

#include "../psyne.hpp"
#include <vector>
#include <array>
#include <string>

namespace psyne {
namespace types {

// Enhanced tensor type optimized for ML workloads
// Supports multiple data types and ML-specific operations
template<typename T>
class MLTensor : public Message<MLTensor<T>> {
public:
    using value_type = T;
    static constexpr uint32_t message_type = 29; // Default, will be specialized
    
    // Common ML data layouts
    enum class Layout {
        NCHW,  // Batch, Channels, Height, Width (common for images)
        NHWC,  // Batch, Height, Width, Channels (TensorFlow default)
        CHW,   // Channels, Height, Width (single image)
        HWC,   // Height, Width, Channels (single image)
        NCL,   // Batch, Channels, Length (1D convolutions)
        NLC,   // Batch, Length, Channels (sequence data)
        Matrix // 2D matrix (rows, cols)
    };
    
    using Message<MLTensor<T>>::Message;
    
    static size_t calculate_size() {
        return 4096; // Default size for ML tensors
    }
    
    // Shape and layout management
    void set_shape(const std::vector<size_t>& shape, Layout layout = Layout::Matrix);
    const std::vector<size_t>& shape() const;
    Layout layout() const;
    size_t ndim() const;
    size_t total_elements() const;
    
    // Element access
    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;
    
    // Raw data access
    T* data();
    const T* data() const;
    
    // ML-specific access patterns
    T& at_nchw(size_t n, size_t c, size_t h, size_t w);
    const T& at_nchw(size_t n, size_t c, size_t h, size_t w) const;
    T& at_nhwc(size_t n, size_t h, size_t w, size_t c);
    const T& at_nhwc(size_t n, size_t h, size_t w, size_t c) const;
    
    // Batch operations
    size_t batch_size() const; // First dimension
    void set_batch_size(size_t batch_size);
    
    // In-place operations for ML
    MLTensor& operator+=(const MLTensor& other);
    MLTensor& operator-=(const MLTensor& other);
    MLTensor& operator*=(const MLTensor& other); // Element-wise
    MLTensor& operator*=(const T& scalar);
    
    // ML activation functions (in-place)
    void relu();
    void sigmoid();
    void tanh_activation();
    void softmax(size_t axis = -1); // Along specified axis
    
    // Normalization operations (in-place)
    void normalize_l2(); // L2 normalization
    void batch_normalize(const T* mean, const T* variance, const T* scale, const T* bias, T epsilon = 1e-5);
    
    // Reduction operations
    T sum() const;
    T mean() const;
    T variance(T mean_val) const;
    T max() const;
    T min() const;
    
    // Statistical operations along axes
    void reduce_sum(size_t axis, MLTensor& result) const;
    void reduce_mean(size_t axis, MLTensor& result) const;
    
    // Reshaping operations (zero-copy when possible)
    void reshape(const std::vector<size_t>& new_shape);
    void transpose(const std::vector<size_t>& axes);
    
    // Quantization support
    void quantize_to_int8(MLTensor<int8_t>& output, T scale, int8_t zero_point) const;
    void dequantize_from_int8(const MLTensor<int8_t>& input, T scale, int8_t zero_point);
    
    // Metadata for debugging/profiling
    void set_name(const std::string& name);
    const std::string& name() const;
    
    void initialize();
    void before_send() override {}

private:
    size_t calculate_index(const std::vector<size_t>& indices) const;
    size_t get_header_size() const;
    uint8_t* get_header_ptr();
    const uint8_t* get_header_ptr() const;
    
    // Header layout:
    // [shape_size: size_t][shape: size_t*][layout: int][name_length: size_t][name: char*][data...]
};

// Specializations for common ML data types
using MLTensorF = MLTensor<float>;
using MLTensorD = MLTensor<double>;
using MLTensorI32 = MLTensor<int32_t>;
using MLTensorI8 = MLTensor<int8_t>;
using MLTensorU8 = MLTensor<uint8_t>;

// Specialized message type IDs
template<> constexpr uint32_t MLTensor<float>::message_type = 29;
template<> constexpr uint32_t MLTensor<double>::message_type = 30;
template<> constexpr uint32_t MLTensor<int32_t>::message_type = 31;
template<> constexpr uint32_t MLTensor<int8_t>::message_type = 32;
template<> constexpr uint32_t MLTensor<uint8_t>::message_type = 33;

} // namespace types
} // namespace psyne