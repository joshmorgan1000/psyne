#include "../../include/psyne/types/ml_tensor.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>

namespace psyne {
namespace types {

template<typename T>
void MLTensor<T>::initialize() {
    if (this->Message<MLTensor<T>>::data()) {
        // Initialize with empty shape
        uint8_t* ptr = this->Message<MLTensor<T>>::data();
        *reinterpret_cast<size_t*>(ptr) = 0; // shape_size = 0
        ptr += sizeof(size_t);
        *reinterpret_cast<int*>(ptr) = static_cast<int>(Layout::Matrix); // default layout
        ptr += sizeof(int);
        *reinterpret_cast<size_t*>(ptr) = 0; // name_length = 0
    }
}

template<typename T>
size_t MLTensor<T>::get_header_size() const {
    if (!this->Message<MLTensor<T>>::data()) return 0;
    
    const uint8_t* ptr = this->Message<MLTensor<T>>::data();
    size_t shape_size = *reinterpret_cast<const size_t*>(ptr);
    ptr += sizeof(size_t) + shape_size * sizeof(size_t); // Skip shape
    ptr += sizeof(int); // Skip layout
    size_t name_length = *reinterpret_cast<const size_t*>(ptr);
    
    return sizeof(size_t) + shape_size * sizeof(size_t) + sizeof(int) + sizeof(size_t) + name_length;
}

template<typename T>
uint8_t* MLTensor<T>::get_header_ptr() {
    return this->Message<MLTensor<T>>::data();
}

template<typename T>
const uint8_t* MLTensor<T>::get_header_ptr() const {
    return this->Message<MLTensor<T>>::data();
}

template<typename T>
void MLTensor<T>::set_shape(const std::vector<size_t>& shape, Layout layout) {
    if (!this->Message<MLTensor<T>>::data()) return;
    
    // Calculate total elements
    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }
    
    // Get current name to preserve it
    std::string current_name = name();
    
    // Calculate new header size
    size_t new_header_size = sizeof(size_t) + shape.size() * sizeof(size_t) + sizeof(int) + 
                            sizeof(size_t) + current_name.size();
    size_t available_data_elements = (this->Message<MLTensor<T>>::size() - new_header_size) / sizeof(T);
    
    if (total > available_data_elements) {
        throw std::runtime_error("Tensor shape exceeds capacity");
    }
    
    // Write new header
    uint8_t* ptr = this->Message<MLTensor<T>>::data();
    
    // Shape size and shape
    *reinterpret_cast<size_t*>(ptr) = shape.size();
    ptr += sizeof(size_t);
    
    for (size_t i = 0; i < shape.size(); ++i) {
        reinterpret_cast<size_t*>(ptr)[i] = shape[i];
    }
    ptr += shape.size() * sizeof(size_t);
    
    // Layout
    *reinterpret_cast<int*>(ptr) = static_cast<int>(layout);
    ptr += sizeof(int);
    
    // Name
    *reinterpret_cast<size_t*>(ptr) = current_name.size();
    ptr += sizeof(size_t);
    std::memcpy(ptr, current_name.data(), current_name.size());
}

template<typename T>
const std::vector<size_t>& MLTensor<T>::shape() const {
    static thread_local std::vector<size_t> cached_shape;
    
    if (!this->Message<MLTensor<T>>::data()) {
        cached_shape.clear();
        return cached_shape;
    }
    
    const uint8_t* ptr = this->Message<MLTensor<T>>::data();
    size_t shape_size = *reinterpret_cast<const size_t*>(ptr);
    ptr += sizeof(size_t);
    
    const size_t* shape_data = reinterpret_cast<const size_t*>(ptr);
    cached_shape.assign(shape_data, shape_data + shape_size);
    return cached_shape;
}

template<typename T>
typename MLTensor<T>::Layout MLTensor<T>::layout() const {
    if (!this->Message<MLTensor<T>>::data()) return Layout::Matrix;
    
    const uint8_t* ptr = this->Message<MLTensor<T>>::data();
    size_t shape_size = *reinterpret_cast<const size_t*>(ptr);
    ptr += sizeof(size_t) + shape_size * sizeof(size_t);
    
    return static_cast<Layout>(*reinterpret_cast<const int*>(ptr));
}

template<typename T>
size_t MLTensor<T>::ndim() const {
    return shape().size();
}

template<typename T>
size_t MLTensor<T>::total_elements() const {
    const auto& s = shape();
    return std::accumulate(s.begin(), s.end(), 1UL, std::multiplies<size_t>());
}

template<typename T>
T* MLTensor<T>::data() {
    if (!this->Message<MLTensor<T>>::data()) return nullptr;
    return reinterpret_cast<T*>(this->Message<MLTensor<T>>::data() + get_header_size());
}

template<typename T>
const T* MLTensor<T>::data() const {
    return const_cast<MLTensor<T>*>(this)->data();
}

template<typename T>
T& MLTensor<T>::operator[](size_t index) {
    if (index >= total_elements()) {
        throw std::out_of_range("MLTensor index out of range");
    }
    return data()[index];
}

template<typename T>
const T& MLTensor<T>::operator[](size_t index) const {
    if (index >= total_elements()) {
        throw std::out_of_range("MLTensor index out of range");
    }
    return data()[index];
}

template<typename T>
size_t MLTensor<T>::calculate_index(const std::vector<size_t>& indices) const {
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

template<typename T>
T& MLTensor<T>::at(const std::vector<size_t>& indices) {
    return data()[calculate_index(indices)];
}

template<typename T>
const T& MLTensor<T>::at(const std::vector<size_t>& indices) const {
    return data()[calculate_index(indices)];
}

template<typename T>
T& MLTensor<T>::at_nchw(size_t n, size_t c, size_t h, size_t w) {
    return at({n, c, h, w});
}

template<typename T>
const T& MLTensor<T>::at_nchw(size_t n, size_t c, size_t h, size_t w) const {
    return at({n, c, h, w});
}

template<typename T>
T& MLTensor<T>::at_nhwc(size_t n, size_t h, size_t w, size_t c) {
    return at({n, h, w, c});
}

template<typename T>
const T& MLTensor<T>::at_nhwc(size_t n, size_t h, size_t w, size_t c) const {
    return at({n, h, w, c});
}

template<typename T>
size_t MLTensor<T>::batch_size() const {
    const auto& s = shape();
    return s.empty() ? 0 : s[0];
}

template<typename T>
void MLTensor<T>::set_batch_size(size_t batch_size) {
    auto s = shape();
    if (!s.empty()) {
        s[0] = batch_size;
        set_shape(s, layout());
    }
}

template<typename T>
MLTensor<T>& MLTensor<T>::operator+=(const MLTensor<T>& other) {
    size_t min_elements = std::min(total_elements(), other.total_elements());
    for (size_t i = 0; i < min_elements; ++i) {
        (*this)[i] += other[i];
    }
    return *this;
}

template<typename T>
MLTensor<T>& MLTensor<T>::operator-=(const MLTensor<T>& other) {
    size_t min_elements = std::min(total_elements(), other.total_elements());
    for (size_t i = 0; i < min_elements; ++i) {
        (*this)[i] -= other[i];
    }
    return *this;
}

template<typename T>
MLTensor<T>& MLTensor<T>::operator*=(const MLTensor<T>& other) {
    size_t min_elements = std::min(total_elements(), other.total_elements());
    for (size_t i = 0; i < min_elements; ++i) {
        (*this)[i] *= other[i];
    }
    return *this;
}

template<typename T>
MLTensor<T>& MLTensor<T>::operator*=(const T& scalar) {
    for (size_t i = 0; i < total_elements(); ++i) {
        (*this)[i] *= scalar;
    }
    return *this;
}

template<typename T>
void MLTensor<T>::relu() {
    for (size_t i = 0; i < total_elements(); ++i) {
        (*this)[i] = std::max(T(0), (*this)[i]);
    }
}

template<typename T>
void MLTensor<T>::sigmoid() {
    for (size_t i = 0; i < total_elements(); ++i) {
        (*this)[i] = T(1) / (T(1) + std::exp(-(*this)[i]));
    }
}

template<typename T>
void MLTensor<T>::tanh_activation() {
    for (size_t i = 0; i < total_elements(); ++i) {
        (*this)[i] = std::tanh((*this)[i]);
    }
}

template<typename T>
void MLTensor<T>::softmax(size_t axis) {
    // Simplified softmax for the last axis
    const auto& s = shape();
    if (axis == static_cast<size_t>(-1)) {
        axis = s.size() - 1;
    }
    
    if (axis >= s.size()) {
        throw std::invalid_argument("Invalid axis for softmax");
    }
    
    size_t axis_size = s[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;
    
    for (size_t i = 0; i < axis; ++i) {
        outer_size *= s[i];
    }
    for (size_t i = axis + 1; i < s.size(); ++i) {
        inner_size *= s[i];
    }
    
    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            // Find max for numerical stability
            T max_val = (*this)[outer * axis_size * inner_size + inner];
            for (size_t i = 1; i < axis_size; ++i) {
                max_val = std::max(max_val, (*this)[outer * axis_size * inner_size + i * inner_size + inner]);
            }
            
            // Compute exp and sum
            T sum = T(0);
            for (size_t i = 0; i < axis_size; ++i) {
                size_t idx = outer * axis_size * inner_size + i * inner_size + inner;
                (*this)[idx] = std::exp((*this)[idx] - max_val);
                sum += (*this)[idx];
            }
            
            // Normalize
            for (size_t i = 0; i < axis_size; ++i) {
                size_t idx = outer * axis_size * inner_size + i * inner_size + inner;
                (*this)[idx] /= sum;
            }
        }
    }
}

template<typename T>
T MLTensor<T>::sum() const {
    T result = T(0);
    for (size_t i = 0; i < total_elements(); ++i) {
        result += (*this)[i];
    }
    return result;
}

template<typename T>
T MLTensor<T>::mean() const {
    return total_elements() > 0 ? sum() / T(total_elements()) : T(0);
}

template<typename T>
T MLTensor<T>::max() const {
    if (total_elements() == 0) return T(0);
    return *std::max_element(data(), data() + total_elements());
}

template<typename T>
T MLTensor<T>::min() const {
    if (total_elements() == 0) return T(0);
    return *std::min_element(data(), data() + total_elements());
}

template<typename T>
void MLTensor<T>::normalize_l2() {
    T sum_squares = T(0);
    size_t total = total_elements();
    
    // Calculate L2 norm
    for (size_t i = 0; i < total; ++i) {
        sum_squares += (*this)[i] * (*this)[i];
    }
    
    T norm = std::sqrt(sum_squares);
    if (norm > T(1e-12)) { // Avoid division by zero
        for (size_t i = 0; i < total; ++i) {
            (*this)[i] /= norm;
        }
    }
}

template<typename T>
void MLTensor<T>::set_name(const std::string& name) {
    // This is a simplified implementation - in practice would need to manage header size carefully
    // For now, just store in a static thread-local map
    static thread_local std::map<const void*, std::string> name_map;
    name_map[this->Message<MLTensor<T>>::data()] = name;
}

template<typename T>
const std::string& MLTensor<T>::name() const {
    static thread_local std::map<const void*, std::string> name_map;
    static const std::string empty_name;
    auto it = name_map.find(this->Message<MLTensor<T>>::data());
    return it != name_map.end() ? it->second : empty_name;
}

// Explicit template instantiations for common ML types
template class MLTensor<float>;
template class MLTensor<double>;
template class MLTensor<int32_t>;
template class MLTensor<int8_t>;
template class MLTensor<uint8_t>;

} // namespace types
} // namespace psyne