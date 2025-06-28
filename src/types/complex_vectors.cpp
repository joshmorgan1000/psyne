#include "../../include/psyne/types/complex_vectors.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace psyne {
namespace types {

// ComplexVector<T> implementation
template<typename T>
void ComplexVector<T>::initialize() {
    // Initialize size to 0
    if (this->Message<ComplexVector<T>>::data()) {
        *reinterpret_cast<size_t*>(this->Message<ComplexVector<T>>::data()) = 0;
    }
}

template<typename T>
typename ComplexVector<T>::value_type& ComplexVector<T>::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("ComplexVector index out of range");
    }
    return begin()[index];
}

template<typename T>
const typename ComplexVector<T>::value_type& ComplexVector<T>::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("ComplexVector index out of range");
    }
    return begin()[index];
}

template<typename T>
typename ComplexVector<T>::value_type& ComplexVector<T>::at(size_t index) {
    return (*this)[index];
}

template<typename T>
const typename ComplexVector<T>::value_type& ComplexVector<T>::at(size_t index) const {
    return (*this)[index];
}

template<typename T>
typename ComplexVector<T>::value_type* ComplexVector<T>::begin() {
    return reinterpret_cast<value_type*>(this->Message<ComplexVector<T>>::data() + sizeof(size_t));
}

template<typename T>
typename ComplexVector<T>::value_type* ComplexVector<T>::end() {
    return begin() + size();
}

template<typename T>
const typename ComplexVector<T>::value_type* ComplexVector<T>::begin() const {
    return reinterpret_cast<const value_type*>(this->Message<ComplexVector<T>>::data() + sizeof(size_t));
}

template<typename T>
const typename ComplexVector<T>::value_type* ComplexVector<T>::end() const {
    return begin() + size();
}

template<typename T>
typename ComplexVector<T>::value_type* ComplexVector<T>::data() {
    return begin();
}

template<typename T>
const typename ComplexVector<T>::value_type* ComplexVector<T>::data() const {
    return begin();
}

template<typename T>
size_t ComplexVector<T>::size() const {
    if (!this->Message<ComplexVector<T>>::data()) return 0;
    size_t stored_size = *reinterpret_cast<const size_t*>(this->Message<ComplexVector<T>>::data());
    if (stored_size > capacity()) {
        return 0;  // Return 0 for invalid size
    }
    return stored_size;
}

template<typename T>
size_t ComplexVector<T>::capacity() const {
    if (!this->Message<ComplexVector<T>>::data()) return 0;
    return (this->Message<ComplexVector<T>>::size() - sizeof(size_t)) / sizeof(value_type);
}

template<typename T>
void ComplexVector<T>::resize(size_t new_size) {
    if (new_size > capacity()) {
        throw std::runtime_error("Cannot resize beyond capacity");
    }
    if (this->Message<ComplexVector<T>>::data()) {
        *reinterpret_cast<size_t*>(this->Message<ComplexVector<T>>::data()) = new_size;
    }
}

template<typename T>
ComplexVector<T>& ComplexVector<T>::operator=(std::initializer_list<value_type> values) {
    resize(values.size());
    std::copy(values.begin(), values.end(), begin());
    return *this;
}

template<typename T>
ComplexVector<T>& ComplexVector<T>::operator+=(const ComplexVector<T>& other) {
    size_t min_size = std::min(size(), other.size());
    for (size_t i = 0; i < min_size; ++i) {
        (*this)[i] += other[i];
    }
    return *this;
}

template<typename T>
ComplexVector<T>& ComplexVector<T>::operator-=(const ComplexVector<T>& other) {
    size_t min_size = std::min(size(), other.size());
    for (size_t i = 0; i < min_size; ++i) {
        (*this)[i] -= other[i];
    }
    return *this;
}

template<typename T>
ComplexVector<T>& ComplexVector<T>::operator*=(const value_type& scalar) {
    for (size_t i = 0; i < size(); ++i) {
        (*this)[i] *= scalar;
    }
    return *this;
}

template<typename T>
ComplexVector<T>& ComplexVector<T>::operator*=(const real_type& scalar) {
    for (size_t i = 0; i < size(); ++i) {
        (*this)[i] *= scalar;
    }
    return *this;
}

template<typename T>
void ComplexVector<T>::conjugate() {
    for (size_t i = 0; i < size(); ++i) {
        (*this)[i] = std::conj((*this)[i]);
    }
}

template<typename T>
std::vector<typename ComplexVector<T>::real_type> ComplexVector<T>::magnitude() const {
    std::vector<real_type> result;
    result.reserve(size());
    for (size_t i = 0; i < size(); ++i) {
        result.push_back(std::abs((*this)[i]));
    }
    return result;
}

template<typename T>
std::vector<typename ComplexVector<T>::real_type> ComplexVector<T>::phase() const {
    std::vector<real_type> result;
    result.reserve(size());
    for (size_t i = 0; i < size(); ++i) {
        result.push_back(std::arg((*this)[i]));
    }
    return result;
}

template<typename T>
void ComplexVector<T>::real_part(real_type* output) const {
    for (size_t i = 0; i < size(); ++i) {
        output[i] = (*this)[i].real();
    }
}

template<typename T>
void ComplexVector<T>::imag_part(real_type* output) const {
    for (size_t i = 0; i < size(); ++i) {
        output[i] = (*this)[i].imag();
    }
}

template<typename T>
void ComplexVector<T>::set_real_part(const real_type* input) {
    for (size_t i = 0; i < size(); ++i) {
        (*this)[i] = value_type(input[i], (*this)[i].imag());
    }
}

template<typename T>
void ComplexVector<T>::set_imag_part(const real_type* input) {
    for (size_t i = 0; i < size(); ++i) {
        (*this)[i] = value_type((*this)[i].real(), input[i]);
    }
}

template<typename T>
typename ComplexVector<T>::value_type ComplexVector<T>::dot_product(const ComplexVector<T>& other) const {
    value_type result(0, 0);
    size_t min_size = std::min(size(), other.size());
    for (size_t i = 0; i < min_size; ++i) {
        result += std::conj((*this)[i]) * other[i];
    }
    return result;
}

template<typename T>
typename ComplexVector<T>::real_type ComplexVector<T>::power() const {
    real_type total_power = 0;
    for (size_t i = 0; i < size(); ++i) {
        real_type magnitude = std::abs((*this)[i]);
        total_power += magnitude * magnitude;
    }
    return total_power;
}

// Explicit template instantiations for common types
template class ComplexVector<float>;
template class ComplexVector<double>;

} // namespace types
} // namespace psyne