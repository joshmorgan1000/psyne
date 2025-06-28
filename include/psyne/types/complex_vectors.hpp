#pragma once

#include "../psyne.hpp"
#include <complex>
#include <array>

namespace psyne {
namespace types {

// Complex number vector for signal processing and scientific computing
template<typename T>
class ComplexVector : public Message<ComplexVector<T>> {
public:
    using value_type = std::complex<T>;
    using real_type = T;
    
    static constexpr uint32_t message_type = 27; // Will need different IDs for float/double
    
    using Message<ComplexVector<T>>::Message;
    
    // Calculate required size (default capacity)
    static size_t calculate_size() {
        return 1024; // Default size for dynamic messages
    }
    
    // Element access
    value_type& operator[](size_t index);
    const value_type& operator[](size_t index) const;
    value_type& at(size_t index);
    const value_type& at(size_t index) const;
    
    // STL interface
    value_type* begin();
    value_type* end();
    const value_type* begin() const;
    const value_type* end() const;
    
    // Raw data access
    value_type* data();
    const value_type* data() const;
    
    // Size management
    size_t size() const;
    size_t capacity() const;
    void resize(size_t new_size);
    
    // Assignment from initializer list
    ComplexVector& operator=(std::initializer_list<value_type> values);
    
    // In-place operations
    ComplexVector& operator+=(const ComplexVector& other);
    ComplexVector& operator-=(const ComplexVector& other);
    ComplexVector& operator*=(const value_type& scalar);
    ComplexVector& operator*=(const real_type& scalar);
    
    // Complex-specific operations
    void conjugate(); // In-place complex conjugate
    std::vector<real_type> magnitude() const; // Magnitude of each element
    std::vector<real_type> phase() const; // Phase of each element
    
    // Separate real and imaginary parts
    void real_part(real_type* output) const;
    void imag_part(real_type* output) const;
    void set_real_part(const real_type* input);
    void set_imag_part(const real_type* input);
    
    // Signal processing operations
    value_type dot_product(const ComplexVector& other) const;
    real_type power() const; // Sum of |z|^2 for all elements
    
    void initialize();
    void before_send() override {}
};

// Specialized versions for common types
using ComplexVectorF = ComplexVector<float>;
using ComplexVectorD = ComplexVector<double>;

// Separate message type IDs for template specializations
template<>
constexpr uint32_t ComplexVector<float>::message_type = 27;

template<>
constexpr uint32_t ComplexVector<double>::message_type = 28;

} // namespace types
} // namespace psyne