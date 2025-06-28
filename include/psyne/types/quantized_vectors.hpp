#pragma once

#include "../psyne.hpp"
#include <array>

namespace psyne {
namespace types {

// Quantized 8-bit signed integer vector for ML inference
class Int8Vector : public Message<Int8Vector> {
public:
    static constexpr uint32_t message_type = 25;
    
    using Message<Int8Vector>::Message;
    
    // Calculate required size (default capacity)
    static size_t calculate_size() {
        return 1024; // Default size for dynamic messages
    }
    
    // Element access
    int8_t& operator[](size_t index);
    const int8_t& operator[](size_t index) const;
    int8_t& at(size_t index);
    const int8_t& at(size_t index) const;
    
    // STL interface
    int8_t* begin();
    int8_t* end();
    const int8_t* begin() const;
    const int8_t* end() const;
    
    // Raw data access
    int8_t* data();
    const int8_t* data() const;
    
    // Size management
    size_t size() const;
    size_t capacity() const;
    void resize(size_t new_size);
    
    // Assignment from initializer list
    Int8Vector& operator=(std::initializer_list<int8_t> values);
    
    // In-place operations
    Int8Vector& operator+=(const Int8Vector& other);
    Int8Vector& operator-=(const Int8Vector& other);
    Int8Vector& operator*=(int8_t scalar);
    
    // Quantization utilities
    void quantize_from_float(const float* float_data, size_t count, float scale, int8_t zero_point);
    void dequantize_to_float(float* float_data, float scale, int8_t zero_point) const;
    
    // Statistics
    int8_t min_value() const;
    int8_t max_value() const;
    int32_t sum() const; // Use int32 to avoid overflow
    
    void initialize();
    void before_send() override {}
};

// Quantized 8-bit unsigned integer vector for ML inference
class UInt8Vector : public Message<UInt8Vector> {
public:
    static constexpr uint32_t message_type = 26;
    
    using Message<UInt8Vector>::Message;
    
    static size_t calculate_size() {
        return 1024; // Default size for dynamic messages
    }
    
    // Element access
    uint8_t& operator[](size_t index);
    const uint8_t& operator[](size_t index) const;
    uint8_t& at(size_t index);
    const uint8_t& at(size_t index) const;
    
    // STL interface
    uint8_t* begin();
    uint8_t* end();
    const uint8_t* begin() const;
    const uint8_t* end() const;
    
    // Raw data access
    uint8_t* data();
    const uint8_t* data() const;
    
    // Size management
    size_t size() const;
    size_t capacity() const;
    void resize(size_t new_size);
    
    // Assignment from initializer list
    UInt8Vector& operator=(std::initializer_list<uint8_t> values);
    
    // In-place operations
    UInt8Vector& operator+=(const UInt8Vector& other);
    UInt8Vector& operator-=(const UInt8Vector& other);
    UInt8Vector& operator*=(uint8_t scalar);
    
    // Quantization utilities
    void quantize_from_float(const float* float_data, size_t count, float scale, uint8_t zero_point);
    void dequantize_to_float(float* float_data, float scale, uint8_t zero_point) const;
    
    // Statistics
    uint8_t min_value() const;
    uint8_t max_value() const;
    uint32_t sum() const; // Use uint32 to avoid overflow
    
    void initialize();
    void before_send() override {}
};

} // namespace types
} // namespace psyne