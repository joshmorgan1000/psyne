#include "../../include/psyne/types/quantized_vectors.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace psyne {
namespace types {

// Int8Vector implementation
void Int8Vector::initialize() {
    // Initialize size to 0
    if (this->data()) {
        *reinterpret_cast<size_t*>(this->data()) = 0;
    }
}

int8_t& Int8Vector::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("Int8Vector index out of range");
    }
    return begin()[index];
}

const int8_t& Int8Vector::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Int8Vector index out of range");
    }
    return begin()[index];
}

int8_t& Int8Vector::at(size_t index) {
    return (*this)[index];
}

const int8_t& Int8Vector::at(size_t index) const {
    return (*this)[index];
}

int8_t* Int8Vector::begin() {
    return reinterpret_cast<int8_t*>(this->Message<Int8Vector>::data() + sizeof(size_t));
}

int8_t* Int8Vector::end() {
    return begin() + size();
}

const int8_t* Int8Vector::begin() const {
    return reinterpret_cast<const int8_t*>(this->Message<Int8Vector>::data() + sizeof(size_t));
}

const int8_t* Int8Vector::end() const {
    return begin() + size();
}

int8_t* Int8Vector::data() {
    return begin();
}

const int8_t* Int8Vector::data() const {
    return begin();
}

size_t Int8Vector::size() const {
    if (!this->Message<Int8Vector>::data()) return 0;
    size_t stored_size = *reinterpret_cast<const size_t*>(this->Message<Int8Vector>::data());
    if (stored_size > capacity()) {
        return 0;  // Return 0 for invalid size
    }
    return stored_size;
}

size_t Int8Vector::capacity() const {
    if (!this->Message<Int8Vector>::data()) return 0;
    return (this->Message<Int8Vector>::size() - sizeof(size_t)) / sizeof(int8_t);
}

void Int8Vector::resize(size_t new_size) {
    if (new_size > capacity()) {
        throw std::runtime_error("Cannot resize beyond capacity");
    }
    if (this->Message<Int8Vector>::data()) {
        *reinterpret_cast<size_t*>(this->Message<Int8Vector>::data()) = new_size;
    }
}

Int8Vector& Int8Vector::operator=(std::initializer_list<int8_t> values) {
    resize(values.size());
    std::copy(values.begin(), values.end(), begin());
    return *this;
}

Int8Vector& Int8Vector::operator+=(const Int8Vector& other) {
    size_t min_size = std::min(size(), other.size());
    for (size_t i = 0; i < min_size; ++i) {
        (*this)[i] += other[i];
    }
    return *this;
}

Int8Vector& Int8Vector::operator-=(const Int8Vector& other) {
    size_t min_size = std::min(size(), other.size());
    for (size_t i = 0; i < min_size; ++i) {
        (*this)[i] -= other[i];
    }
    return *this;
}

Int8Vector& Int8Vector::operator*=(int8_t scalar) {
    for (size_t i = 0; i < size(); ++i) {
        (*this)[i] *= scalar;
    }
    return *this;
}

void Int8Vector::quantize_from_float(const float* float_data, size_t count, float scale, int8_t zero_point) {
    resize(count);
    for (size_t i = 0; i < count; ++i) {
        // Quantize: q = clamp(round(x/scale) + zero_point, -128, 127)
        int32_t quantized = static_cast<int32_t>(std::round(float_data[i] / scale)) + zero_point;
        (*this)[i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
    }
}

void Int8Vector::dequantize_to_float(float* float_data, float scale, int8_t zero_point) const {
    for (size_t i = 0; i < size(); ++i) {
        // Dequantize: x = scale * (q - zero_point)
        float_data[i] = scale * ((*this)[i] - zero_point);
    }
}

int8_t Int8Vector::min_value() const {
    if (size() == 0) return 0;
    return *std::min_element(begin(), end());
}

int8_t Int8Vector::max_value() const {
    if (size() == 0) return 0;
    return *std::max_element(begin(), end());
}

int32_t Int8Vector::sum() const {
    int32_t total = 0;
    for (size_t i = 0; i < size(); ++i) {
        total += (*this)[i];
    }
    return total;
}

// UInt8Vector implementation
void UInt8Vector::initialize() {
    // Initialize size to 0
    if (this->data()) {
        *reinterpret_cast<size_t*>(this->data()) = 0;
    }
}

uint8_t& UInt8Vector::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("UInt8Vector index out of range");
    }
    return begin()[index];
}

const uint8_t& UInt8Vector::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("UInt8Vector index out of range");
    }
    return begin()[index];
}

uint8_t& UInt8Vector::at(size_t index) {
    return (*this)[index];
}

const uint8_t& UInt8Vector::at(size_t index) const {
    return (*this)[index];
}

uint8_t* UInt8Vector::begin() {
    return reinterpret_cast<uint8_t*>(this->Message<UInt8Vector>::data() + sizeof(size_t));
}

uint8_t* UInt8Vector::end() {
    return begin() + size();
}

const uint8_t* UInt8Vector::begin() const {
    return reinterpret_cast<const uint8_t*>(this->Message<UInt8Vector>::data() + sizeof(size_t));
}

const uint8_t* UInt8Vector::end() const {
    return begin() + size();
}

uint8_t* UInt8Vector::data() {
    return begin();
}

const uint8_t* UInt8Vector::data() const {
    return begin();
}

size_t UInt8Vector::size() const {
    if (!this->Message<UInt8Vector>::data()) return 0;
    size_t stored_size = *reinterpret_cast<const size_t*>(this->Message<UInt8Vector>::data());
    if (stored_size > capacity()) {
        return 0;  // Return 0 for invalid size
    }
    return stored_size;
}

size_t UInt8Vector::capacity() const {
    if (!this->Message<UInt8Vector>::data()) return 0;
    return (this->Message<UInt8Vector>::size() - sizeof(size_t)) / sizeof(uint8_t);
}

void UInt8Vector::resize(size_t new_size) {
    if (new_size > capacity()) {
        throw std::runtime_error("Cannot resize beyond capacity");
    }
    if (this->Message<UInt8Vector>::data()) {
        *reinterpret_cast<size_t*>(this->Message<UInt8Vector>::data()) = new_size;
    }
}

UInt8Vector& UInt8Vector::operator=(std::initializer_list<uint8_t> values) {
    resize(values.size());
    std::copy(values.begin(), values.end(), begin());
    return *this;
}

UInt8Vector& UInt8Vector::operator+=(const UInt8Vector& other) {
    size_t min_size = std::min(size(), other.size());
    for (size_t i = 0; i < min_size; ++i) {
        (*this)[i] += other[i];
    }
    return *this;
}

UInt8Vector& UInt8Vector::operator-=(const UInt8Vector& other) {
    size_t min_size = std::min(size(), other.size());
    for (size_t i = 0; i < min_size; ++i) {
        (*this)[i] -= other[i];
    }
    return *this;
}

UInt8Vector& UInt8Vector::operator*=(uint8_t scalar) {
    for (size_t i = 0; i < size(); ++i) {
        (*this)[i] *= scalar;
    }
    return *this;
}

void UInt8Vector::quantize_from_float(const float* float_data, size_t count, float scale, uint8_t zero_point) {
    resize(count);
    for (size_t i = 0; i < count; ++i) {
        // Quantize: q = clamp(round(x/scale) + zero_point, 0, 255)
        int32_t quantized = static_cast<int32_t>(std::round(float_data[i] / scale)) + zero_point;
        (*this)[i] = static_cast<uint8_t>(std::clamp(quantized, 0, 255));
    }
}

void UInt8Vector::dequantize_to_float(float* float_data, float scale, uint8_t zero_point) const {
    for (size_t i = 0; i < size(); ++i) {
        // Dequantize: x = scale * (q - zero_point)
        float_data[i] = scale * ((*this)[i] - zero_point);
    }
}

uint8_t UInt8Vector::min_value() const {
    if (size() == 0) return 0;
    return *std::min_element(begin(), end());
}

uint8_t UInt8Vector::max_value() const {
    if (size() == 0) return 0;
    return *std::max_element(begin(), end());
}

uint32_t UInt8Vector::sum() const {
    uint32_t total = 0;
    for (size_t i = 0; i < size(); ++i) {
        total += (*this)[i];
    }
    return total;
}

} // namespace types
} // namespace psyne