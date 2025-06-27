#pragma once

#include "variant.hpp"
#include <span>
#include <iterator>
#include <stdexcept>

namespace psyne {

template<typename T>
class VariantView {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    
    VariantView() : header_(nullptr) {}
    
    explicit VariantView(VariantHdr* header) : header_(header) {
        validate();
    }
    
    explicit VariantView(const VariantHdr* header) 
        : header_(const_cast<VariantHdr*>(header)) {
        validate();
    }
    
    T* data() { 
        return header_ ? header_->as<T>() : nullptr; 
    }
    
    const T* data() const { 
        return header_ ? header_->as<T>() : nullptr; 
    }
    
    size_t size() const {
        if (!header_) return 0;
        
        if (is_scalar()) return 1;
        return header_->byteLen / sizeof(T);
    }
    
    bool empty() const { return size() == 0; }
    
    T& operator[](size_t idx) {
        return data()[idx];
    }
    
    const T& operator[](size_t idx) const {
        return data()[idx];
    }
    
    T& at(size_t idx) {
        if (idx >= size()) {
            throw std::out_of_range("VariantView index out of range");
        }
        return data()[idx];
    }
    
    const T& at(size_t idx) const {
        if (idx >= size()) {
            throw std::out_of_range("VariantView index out of range");
        }
        return data()[idx];
    }
    
    std::span<T> as_span() {
        return std::span<T>(data(), size());
    }
    
    std::span<const T> as_span() const {
        return std::span<const T>(data(), size());
    }
    
    T* begin() { return data(); }
    T* end() { return data() + size(); }
    const T* begin() const { return data(); }
    const T* end() const { return data() + size(); }
    
    bool is_scalar() const {
        if (!header_) return false;
        auto type = static_cast<VariantType>(header_->type);
        return type == VariantTraits<T>::type();
    }
    
    bool is_array() const {
        if (!header_) return false;
        auto type = static_cast<VariantType>(header_->type);
        return type == VariantTraits<T>::array_type();
    }
    
    bool is_gpu_buffer() const {
        return header_ && (header_->flags & static_cast<uint8_t>(VariantFlags::GpuBuffer));
    }
    
    bool is_readonly() const {
        return header_ && (header_->flags & static_cast<uint8_t>(VariantFlags::Readonly));
    }
    
    VariantHdr* header() { return header_; }
    const VariantHdr* header() const { return header_; }
    
private:
    void validate() {
        if (!header_) return;
        
        auto type = static_cast<VariantType>(header_->type);
        auto expected_scalar = VariantTraits<T>::type();
        auto expected_array = VariantTraits<T>::array_type();
        
        if (type != expected_scalar && type != expected_array) {
            throw std::runtime_error("Type mismatch in VariantView");
        }
    }
    
    VariantHdr* header_;
};

template<typename T>
class VariantArrayView : public VariantView<T> {
public:
    using Base = VariantView<T>;
    using Base::Base;
    
    explicit VariantArrayView(VariantHdr* header) : Base(header) {
        if (!this->is_array()) {
            throw std::runtime_error("VariantArrayView requires array type");
        }
    }
};

class VariantIterator {
public:
    VariantIterator(uint8_t* ptr) : ptr_(ptr) {}
    
    VariantHdr* operator*() {
        return reinterpret_cast<VariantHdr*>(ptr_);
    }
    
    VariantHdr* operator->() {
        return reinterpret_cast<VariantHdr*>(ptr_);
    }
    
    VariantIterator& operator++() {
        auto* header = reinterpret_cast<VariantHdr*>(ptr_);
        size_t total_size = sizeof(VariantHdr) + header->byteLen;
        total_size = (total_size + 7) & ~7;  // 8-byte align
        ptr_ += total_size;
        return *this;
    }
    
    bool operator==(const VariantIterator& other) const {
        return ptr_ == other.ptr_;
    }
    
    bool operator!=(const VariantIterator& other) const {
        return ptr_ != other.ptr_;
    }
    
private:
    uint8_t* ptr_;
};

class MessageView {
public:
    MessageView(void* data, size_t size) 
        : data_(static_cast<uint8_t*>(data)), size_(size) {}
    
    VariantIterator begin() {
        return VariantIterator(data_ + sizeof(uint32_t) + sizeof(uint32_t));
    }
    
    VariantIterator end() {
        return VariantIterator(data_ + size_);
    }
    
    VariantIterator begin() const {
        return VariantIterator(const_cast<uint8_t*>(data_ + sizeof(uint32_t) + sizeof(uint32_t)));
    }
    
    VariantIterator end() const {
        return VariantIterator(const_cast<uint8_t*>(data_ + size_));
    }
    
    template<typename T>
    VariantView<T> get(size_t index) {
        size_t current = 0;
        for (auto it = begin(); it != end(); ++it) {
            if (current == index) {
                return VariantView<T>(*it);
            }
            ++current;
        }
        return VariantView<T>();
    }
    
    size_t variant_count() const {
        size_t count = 0;
        for (auto it = begin(); it != end(); ++it) {
            ++count;
        }
        return count;
    }
    
private:
    uint8_t* data_;
    size_t size_;
};

}  // namespace psyne