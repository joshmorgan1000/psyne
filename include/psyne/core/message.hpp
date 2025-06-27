#pragma once

#include "variant.hpp"
#include "variant_view.hpp"
#include "../memory/ring_buffer.hpp"
#include <memory>
#include <span>
#include <initializer_list>
#include <cstring>
#include <any>
#include <cmath>
#include <functional>
#include <Eigen/Core>

namespace psyne {

// Base message class that all message types inherit from
template<typename Derived>
class Message {
public:
    // For outgoing messages - allocates in channel
    template<typename Channel>
    explicit Message(Channel& channel) 
        : mode_(Mode::Outgoing)
        , channel_(&channel) {
        size_t required_size = Derived::calculate_size();
        auto* rb = channel.ring_buffer();
        if (rb) {
            auto write_handle = rb->reserve(required_size);
            if (write_handle) {
                // Store both the handle and a commit function
                handle_ = *write_handle;
                commit_fn_ = [h = *write_handle]() mutable { h.commit(); };
                data_ = static_cast<uint8_t*>(write_handle->data);
                static_cast<Derived*>(this)->initialize_storage(data_);
            } else {
                // Allocation failed
                data_ = nullptr;
            }
        } else {
            // No ring buffer
            data_ = nullptr;
        }
    }
    
    // For incoming messages - view of existing data
    explicit Message(const void* data, size_t size)
        : mode_(Mode::Incoming)
        , data_(const_cast<uint8_t*>(static_cast<const uint8_t*>(data)))
        , size_(size) {
        static_cast<Derived*>(this)->initialize_view(data_);
    }
    
    ~Message() = default;
    
    // Move only
    Message(Message&&) = default;
    Message& operator=(Message&&) = default;
    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;
    
    bool is_valid() const { return data_ != nullptr; }
    
    void send() {
        if (mode_ == Mode::Outgoing && commit_fn_) {
            // Allow derived classes to finalize before sending
            static_cast<Derived*>(this)->before_send();
            commit_fn_();
        }
    }
    
    static constexpr uint32_t type() { return Derived::message_type; }
    
    // Default implementation - derived classes can override
    void before_send() {}
    
    // Copy message data to a destination buffer
    void copy_to(void* dest) const {
        // This is a simplified version - derived classes should implement proper serialization
        if (data_ && size_ > 0) {
            std::memcpy(dest, data_, size_);
        }
    }
    
protected:
    enum class Mode { Incoming, Outgoing };
    
    Mode mode_;
    uint8_t* data_ = nullptr;
    size_t size_ = 0;
    void* channel_ = nullptr;
    std::any handle_;
    std::function<void()> commit_fn_;
    
    template<typename T>
    T* write_at(size_t offset, T value) {
        if (!data_) return nullptr;
        T* ptr = reinterpret_cast<T*>(data_ + offset);
        *ptr = value;
        return ptr;
    }
    
    template<typename T>
    const T* read_at(size_t offset) const {
        if (!data_) return nullptr;
        return reinterpret_cast<const T*>(data_ + offset);
    }
};

// Pre-defined message type: FloatVector
class FloatVector : public Message<FloatVector> {
public:
    static constexpr uint32_t message_type = 1;
    static constexpr size_t max_elements = 1024;  // Default max 1024 floats
    
    using Message::Message;
    using EigenVector = Eigen::VectorXf;
    using EigenMap = Eigen::Map<EigenVector>;
    using ConstEigenMap = Eigen::Map<const EigenVector>;
    
    // Assignment from initializer list
    FloatVector& operator=(std::initializer_list<float> values) {
        if (!data_ || mode_ != Mode::Outgoing) return *this;
        
        size_t count = std::min(values.size(), capacity());
        size_t i = 0;
        for (float v : values) {
            (*this)[i++] = v;
            if (i >= count) break;
        }
        current_size_ = count;
        
        return *this;
    }
    
    // Array access
    float& operator[](size_t index) {
        return data_span_[index];
    }
    
    const float& operator[](size_t index) const {
        return data_span_[index];
    }
    
    // STL-like interface
    float* begin() { return data_span_.data(); }
    float* end() { return data_span_.data() + size(); }
    const float* begin() const { return data_span_.data(); }
    const float* end() const { return data_span_.data() + size(); }
    
    size_t size() const { 
        return current_size_;
    }
    
    size_t capacity() const {
        return capacity_;
    }
    
    // Resize within allocated capacity
    void resize(size_t new_size) {
        if (new_size <= capacity_) {
            current_size_ = new_size;
        }
    }
    
    // Eigen views - zero copy!
    EigenMap as_eigen() {
        return EigenMap(data_span_.data(), size());
    }
    
    ConstEigenMap as_eigen() const {
        return ConstEigenMap(data_span_.data(), size());
    }
    
    static constexpr size_t calculate_size() {
        // For single-type channels: just the data
        // For multi-type channels: add header space (handled by channel)
        return sizeof(float) * max_elements + sizeof(size_t);  // data + size field
    }
    
    // Override copy_to to include size information
    void copy_to(void* dest) const {
        if (!data_ || current_size_ == 0) return;
        
        // Copy size first
        *reinterpret_cast<size_t*>(dest) = current_size_;
        // Then copy data
        std::memcpy(reinterpret_cast<uint8_t*>(dest) + sizeof(size_t), 
                    data_span_.data(), current_size_ * sizeof(float));
    }
    
    // Called before sending to update the size field
    void before_send() {
        if (size_ptr_ && mode_ == Mode::Outgoing) {
            *size_ptr_ = current_size_;
        }
    }
    
private:
    friend class Message<FloatVector>;
    
    void initialize_storage(void* ptr) {
        if (!ptr) {
            data_span_ = std::span<float>();
            current_size_ = 0;
            capacity_ = 0;
            return;
        }
        
        // Store size at the beginning of the buffer
        size_ptr_ = reinterpret_cast<size_t*>(ptr);
        *size_ptr_ = 0;  // Initialize size to 0
        
        // Float data starts after the size field
        float* data_ptr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ptr) + sizeof(size_t));
        capacity_ = (calculate_size() - sizeof(size_t)) / sizeof(float);
        data_span_ = std::span<float>(data_ptr, capacity_);
        current_size_ = 0;
    }
    
    void initialize_view(void* ptr) {
        if (!ptr) {
            data_span_ = std::span<float>();
            current_size_ = 0;
            capacity_ = 0;
            return;
        }
        
        // Read size from the beginning of the buffer
        size_ptr_ = reinterpret_cast<size_t*>(ptr);
        current_size_ = *size_ptr_;
        
        // Float data starts after the size field
        float* data_ptr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ptr) + sizeof(size_t));
        capacity_ = (size_ - sizeof(size_t)) / sizeof(float);  // Use size_ from base class
        data_span_ = std::span<float>(data_ptr, std::min(current_size_, capacity_));
    }
    
    std::span<float> data_span_;
    size_t current_size_ = 0;
    size_t capacity_ = 0;
    size_t* size_ptr_ = nullptr;
};

// Pre-defined message type: DoubleMatrix
class DoubleMatrix : public Message<DoubleMatrix> {
public:
    static constexpr uint32_t message_type = 2;
    
    using Message::Message;
    using EigenMatrix = Eigen::MatrixXd;
    using EigenMap = Eigen::Map<EigenMatrix>;
    using ConstEigenMap = Eigen::Map<const EigenMatrix>;
    
    void set_dimensions(size_t rows, size_t cols) {
        if (!header_ || mode_ != Mode::Outgoing) return;
        
        rows_ = rows;
        cols_ = cols;
        header_->byteLen = rows * cols * sizeof(double);
    }
    
    double& at(size_t row, size_t col) {
        return data_span_[row * cols_ + col];
    }
    
    const double& at(size_t row, size_t col) const {
        return data_span_[row * cols_ + col];
    }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    // Eigen views - zero copy!
    EigenMap as_eigen() {
        return EigenMap(data_span_.data(), rows_, cols_);
    }
    
    ConstEigenMap as_eigen() const {
        return ConstEigenMap(data_span_.data(), rows_, cols_);
    }
    
    static constexpr size_t calculate_size() {
        return sizeof(VariantHdr) + sizeof(double) * 1024 * 1024;  // Max 1M doubles
    }
    
private:
    friend class Message<DoubleMatrix>;
    
    void initialize_storage(void* ptr) {
        header_ = reinterpret_cast<VariantHdr*>(ptr);
        header_->type = static_cast<uint8_t>(VariantType::Float64Array);
        header_->flags = 0;
        header_->reserved = 0;
        header_->byteLen = 0;
        
        double* data_ptr = reinterpret_cast<double*>(header_->data());
        size_t max_elements = (calculate_size() - sizeof(VariantHdr)) / sizeof(double);
        data_span_ = std::span<double>(data_ptr, max_elements);
    }
    
    void initialize_view(void* ptr) {
        header_ = reinterpret_cast<VariantHdr*>(ptr);
        double* data_ptr = reinterpret_cast<double*>(header_->data());
        size_t total_elements = header_->byteLen / sizeof(double);
        data_span_ = std::span<double>(data_ptr, total_elements);
        
        // Try to infer dimensions (assume square if possible)
        size_t sqrt_n = static_cast<size_t>(std::sqrt(total_elements));
        if (sqrt_n * sqrt_n == total_elements) {
            rows_ = cols_ = sqrt_n;
        } else {
            rows_ = total_elements;
            cols_ = 1;
        }
    }
    
    VariantHdr* header_ = nullptr;
    std::span<double> data_span_;
    size_t rows_ = 0;
    size_t cols_ = 0;
};

// Template for user-defined message types
template<uint32_t TypeID, typename... Fields>
class CustomMessage : public Message<CustomMessage<TypeID, Fields...>> {
public:
    static constexpr uint32_t message_type = TypeID;
    
    using Base = Message<CustomMessage<TypeID, Fields...>>;
    using Base::Base;
    
    template<size_t Index>
    auto& get() {
        static_assert(Index < sizeof...(Fields));
        return std::get<Index>(fields_);
    }
    
    template<size_t Index>
    const auto& get() const {
        static_assert(Index < sizeof...(Fields));
        return std::get<Index>(fields_);
    }
    
    static constexpr size_t calculate_size() {
        return calculate_size_impl<Fields...>();
    }
    
private:
    friend class Message<CustomMessage>;
    
    template<typename T, typename... Rest>
    static constexpr size_t calculate_size_impl() {
        size_t size = sizeof(VariantHdr) + sizeof(T);
        size = (size + 7) & ~7;
        if constexpr (sizeof...(Rest) > 0) {
            return size + calculate_size_impl<Rest...>();
        }
        return size;
    }
    
    void initialize_storage(void* ptr) {
        // Initialize each field's storage
        uint8_t* current = static_cast<uint8_t*>(ptr);
        initialize_fields<0, Fields...>(current);
    }
    
    void initialize_view(void* ptr) {
        // Initialize views of each field
        uint8_t* current = static_cast<uint8_t*>(ptr);
        view_fields<0, Fields...>(current);
    }
    
    template<size_t Index, typename T, typename... Rest>
    void initialize_fields(uint8_t*& ptr) {
        auto* header = reinterpret_cast<VariantHdr*>(ptr);
        header->type = static_cast<uint8_t>(VariantTraits<T>::type());
        header->flags = 0;
        header->reserved = 0;
        header->byteLen = sizeof(T);
        
        std::get<Index>(fields_) = reinterpret_cast<T*>(header->data());
        
        ptr += (sizeof(VariantHdr) + sizeof(T) + 7) & ~7;
        
        if constexpr (sizeof...(Rest) > 0) {
            initialize_fields<Index + 1, Rest...>(ptr);
        }
    }
    
    template<size_t Index, typename T, typename... Rest>
    void view_fields(uint8_t*& ptr) {
        auto* header = reinterpret_cast<VariantHdr*>(ptr);
        std::get<Index>(fields_) = reinterpret_cast<T*>(header->data());
        
        ptr += (sizeof(VariantHdr) + header->byteLen + 7) & ~7;
        
        if constexpr (sizeof...(Rest) > 0) {
            view_fields<Index + 1, Rest...>(ptr);
        }
    }
    
    std::tuple<Fields*...> fields_;
};

}  // namespace psyne