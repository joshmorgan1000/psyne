#pragma once

#include "variant.hpp"
#include "variant_view.hpp"
#include "../memory/ring_buffer.hpp"
#include <vector>
#include <cstring>

namespace psyne {

template<typename RingBuffer>
class MessageBuilder {
public:
    using WriteHandle = typename RingBuffer::WriteHandle;
    
    MessageBuilder(RingBuffer& buffer) : buffer_(buffer), handle_(std::nullopt) {}
    
    ~MessageBuilder() {
        if (handle_ && !committed_) {
            // If we haven't committed, we need to release the reservation
            // This would require adding an abandon() method to WriteHandle
        }
    }
    
    // Plan the message structure first
    MessageBuilder& plan_scalar(size_t count = 1) {
        size_t variant_size = count * (sizeof(VariantHdr) + sizeof(double)); // Max scalar size
        planned_size_ += align_up(variant_size);
        return *this;
    }
    
    template<typename T>
    MessageBuilder& plan_array(size_t element_count) {
        size_t variant_size = sizeof(VariantHdr) + sizeof(T) * element_count;
        planned_size_ += align_up(variant_size);
        return *this;
    }
    
    // Reserve space after planning
    bool reserve() {
        handle_ = buffer_.reserve(planned_size_);
        if (!handle_) return false;
        
        current_ptr_ = static_cast<uint8_t*>(handle_->data);
        return true;
    }
    
    // Direct write methods - no copying
    template<typename T>
    T* write_scalar(VariantFlags flags = VariantFlags::None) {
        if (!handle_) return nullptr;
        
        auto* header = reinterpret_cast<VariantHdr*>(current_ptr_);
        header->type = static_cast<uint8_t>(VariantTraits<T>::type());
        header->flags = static_cast<uint8_t>(flags);
        header->reserved = 0;
        header->byteLen = sizeof(T);
        
        T* data_ptr = reinterpret_cast<T*>(header->data());
        
        current_ptr_ += align_up(sizeof(VariantHdr) + sizeof(T));
        return data_ptr;
    }
    
    template<typename T>
    std::span<T> write_array(size_t count, VariantFlags flags = VariantFlags::None) {
        if (!handle_) return {};
        
        auto* header = reinterpret_cast<VariantHdr*>(current_ptr_);
        header->type = static_cast<uint8_t>(VariantTraits<T>::array_type());
        header->flags = static_cast<uint8_t>(flags);
        header->reserved = 0;
        header->byteLen = sizeof(T) * count;
        
        T* data_ptr = reinterpret_cast<T*>(header->data());
        
        current_ptr_ += align_up(sizeof(VariantHdr) + sizeof(T) * count);
        return std::span<T>(data_ptr, count);
    }
    
    // Get a VariantView for the data just written
    template<typename T>
    VariantView<T> get_last_view() {
        if (!handle_ || current_ptr_ == handle_->data) return VariantView<T>();
        
        // Go back to the last written header
        uint8_t* last_header_ptr = current_ptr_;
        
        // Find the previous header by scanning backwards
        while (last_header_ptr > static_cast<uint8_t*>(handle_->data)) {
            last_header_ptr -= 8; // Min alignment
            auto* test_header = reinterpret_cast<VariantHdr*>(last_header_ptr);
            size_t test_size = align_up(sizeof(VariantHdr) + test_header->byteLen);
            if (last_header_ptr + test_size == current_ptr_) {
                return VariantView<T>(test_header);
            }
        }
        
        return VariantView<T>();
    }
    
    void commit() {
        if (handle_) {
            handle_->commit();
            committed_ = true;
        }
    }
    
    // For cases where you need to write raw data
    void* write_raw(size_t size) {
        if (!handle_) return nullptr;
        
        void* ptr = current_ptr_;
        current_ptr_ += align_up(size);
        return ptr;
    }
    
private:
    static constexpr size_t align_up(size_t value) {
        return (value + 7) & ~7;
    }
    
    RingBuffer& buffer_;
    std::optional<WriteHandle> handle_;
    uint8_t* current_ptr_ = nullptr;
    size_t planned_size_ = 0;
    bool committed_ = false;
};

// Convenience function for simple cases
template<typename RingBuffer, typename Func>
bool build_message(RingBuffer& buffer, size_t size, Func&& builder_func) {
    auto handle = buffer.reserve(size);
    if (!handle) return false;
    
    builder_func(handle->data);
    handle->commit();
    return true;
}

}  // namespace psyne