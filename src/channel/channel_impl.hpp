#pragma once

#include <psyne/psyne.hpp>
#include "../memory/ring_buffer_impl.hpp"
#include <memory>
#include <string>
#include <atomic>

namespace psyne {
namespace detail {

// Base implementation class
class ChannelImpl {
public:
    ChannelImpl(const std::string& uri, size_t buffer_size, 
                ChannelMode mode, ChannelType type)
        : uri_(uri)
        , mode_(mode)
        , type_(type)
        , stopped_(false) {}
        
    virtual ~ChannelImpl() = default;
    
    // Core operations
    virtual void* reserve_space(size_t size) = 0;
    virtual void commit_message(void* handle) = 0;
    virtual void* receive_message(size_t& size, uint32_t& type) = 0;
    virtual void release_message(void* handle) = 0;
    
    // Control
    void stop() { stopped_.store(true); }
    bool is_stopped() const { return stopped_.load(); }
    
    // Properties
    const std::string& uri() const { return uri_; }
    ChannelMode mode() const { return mode_; }
    ChannelType type() const { return type_; }
    
protected:
    std::string uri_;
    ChannelMode mode_;
    ChannelType type_;
    std::atomic<bool> stopped_;
};

// Template implementation for different ring buffer types
template<typename RingBufferType>
class ChannelImplT : public ChannelImpl {
public:
    ChannelImplT(const std::string& uri, size_t buffer_size,
                 ChannelMode mode, ChannelType type)
        : ChannelImpl(uri, buffer_size, mode, type)
        , ring_buffer_(buffer_size) {}
        
    void* reserve_space(size_t size) override {
        auto handle = ring_buffer_.reserve(size);
        if (handle) {
            // Commit the handle to set the len field
            handle->commit();
            return handle->header;
        }
        return nullptr;
    }
    
    void commit_message(void* handle) override {
        // Nothing to do - already committed in reserve_space
    }
    
    void* receive_message(size_t& size, uint32_t& type) override {
        auto handle = ring_buffer_.read();
        if (handle) {
            size = handle->size;
            // For single-type channels, type is always 1 (FloatVector)
            // For multi-type, we'd need to read from message header
            type = (type_ == ChannelType::SingleType) ? 1 : 0;
            return const_cast<void*>(handle->data);
        }
        return nullptr;
    }
    
    void release_message(void* handle) override {
        // Release read handle
    }
    
private:
    RingBufferType ring_buffer_;
};

} // namespace detail
} // namespace psyne