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
                ChannelMode mode, ChannelType type, bool enable_metrics = false)
        : uri_(uri)
        , mode_(mode)
        , type_(type)
        , stopped_(false)
        , metrics_enabled_(enable_metrics) {}
        
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
    
    // Metrics support - default implementation returns empty metrics
    virtual bool has_metrics() const { return metrics_enabled_; }
    virtual debug::ChannelMetrics get_metrics() const {
        return debug::ChannelMetrics{};
    }
    virtual void reset_metrics() {
        // Default: no-op
    }
    
protected:
    std::string uri_;
    ChannelMode mode_;
    ChannelType type_;
    std::atomic<bool> stopped_;
    bool metrics_enabled_;
};

// Template implementation for different ring buffer types
template<typename RingBufferType>
class ChannelImplT : public ChannelImpl {
public:
    ChannelImplT(const std::string& uri, size_t buffer_size,
                 ChannelMode mode, ChannelType type, bool enable_metrics = false)
        : ChannelImpl(uri, buffer_size, mode, type, enable_metrics)
        , ring_buffer_(buffer_size) {
        if (metrics_enabled_) {
            if (mode == ChannelMode::SPSC) {
                metrics_ = std::make_unique<debug::ChannelMetrics>();
            } else {
                atomic_metrics_ = std::make_unique<debug::AtomicChannelMetrics>();
            }
        }
    }
        
    void* reserve_space(size_t size) override {
        auto handle = ring_buffer_.reserve(size);
        if (handle) {
            // Commit the handle to set the len field
            handle->commit();
            
            // Update metrics
            if (metrics_enabled_) {
                if (metrics_) {
                    metrics_->messages_sent++;
                    metrics_->bytes_sent += size;
                } else if (atomic_metrics_) {
                    atomic_metrics_->messages_sent.fetch_add(1, std::memory_order_relaxed);
                    atomic_metrics_->bytes_sent.fetch_add(size, std::memory_order_relaxed);
                }
            }
            
            return handle->header;
        } else {
            // Track blocking
            if (metrics_enabled_) {
                if (metrics_) {
                    metrics_->send_blocks++;
                } else if (atomic_metrics_) {
                    atomic_metrics_->send_blocks.fetch_add(1, std::memory_order_relaxed);
                }
            }
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
            
            // Update metrics
            if (metrics_enabled_) {
                if (metrics_) {
                    metrics_->messages_received++;
                    metrics_->bytes_received += size;
                } else if (atomic_metrics_) {
                    atomic_metrics_->messages_received.fetch_add(1, std::memory_order_relaxed);
                    atomic_metrics_->bytes_received.fetch_add(size, std::memory_order_relaxed);
                }
            }
            
            return const_cast<void*>(handle->data);
        } else {
            // Track blocking
            if (metrics_enabled_) {
                if (metrics_) {
                    metrics_->receive_blocks++;
                } else if (atomic_metrics_) {
                    atomic_metrics_->receive_blocks.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        return nullptr;
    }
    
    void release_message(void* handle) override {
        // Release read handle
    }
    
    debug::ChannelMetrics get_metrics() const override {
        if (!metrics_enabled_) {
            return debug::ChannelMetrics{};
        }
        
        if (metrics_) {
            return *metrics_;
        } else if (atomic_metrics_) {
            return atomic_metrics_->current();
        }
        
        return debug::ChannelMetrics{};
    }
    
    void reset_metrics() override {
        if (!metrics_enabled_) return;
        
        if (metrics_) {
            *metrics_ = debug::ChannelMetrics{};
        } else if (atomic_metrics_) {
            atomic_metrics_->messages_sent.store(0, std::memory_order_relaxed);
            atomic_metrics_->bytes_sent.store(0, std::memory_order_relaxed);
            atomic_metrics_->messages_received.store(0, std::memory_order_relaxed);
            atomic_metrics_->bytes_received.store(0, std::memory_order_relaxed);
            atomic_metrics_->send_blocks.store(0, std::memory_order_relaxed);
            atomic_metrics_->receive_blocks.store(0, std::memory_order_relaxed);
        }
    }
    
private:
    RingBufferType ring_buffer_;
    
    // Metrics - one or the other based on channel mode
    std::unique_ptr<debug::ChannelMetrics> metrics_;          // SPSC only
    std::unique_ptr<debug::AtomicChannelMetrics> atomic_metrics_;  // SPMC/MPSC/MPMC
};

} // namespace detail
} // namespace psyne