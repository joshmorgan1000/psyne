#pragma once

#include "ring_buffer.hpp"
#include "dynamic_slab_allocator.hpp"
#include <memory>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <deque>
#include <atomic>

namespace psyne {

// Dynamic ring buffer that can resize based on usage patterns
template<typename ProducerType, typename ConsumerType>
class DynamicRingBuffer {
public:
    using BaseRingBuffer = RingBuffer<ProducerType, ConsumerType>;
    using Header = typename BaseRingBuffer::Header;
    using WriteHandle = typename BaseRingBuffer::WriteHandle;
    using ReadHandle = typename BaseRingBuffer::ReadHandle;
    
    struct Config {
        size_t initial_size;
        size_t min_size;
        size_t max_size;
        double resize_up_threshold;
        double resize_down_threshold;
        size_t resize_factor;
        std::chrono::seconds resize_check_interval;
        size_t high_water_mark_window;
        
        // Default constructor with default values
        Config()
            : initial_size(64 * 1024)               // 64KB initial
            , min_size(4 * 1024)                    // 4KB minimum
            , max_size(128 * 1024 * 1024)           // 128MB maximum
            , resize_up_threshold(0.9)              // Resize up when 90% full
            , resize_down_threshold(0.1)            // Resize down when <10% used
            , resize_factor(2)                      // Double/halve on resize
            , resize_check_interval(5)              // Check every 5 seconds
            , high_water_mark_window(1000) {}       // Track last 1000 operations
    };
    
    struct Stats {
        size_t current_size = 0;
        size_t peak_usage = 0;
        size_t total_writes = 0;
        size_t total_reads = 0;
        size_t resize_count = 0;
        size_t failed_reserves = 0;
        double average_usage = 0.0;
        std::chrono::steady_clock::time_point last_resize;
    };
    
    explicit DynamicRingBuffer(const Config& config = {})
        : config_(config)
        , buffer_(std::make_unique<BaseRingBuffer>(config.initial_size))
        , migrating_(false)
        , last_check_(std::chrono::steady_clock::now()) {
        stats_.current_size = config.initial_size;
        stats_.last_resize = last_check_;
    }
    
    // Reserve space, potentially triggering resize
    std::optional<WriteHandle> reserve(size_t size) {
        // Fast path - try current buffer
        auto handle = buffer_->reserve(size);
        if (handle) {
            update_usage_stats(size, true);
            return handle;
        }
        
        // Slow path - might need to resize
        std::lock_guard<std::mutex> lock(resize_mutex_);
        
        // Try again (another thread might have resized)
        handle = buffer_->reserve(size);
        if (handle) {
            update_usage_stats(size, true);
            return handle;
        }
        
        stats_.failed_reserves++;
        
        // Check if we should resize
        // Force resize if we have many failed reserves, regardless of time
        bool force_resize = stats_.failed_reserves > 5;
        if (force_resize || should_resize_up()) {
            resize_up();
            // Try once more with new buffer
            handle = buffer_->reserve(size);
            if (handle) {
                update_usage_stats(size, true);
                return handle;
            }
        }
        
        return std::nullopt;
    }
    
    // Read from buffer
    std::optional<ReadHandle> read() {
        auto handle = buffer_->read();
        if (handle) {
            update_usage_stats(handle->size, false);
        }
        
        // Periodic resize check
        check_resize_down();
        
        return handle;
    }
    
    bool empty() const {
        return buffer_->empty();
    }
    
    void* base() { return buffer_->base(); }
    size_t capacity() const { return buffer_->capacity(); }
    
    // Get current statistics
    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }
    
    // Force a resize check
    void check_resize() {
        std::lock_guard<std::mutex> lock(resize_mutex_);
        
        if (should_resize_up()) {
            resize_up();
        } else if (should_resize_down()) {
            resize_down();
        }
    }
    
private:
    void update_usage_stats(size_t size, bool is_write) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        
        if (is_write) {
            stats_.total_writes++;
            current_usage_ += size;
            
            // Update peak usage
            if (current_usage_ > stats_.peak_usage) {
                stats_.peak_usage = current_usage_;
            }
            
            // Update high water marks
            usage_history_.push_back(current_usage_);
            if (usage_history_.size() > config_.high_water_mark_window) {
                usage_history_.pop_front();
            }
        } else {
            stats_.total_reads++;
            current_usage_ = (current_usage_ > size) ? current_usage_ - size : 0;
        }
        
        // Calculate average usage
        if (!usage_history_.empty()) {
            size_t sum = 0;
            for (size_t usage : usage_history_) {
                sum += usage;
            }
            stats_.average_usage = static_cast<double>(sum) / 
                                  static_cast<double>(usage_history_.size() * stats_.current_size);
        }
    }
    
    bool should_resize_up() const {
        // Don't resize too frequently
        auto now = std::chrono::steady_clock::now();
        if (now - stats_.last_resize < config_.resize_check_interval) {
            return false;
        }
        
        // Check if we're at max size
        if (stats_.current_size >= config_.max_size) {
            return false;
        }
        
        // Check usage threshold
        double usage_ratio = static_cast<double>(current_usage_) / 
                           static_cast<double>(stats_.current_size);
        
        return usage_ratio > config_.resize_up_threshold ||
               stats_.failed_reserves > 10;  // Many failed reserves
    }
    
    bool should_resize_down() const {
        // Don't resize too frequently
        auto now = std::chrono::steady_clock::now();
        if (now - stats_.last_resize < config_.resize_check_interval) {
            return false;
        }
        
        // Check if we're at min size
        if (stats_.current_size <= config_.min_size) {
            return false;
        }
        
        // Check average usage over time
        return stats_.average_usage < config_.resize_down_threshold;
    }
    
    void resize_up() {
        size_t new_size = std::min(
            stats_.current_size * config_.resize_factor,
            config_.max_size
        );
        
        if (new_size > stats_.current_size) {
            perform_resize(new_size);
        }
    }
    
    void resize_down() {
        size_t new_size = std::max(
            stats_.current_size / config_.resize_factor,
            config_.min_size
        );
        
        if (new_size < stats_.current_size) {
            perform_resize(new_size);
        }
    }
    
    void perform_resize(size_t new_size) {
        // Create new buffer
        auto new_buffer = std::make_unique<BaseRingBuffer>(new_size);
        
        // Mark as migrating
        migrating_.store(true);
        
        // Copy existing data
        // Note: This is the only place where we violate zero-copy,
        // but it's necessary for resizing and happens infrequently
        while (!buffer_->empty()) {
            auto read_handle = buffer_->read();
            if (!read_handle) break;
            
            auto write_handle = new_buffer->reserve(read_handle->size);
            if (!write_handle) {
                // New buffer too small? This shouldn't happen
                migrating_.store(false);
                return;
            }
            
            // Copy data
            std::memcpy(write_handle->data, read_handle->data, read_handle->size);
            write_handle->commit();
        }
        
        // Swap buffers
        buffer_ = std::move(new_buffer);
        
        // Update stats
        stats_.current_size = new_size;
        stats_.resize_count++;
        stats_.last_resize = std::chrono::steady_clock::now();
        stats_.failed_reserves = 0;  // Reset counter
        
        migrating_.store(false);
    }
    
    void check_resize_down() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_check_ > config_.resize_check_interval) {
            last_check_ = now;
            
            std::lock_guard<std::mutex> lock(resize_mutex_);
            if (should_resize_down()) {
                resize_down();
            }
        }
    }
    
    Config config_;
    std::unique_ptr<BaseRingBuffer> buffer_;
    std::atomic<bool> migrating_;
    
    mutable std::mutex resize_mutex_;
    mutable std::mutex stats_mutex_;
    
    Stats stats_;
    size_t current_usage_ = 0;
    std::deque<size_t> usage_history_;
    std::chrono::steady_clock::time_point last_check_;
};

// Type aliases for common configurations
template<typename P, typename C>
using DynamicRingBufferPtr = std::unique_ptr<DynamicRingBuffer<P, C>>;

using DynamicSPSCRingBuffer = DynamicRingBuffer<SingleProducer, SingleConsumer>;
using DynamicSPMCRingBuffer = DynamicRingBuffer<SingleProducer, MultiConsumer>;
using DynamicMPSCRingBuffer = DynamicRingBuffer<MultiProducer, SingleConsumer>;
using DynamicMPMCRingBuffer = DynamicRingBuffer<MultiProducer, MultiConsumer>;

}  // namespace psyne