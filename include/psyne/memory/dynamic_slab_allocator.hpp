#pragma once

#include "slab_allocator.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <algorithm>
#include <chrono>

namespace psyne {

// Dynamic slab allocator that can grow based on usage
class DynamicSlabAllocator {
public:
    struct Config {
        size_t initial_slab_size;
        size_t max_slab_size;
        size_t growth_factor;
        double growth_threshold;
        double shrink_threshold;
        std::chrono::seconds cleanup_interval;
        
        // Default constructor with default values
        Config() 
            : initial_slab_size(1024 * 1024)        // 1MB initial
            , max_slab_size(128 * 1024 * 1024)      // 128MB max per slab
            , growth_factor(2)                       // Double size on growth
            , growth_threshold(0.75)                 // Grow when 75% full
            , shrink_threshold(0.25)                 // Consider shrinking when <25% used
            , cleanup_interval(60) {}                // Check for cleanup every minute
    };
    
    struct Stats {
        size_t total_allocated = 0;
        size_t total_used = 0;
        size_t num_slabs = 0;
        size_t num_allocations = 0;
        size_t num_growths = 0;
        double usage_ratio = 0.0;
    };
    
    explicit DynamicSlabAllocator(const Config& config = {})
        : config_(config)
        , current_slab_index_(0)
        , last_cleanup_(std::chrono::steady_clock::now()) {
        // Start with initial slab
        add_slab(config_.initial_slab_size);
    }
    
    ~DynamicSlabAllocator() = default;
    
    // Allocate memory, growing if necessary
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Try current slab first
        if (current_slab_index_ < slabs_.size()) {
            void* ptr = slabs_[current_slab_index_]->allocate(size);
            if (ptr) {
                stats_.num_allocations++;
                update_stats();
                return ptr;
            }
        }
        
        // Try other slabs
        for (size_t i = 0; i < slabs_.size(); ++i) {
            if (i == current_slab_index_) continue;
            void* ptr = slabs_[i]->allocate(size);
            if (ptr) {
                current_slab_index_ = i;
                stats_.num_allocations++;
                update_stats();
                return ptr;
            }
        }
        
        // Need to grow
        if (should_grow()) {
            grow();
            // Try again with new slab
            if (current_slab_index_ < slabs_.size()) {
                void* ptr = slabs_[current_slab_index_]->allocate(size);
                if (ptr) {
                    stats_.num_allocations++;
                    update_stats();
                    return ptr;
                }
            }
        }
        
        return nullptr;  // Out of memory
    }
    
    // Get current statistics
    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }
    
    // Force a cleanup check
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        perform_cleanup();
    }
    
    // Get recommended ring buffer size based on usage patterns
    size_t get_recommended_buffer_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Base recommendation on current usage
        if (stats_.usage_ratio > 0.9) {
            // Heavy usage - recommend larger buffers
            return std::min(config_.max_slab_size / 4, size_t(16 * 1024 * 1024));
        } else if (stats_.usage_ratio > 0.5) {
            // Moderate usage
            return std::min(config_.max_slab_size / 8, size_t(4 * 1024 * 1024));
        } else {
            // Light usage
            return std::min(config_.initial_slab_size, size_t(1024 * 1024));
        }
    }
    
private:
    void add_slab(size_t size) {
        slabs_.push_back(std::make_unique<SlabAllocator>(size));
        current_slab_index_ = slabs_.size() - 1;
        stats_.num_slabs = slabs_.size();
        stats_.total_allocated += size;
    }
    
    bool should_grow() const {
        // Check if we should add a new slab
        if (slabs_.empty()) return true;
        
        // Check overall usage
        return stats_.usage_ratio > config_.growth_threshold;
    }
    
    void grow() {
        if (slabs_.empty()) {
            add_slab(config_.initial_slab_size);
            return;
        }
        
        // Calculate new slab size
        size_t last_size = slabs_.back()->capacity();
        size_t new_size = std::min(
            last_size * config_.growth_factor,
            config_.max_slab_size
        );
        
        add_slab(new_size);
        stats_.num_growths++;
    }
    
    void update_stats() {
        stats_.total_used = 0;
        stats_.total_allocated = 0;
        
        for (const auto& slab : slabs_) {
            stats_.total_allocated += slab->capacity();
            stats_.total_used += (slab->capacity() - slab->available());
        }
        
        if (stats_.total_allocated > 0) {
            stats_.usage_ratio = static_cast<double>(stats_.total_used) / 
                                static_cast<double>(stats_.total_allocated);
        }
        
        // Check if we should do cleanup
        auto now = std::chrono::steady_clock::now();
        if (now - last_cleanup_ > config_.cleanup_interval) {
            perform_cleanup();
            last_cleanup_ = now;
        }
    }
    
    void perform_cleanup() {
        // Remove unused slabs (keep at least one)
        if (slabs_.size() <= 1) return;
        
        // Find completely unused slabs
        std::vector<size_t> unused_indices;
        for (size_t i = 0; i < slabs_.size(); ++i) {
            if (slabs_[i]->available() == slabs_[i]->capacity()) {
                unused_indices.push_back(i);
            }
        }
        
        // Remove half of unused slabs to avoid thrashing
        size_t to_remove = unused_indices.size() / 2;
        if (to_remove > 0 && slabs_.size() - to_remove >= 1) {
            // Sort by index descending to remove from end first
            std::sort(unused_indices.rbegin(), unused_indices.rend());
            
            for (size_t i = 0; i < to_remove; ++i) {
                size_t idx = unused_indices[i];
                slabs_.erase(slabs_.begin() + idx);
                
                // Adjust current slab index
                if (current_slab_index_ >= idx && current_slab_index_ > 0) {
                    current_slab_index_--;
                }
            }
            
            stats_.num_slabs = slabs_.size();
        }
    }
    
    Config config_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<SlabAllocator>> slabs_;
    size_t current_slab_index_;
    Stats stats_;
    std::chrono::steady_clock::time_point last_cleanup_;
};

}  // namespace psyne