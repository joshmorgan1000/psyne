/**
 * @file dynamic_slab_allocator.hpp
 * @brief Dynamic slab allocator for adaptive memory management
 * 
 * This allocator manages slabs that grow based on usage patterns,
 * optimizing for both small frequent messages and large bulk transfers.
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <mutex>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <chrono>

namespace psyne {
namespace memory {

/**
 * @brief Configuration for dynamic slab allocator
 */
struct DynamicSlabConfig {
    size_t initial_slab_size = 64 * 1024 * 1024;    ///< Initial slab size (64MB default)
    size_t max_slab_size = 1024 * 1024 * 1024;      ///< Maximum slab size (1GB default)
    size_t min_slab_size = 1024 * 1024;             ///< Minimum slab size (1MB default)
    double growth_factor = 2.0;                      ///< Slab size growth multiplier
    double high_water_mark = 0.75;                   ///< Trigger growth at 75% usage
    double low_water_mark = 0.25;                    ///< Consider shrinking below 25% usage
    size_t max_slabs = 16;                           ///< Maximum number of slabs
    bool enable_shrinking = true;                    ///< Allow slabs to shrink
    std::chrono::seconds shrink_delay{60};           ///< Delay before shrinking
};

/**
 * @brief Statistics for dynamic slab allocator
 */
struct DynamicSlabStats {
    size_t num_slabs = 0;                            ///< Current number of slabs
    size_t total_capacity = 0;                       ///< Total allocated memory
    size_t total_used = 0;                           ///< Currently used memory
    size_t largest_slab_size = 0;                    ///< Size of largest slab
    size_t smallest_slab_size = 0;                   ///< Size of smallest slab
    double usage_ratio = 0.0;                        ///< Overall usage ratio
    uint64_t allocations = 0;                        ///< Total allocation count
    uint64_t deallocations = 0;                      ///< Total deallocation count
    uint64_t slab_growths = 0;                       ///< Number of slab growths
    uint64_t slab_shrinks = 0;                       ///< Number of slab shrinks
};

/**
 * @brief Individual slab in the allocator
 */
class Slab {
public:
    Slab(size_t size);
    ~Slab();
    
    // Allocation operations
    void* allocate(size_t size, size_t alignment = 8);
    void deallocate(void* ptr, size_t size);
    
    // Slab properties
    size_t size() const { return size_; }
    size_t used() const { return used_.load(std::memory_order_relaxed); }
    size_t available() const { return size_ - used(); }
    double usage_ratio() const { return static_cast<double>(used()) / size_; }
    bool contains(void* ptr) const;
    
    // Reset slab for reuse
    void reset();
    
private:
    std::unique_ptr<uint8_t, decltype(&std::free)> memory_;
    size_t size_;
    std::atomic<size_t> used_{0};
    std::atomic<size_t> offset_{0};
    std::mutex mutex_;
    
    // Free list for deallocated chunks
    struct FreeBlock {
        size_t offset;
        size_t size;
        bool operator<(const FreeBlock& other) const {
            return offset < other.offset;
        }
    };
    std::vector<FreeBlock> free_list_;
    
    // Helper methods
    void coalesce_free_blocks();
    void* allocate_from_free_list(size_t size, size_t alignment);
};

/**
 * @brief Dynamic slab allocator with adaptive sizing
 */
class DynamicSlabAllocator {
public:
    explicit DynamicSlabAllocator(const DynamicSlabConfig& config = {});
    ~DynamicSlabAllocator();
    
    /**
     * @brief Allocate memory from slabs
     * @param size Size in bytes
     * @param alignment Alignment requirement
     * @return Pointer to allocated memory or nullptr if failed
     */
    void* allocate(size_t size, size_t alignment = 8);
    
    /**
     * @brief Deallocate memory
     * @param ptr Pointer to deallocate
     * @param size Size of allocation
     */
    void deallocate(void* ptr, size_t size);
    
    /**
     * @brief Get allocator statistics
     */
    DynamicSlabStats get_stats() const;
    
    /**
     * @brief Force a maintenance cycle (growth/shrink check)
     */
    void perform_maintenance();
    
    /**
     * @brief Reset all slabs (deallocate everything)
     */
    void reset();
    
    /**
     * @brief Set high water mark for triggering growth
     */
    void set_high_water_mark(double ratio) {
        config_.high_water_mark = std::clamp(ratio, 0.5, 0.95);
    }
    
    /**
     * @brief Set low water mark for triggering shrink
     */
    void set_low_water_mark(double ratio) {
        config_.low_water_mark = std::clamp(ratio, 0.05, 0.5);
    }
    
private:
    DynamicSlabConfig config_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Slab>> slabs_;
    
    // Usage tracking
    std::atomic<uint64_t> total_allocations_{0};
    std::atomic<uint64_t> total_deallocations_{0};
    std::atomic<uint64_t> slab_growths_{0};
    std::atomic<uint64_t> slab_shrinks_{0};
    
    // Maintenance timing
    std::chrono::steady_clock::time_point last_shrink_check_;
    std::chrono::steady_clock::time_point last_growth_;
    
    // Helper methods
    void grow_slabs();
    void shrink_slabs();
    size_t calculate_next_slab_size() const;
    Slab* find_suitable_slab(size_t size);
    bool should_grow() const;
    bool should_shrink() const;
};

/**
 * @brief Thread-local slab allocator for reduced contention
 */
class ThreadLocalSlabAllocator {
public:
    explicit ThreadLocalSlabAllocator(const DynamicSlabConfig& config = {});
    
    void* allocate(size_t size, size_t alignment = 8);
    void deallocate(void* ptr, size_t size);
    DynamicSlabStats get_stats() const;
    
private:
    static thread_local std::unique_ptr<DynamicSlabAllocator> tls_allocator_;
    DynamicSlabConfig config_;
    
    DynamicSlabAllocator* get_allocator();
};

/**
 * @brief Global allocator instance
 */
class GlobalSlabAllocator {
public:
    static GlobalSlabAllocator& instance();
    
    void* allocate(size_t size, size_t alignment = 8);
    void deallocate(void* ptr, size_t size);
    void configure(const DynamicSlabConfig& config);
    DynamicSlabStats get_stats() const;
    
private:
    GlobalSlabAllocator();
    ~GlobalSlabAllocator();
    
    std::unique_ptr<DynamicSlabAllocator> allocator_;
    mutable std::mutex mutex_;
};

/**
 * @brief Scoped allocator for automatic cleanup
 */
class ScopedSlabAllocator {
public:
    explicit ScopedSlabAllocator(DynamicSlabAllocator& allocator)
        : allocator_(allocator) {}
    
    ~ScopedSlabAllocator() {
        for (auto& [ptr, size] : allocations_) {
            allocator_.deallocate(ptr, size);
        }
    }
    
    void* allocate(size_t size, size_t alignment = 8) {
        void* ptr = allocator_.allocate(size, alignment);
        if (ptr) {
            allocations_.emplace_back(ptr, size);
        }
        return ptr;
    }
    
    void deallocate(void* ptr, size_t size) {
        allocator_.deallocate(ptr, size);
        allocations_.erase(
            std::remove_if(allocations_.begin(), allocations_.end(),
                [ptr](const auto& p) { return p.first == ptr; }),
            allocations_.end()
        );
    }
    
private:
    DynamicSlabAllocator& allocator_;
    std::vector<std::pair<void*, size_t>> allocations_;
};

} // namespace memory
} // namespace psyne