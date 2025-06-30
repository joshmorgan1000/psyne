/**
 * @file custom_allocator.hpp
 * @brief High-performance custom memory allocator with huge page support
 *
 * Provides a custom allocator optimized for large tensor allocations
 * with support for huge pages, NUMA awareness, and memory pooling.
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#endif

namespace psyne {
namespace memory {

/**
 * @brief Memory allocation flags
 */
enum class AllocFlags : uint32_t {
    None = 0,
    HugePage = 1 << 0,     // Use huge pages (2MB/1GB)
    Pinned = 1 << 1,       // Pin memory (prevent swapping)
    Zeroed = 1 << 2,       // Zero-initialize memory
    NumaLocal = 1 << 3,    // Allocate on local NUMA node
    Aligned64 = 1 << 4,    // 64-byte cache line alignment
    Aligned4K = 1 << 5,    // 4KB page alignment
    GPUAccessible = 1 << 6 // Accessible from GPU
};

inline AllocFlags operator|(AllocFlags a, AllocFlags b) {
    return static_cast<AllocFlags>(static_cast<uint32_t>(a) |
                                   static_cast<uint32_t>(b));
}

inline AllocFlags operator&(AllocFlags a, AllocFlags b) {
    return static_cast<AllocFlags>(static_cast<uint32_t>(a) &
                                   static_cast<uint32_t>(b));
}

/**
 * @brief Memory allocation statistics
 */
struct AllocStats {
    std::atomic<size_t> total_allocated{0};
    std::atomic<size_t> total_freed{0};
    std::atomic<size_t> current_usage{0};
    std::atomic<size_t> peak_usage{0};
    std::atomic<size_t> huge_page_count{0};
    std::atomic<size_t> allocation_count{0};
    std::atomic<size_t> free_count{0};
};

/**
 * @brief Custom memory allocator with advanced features
 */
class CustomAllocator {
public:
    static CustomAllocator &instance() {
        static CustomAllocator allocator;
        return allocator;
    }

    /**
     * @brief Allocate memory with specified flags
     * @param size Size in bytes
     * @param flags Allocation flags
     * @param alignment Required alignment (0 for default)
     * @return Allocated memory pointer
     */
    void *allocate(size_t size, AllocFlags flags = AllocFlags::None,
                   size_t alignment = 0);

    /**
     * @brief Free previously allocated memory
     * @param ptr Pointer to free
     */
    void deallocate(void *ptr);

    /**
     * @brief Get allocation size for a pointer
     * @param ptr Allocated pointer
     * @return Size in bytes
     */
    size_t get_size(void *ptr) const;

    /**
     * @brief Get allocation statistics
     */
    const AllocStats &get_stats() const {
        return stats_;
    }

    /**
     * @brief Enable/disable huge page support
     */
    void set_huge_pages_enabled(bool enabled) {
        huge_pages_enabled_ = enabled;
    }

    /**
     * @brief Set preferred NUMA node (-1 for any)
     */
    void set_numa_node(int node) {
        preferred_numa_node_ = node;
    }

private:
    CustomAllocator();
    ~CustomAllocator();

    // Allocation metadata
    struct AllocInfo {
        size_t size;
        AllocFlags flags;
        void *actual_ptr; // For aligned allocations
        size_t actual_size;
    };

    // Memory pools for different size classes
    struct MemoryPool {
        std::vector<void *> free_list;
        size_t block_size;
        size_t blocks_per_chunk;
        std::vector<void *> chunks;
        std::mutex mutex;
    };

    // Allocate from pool
    void *allocate_from_pool(size_t size);
    void return_to_pool(void *ptr, size_t size);

    // Allocate huge page
    void *allocate_huge_page(size_t size, int numa_node = -1);

    // Statistics
    AllocStats stats_;

    // Configuration
    bool huge_pages_enabled_ = true;
    int preferred_numa_node_ = -1;

    // Metadata tracking
    std::unordered_map<void *, AllocInfo> allocations_;
    mutable std::mutex allocations_mutex_;

public:
    // Memory pools for common sizes (needs to be public for implementation)
    static constexpr size_t pool_sizes[] = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
        
private:
    std::array<MemoryPool, sizeof(pool_sizes) / sizeof(pool_sizes[0])> pools_;

    // Platform-specific initialization
    void platform_init();
};

/**
 * @brief STL-compatible allocator using CustomAllocator
 */
template <typename T>
class StlCustomAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    StlCustomAllocator() noexcept = default;

    template <typename U>
    StlCustomAllocator(const StlCustomAllocator<U> &) noexcept {}

    T *allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }

        void *ptr = CustomAllocator::instance().allocate(
            n * sizeof(T), AllocFlags::Aligned64 | AllocFlags::Zeroed);

        if (!ptr) {
            throw std::bad_alloc();
        }

        return static_cast<T *>(ptr);
    }

    void deallocate(T *ptr, size_type) noexcept {
        CustomAllocator::instance().deallocate(ptr);
    }

    template <typename U>
    bool operator==(const StlCustomAllocator<U> &) const noexcept {
        return true;
    }

    template <typename U>
    bool operator!=(const StlCustomAllocator<U> &) const noexcept {
        return false;
    }
};

/**
 * @brief RAII wrapper for custom allocations
 */
class UniqueAlloc {
public:
    UniqueAlloc() noexcept : ptr_(nullptr), size_(0) {}

    explicit UniqueAlloc(size_t size, AllocFlags flags = AllocFlags::None)
        : ptr_(CustomAllocator::instance().allocate(size, flags)), size_(size) {
    }

    ~UniqueAlloc() {
        if (ptr_) {
            CustomAllocator::instance().deallocate(ptr_);
        }
    }

    // Move semantics
    UniqueAlloc(UniqueAlloc &&other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    UniqueAlloc &operator=(UniqueAlloc &&other) noexcept {
        if (this != &other) {
            if (ptr_) {
                CustomAllocator::instance().deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Delete copy operations
    UniqueAlloc(const UniqueAlloc &) = delete;
    UniqueAlloc &operator=(const UniqueAlloc &) = delete;

    void *get() noexcept {
        return ptr_;
    }
    const void *get() const noexcept {
        return ptr_;
    }
    size_t size() const noexcept {
        return size_;
    }

    void *release() noexcept {
        void *tmp = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return tmp;
    }

    void reset(void *ptr = nullptr, size_t size = 0) {
        if (ptr_) {
            CustomAllocator::instance().deallocate(ptr_);
        }
        ptr_ = ptr;
        size_ = size;
    }

    explicit operator bool() const noexcept {
        return ptr_ != nullptr;
    }

private:
    void *ptr_;
    size_t size_;
};

/**
 * @brief Allocate memory for tensor with optimal settings
 */
inline void *allocate_tensor(size_t size, bool use_huge_pages = true) {
    AllocFlags flags = AllocFlags::Aligned64 | AllocFlags::Zeroed;
    if (use_huge_pages && size >= 2 * 1024 * 1024) { // 2MB threshold
        flags = flags | AllocFlags::HugePage;
    }
    return CustomAllocator::instance().allocate(size, flags);
}

/**
 * @brief Free tensor memory
 */
inline void deallocate_tensor(void *ptr) {
    CustomAllocator::instance().deallocate(ptr);
}

} // namespace memory
} // namespace psyne