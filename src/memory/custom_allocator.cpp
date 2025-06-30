/**
 * @file custom_allocator.cpp
 * @brief Custom allocator implementation
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include "custom_allocator.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace psyne {
namespace memory {

// Size class helpers
static size_t round_up_to_power_of_2(size_t size) {
    size--;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    size |= size >> 32;
    size++;
    return size;
}

static int get_pool_index(size_t size) {
    for (size_t i = 0; i < sizeof(CustomAllocator::pool_sizes) /
                               sizeof(CustomAllocator::pool_sizes[0]);
         ++i) {
        if (size <= CustomAllocator::pool_sizes[i]) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

CustomAllocator::CustomAllocator() {
    platform_init();

    // Initialize memory pools
    for (size_t i = 0; i < pools_.size(); ++i) {
        pools_[i].block_size = pool_sizes[i];
        pools_[i].blocks_per_chunk =
            std::max(size_t(1), size_t(1024 * 1024) / pool_sizes[i]);
    }
}

CustomAllocator::~CustomAllocator() {
    // Clean up all pools
    for (auto &pool : pools_) {
        std::lock_guard<std::mutex> lock(pool.mutex);
        for (void *chunk : pool.chunks) {
            std::free(chunk);
        }
    }
}

void CustomAllocator::platform_init() {
#ifdef __linux__
    // Check if huge pages are available
    FILE *fp = fopen("/proc/sys/vm/nr_hugepages", "r");
    if (fp) {
        int nr_hugepages = 0;
        if (fscanf(fp, "%d", &nr_hugepages) == 1 && nr_hugepages > 0) {
            huge_pages_enabled_ = true;
        }
        fclose(fp);
    }

    // Initialize NUMA if available
    if (numa_available() >= 0) {
        preferred_numa_node_ = numa_node_of_cpu(sched_getcpu());
    }
#endif
}

void *CustomAllocator::allocate(size_t size, AllocFlags flags,
                                size_t alignment) {
    if (size == 0)
        return nullptr;

    // Update statistics
    stats_.allocation_count.fetch_add(1, std::memory_order_relaxed);

    // Determine alignment
    if (alignment == 0) {
        if (static_cast<uint32_t>(flags & AllocFlags::Aligned4K)) {
            alignment = 4096;
        } else if (static_cast<uint32_t>(flags & AllocFlags::Aligned64)) {
            alignment = 64;
        } else {
            alignment = alignof(std::max_align_t);
        }
    }

    // Try pool allocation for small sizes
    if (!static_cast<uint32_t>(flags & AllocFlags::HugePage) &&
        size <= pool_sizes[pools_.size() - 1]) {
        void *ptr = allocate_from_pool(size);
        if (ptr) {
            // Track allocation
            std::lock_guard<std::mutex> lock(allocations_mutex_);
            allocations_[ptr] = {size, flags, ptr, size};

            // Update stats
            stats_.total_allocated.fetch_add(size, std::memory_order_relaxed);
            stats_.current_usage.fetch_add(size, std::memory_order_relaxed);

            size_t current =
                stats_.current_usage.load(std::memory_order_relaxed);
            size_t peak = stats_.peak_usage.load(std::memory_order_relaxed);
            while (current > peak &&
                   !stats_.peak_usage.compare_exchange_weak(peak, current)) {
            }

            return ptr;
        }
    }

    void *ptr = nullptr;
    void *actual_ptr = nullptr;
    size_t actual_size = size + alignment - 1;

#ifdef __linux__
    // Try huge page allocation
    if (static_cast<uint32_t>(flags & AllocFlags::HugePage) && huge_pages_enabled_ &&
        size >= 2 * 1024 * 1024) {
        ptr = allocate_huge_page(size, preferred_numa_node_);
        if (ptr) {
            actual_ptr = ptr;
            actual_size = round_up_to_power_of_2(size);
            stats_.huge_page_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
#endif

    // Fall back to regular allocation
    if (!ptr) {
#ifdef _WIN32
        actual_ptr = _aligned_malloc(size, alignment);
        ptr = actual_ptr;
#else
        if (posix_memalign(&actual_ptr, alignment, size) == 0) {
            ptr = actual_ptr;
        }
#endif
    }

    if (!ptr) {
        return nullptr;
    }

    // Zero memory if requested
    if (static_cast<uint32_t>(flags & AllocFlags::Zeroed)) {
        // Zero-copy compliant: manual loop instead of memset
        char *bytes = static_cast<char *>(ptr);
        for (size_t i = 0; i < size; ++i) {
            bytes[i] = 0;
        }
    }

    // Pin memory if requested
    if (static_cast<uint32_t>(flags & AllocFlags::Pinned)) {
#ifdef __linux__
        mlock(ptr, size);
#elif defined(_WIN32)
        VirtualLock(ptr, size);
#endif
    }

    // Track allocation
    {
        std::lock_guard<std::mutex> lock(allocations_mutex_);
        allocations_[ptr] = {size, flags, actual_ptr, actual_size};
    }

    // Update statistics
    stats_.total_allocated.fetch_add(size, std::memory_order_relaxed);
    stats_.current_usage.fetch_add(size, std::memory_order_relaxed);

    size_t current = stats_.current_usage.load(std::memory_order_relaxed);
    size_t peak = stats_.peak_usage.load(std::memory_order_relaxed);
    while (current > peak &&
           !stats_.peak_usage.compare_exchange_weak(peak, current)) {
    }

    return ptr;
}

void CustomAllocator::deallocate(void *ptr) {
    if (!ptr)
        return;

    AllocInfo info;
    {
        std::lock_guard<std::mutex> lock(allocations_mutex_);
        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            return; // Not our allocation
        }
        info = it->second;
        allocations_.erase(it);
    }

    // Update statistics
    stats_.free_count.fetch_add(1, std::memory_order_relaxed);
    stats_.total_freed.fetch_add(info.size, std::memory_order_relaxed);
    stats_.current_usage.fetch_sub(info.size, std::memory_order_relaxed);

    // Unpin memory if it was pinned
    if (static_cast<uint32_t>(info.flags & AllocFlags::Pinned)) {
#ifdef __linux__
        munlock(ptr, info.size);
#elif defined(_WIN32)
        VirtualUnlock(ptr, info.size);
#endif
    }

    // Try to return to pool
    if (!static_cast<uint32_t>(info.flags & AllocFlags::HugePage) &&
        info.size <= pool_sizes[pools_.size() - 1]) {
        return_to_pool(ptr, info.size);
        return;
    }

    // Free the memory
#ifdef __linux__
    if (static_cast<uint32_t>(info.flags & AllocFlags::HugePage)) {
        munmap(info.actual_ptr, info.actual_size);
        stats_.huge_page_count.fetch_sub(1, std::memory_order_relaxed);
    } else {
        std::free(info.actual_ptr);
    }
#elif defined(_WIN32)
    _aligned_free(info.actual_ptr);
#else
    std::free(info.actual_ptr);
#endif
}

size_t CustomAllocator::get_size(void *ptr) const {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        return it->second.size;
    }
    return 0;
}

void *CustomAllocator::allocate_from_pool(size_t size) {
    int pool_index = get_pool_index(size);
    if (pool_index < 0)
        return nullptr;

    auto &pool = pools_[pool_index];
    std::lock_guard<std::mutex> lock(pool.mutex);

    // Try to get from free list
    if (!pool.free_list.empty()) {
        void *ptr = pool.free_list.back();
        pool.free_list.pop_back();
        return ptr;
    }

    // Allocate new chunk
    size_t chunk_size = pool.block_size * pool.blocks_per_chunk;
    void *chunk = std::aligned_alloc(64, chunk_size);
    if (!chunk)
        return nullptr;

    pool.chunks.push_back(chunk);

    // Add blocks to free list
    uint8_t *block = static_cast<uint8_t *>(chunk);
    for (size_t i = 1; i < pool.blocks_per_chunk; ++i) {
        pool.free_list.push_back(block + i * pool.block_size);
    }

    // Return first block
    return chunk;
}

void CustomAllocator::return_to_pool(void *ptr, size_t size) {
    int pool_index = get_pool_index(size);
    if (pool_index < 0) {
        std::free(ptr);
        return;
    }

    auto &pool = pools_[pool_index];
    std::lock_guard<std::mutex> lock(pool.mutex);
    pool.free_list.push_back(ptr);
}

void *CustomAllocator::allocate_huge_page(size_t size,
                                          [[maybe_unused]] int numa_node) {
#ifdef __linux__
    // Round up to huge page size (2MB)
    const size_t huge_page_size = 2 * 1024 * 1024;
    size_t aligned_size =
        ((size + huge_page_size - 1) / huge_page_size) * huge_page_size;

    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB;

    void *ptr =
        mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
        return nullptr;
    }

    // Set NUMA policy if requested
#ifdef HAVE_NUMA
    if (numa_node >= 0 && numa_available() >= 0) {
        unsigned long nodemask = 1UL << numa_node;
        mbind(ptr, aligned_size, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0);
    }
#endif

    return ptr;
#else
    // Huge pages not supported on this platform
    return nullptr;
#endif
}

} // namespace memory
} // namespace psyne