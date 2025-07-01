#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace psyne {

/**
 * @brief Memory slab configuration options
 *
 * Controls how memory slabs are allocated and managed for channels.
 */
struct MemorySlabConfig {
    size_t size_bytes = 32 * 1024 * 1024; // Default 32MB
    bool use_huge_pages = true;           // Use 2MB huge pages if available
    bool gpu_accessible = false;          // Make memory GPU accessible
    int numa_node = -1;                   // NUMA node affinity (-1 = any)
    size_t alignment = 64;                // Cache line alignment
};

/**
 * @brief A contiguous memory region for zero-copy message passing
 *
 * Memory slabs are the foundation of Psyne's zero-copy architecture.
 * Messages are allocated directly within slabs, eliminating copies.
 */
class MemorySlab {
public:
    /**
     * @brief Construct a memory slab with the given configuration
     */
    explicit MemorySlab(const MemorySlabConfig &config);

    ~MemorySlab();

    // Delete copy operations
    MemorySlab(const MemorySlab &) = delete;
    MemorySlab &operator=(const MemorySlab &) = delete;

    // Allow move operations
    MemorySlab(MemorySlab &&other) noexcept;
    MemorySlab &operator=(MemorySlab &&other) noexcept;

    /**
     * @brief Get the base address of the slab
     */
    void *data() noexcept {
        return base_;
    }
    const void *data() const noexcept {
        return base_;
    }

    /**
     * @brief Get the size of the slab in bytes
     */
    size_t size() const noexcept {
        return size_;
    }

    /**
     * @brief Check if the slab uses huge pages
     */
    bool uses_huge_pages() const noexcept {
        return huge_pages_enabled_;
    }

    /**
     * @brief Check if the slab is GPU accessible
     */
    bool is_gpu_accessible() const noexcept {
        return gpu_accessible_;
    }

    /**
     * @brief Get pointer at specific offset
     */
    void *at(size_t offset) noexcept {
        return static_cast<char *>(base_) + offset;
    }

    /**
     * @brief Get NUMA node affinity
     */
    int numa_node() const noexcept {
        return numa_node_;
    }

    /**
     * @brief Pin memory for GPU access (CUDA/Metal/Vulkan)
     */
    void pin_for_gpu();

    /**
     * @brief Unpin GPU memory
     */
    void unpin_from_gpu();

    /**
     * @brief Prefetch memory to GPU
     */
    void prefetch_to_gpu(int device_id, size_t offset = 0, size_t size = 0);

    /**
     * @brief Prefetch memory to CPU
     */
    void prefetch_to_cpu(size_t offset = 0, size_t size = 0);

private:
    void *base_ = nullptr;
    size_t size_ = 0;
    bool huge_pages_enabled_ = false;
    bool gpu_accessible_ = false;
    bool gpu_pinned_ = false;
    int numa_node_ = -1;
    size_t alignment_ = 64;

    // Platform-specific handles
#ifdef __linux__
    int memfd_ = -1; // For huge page support
#endif

    void allocate_memory(const MemorySlabConfig &config);
    void deallocate_memory();
};

/**
 * @brief Pool of memory slabs for efficient allocation
 *
 * Manages a pool of pre-allocated slabs to avoid allocation overhead
 * in the critical path.
 */
class MemorySlabPool {
public:
    explicit MemorySlabPool(const MemorySlabConfig &config,
                            size_t initial_slabs = 4);

    ~MemorySlabPool() = default;

    /**
     * @brief Get a slab from the pool
     */
    std::unique_ptr<MemorySlab> acquire();

    /**
     * @brief Return a slab to the pool
     */
    void release(std::unique_ptr<MemorySlab> slab);

    /**
     * @brief Get number of available slabs
     */
    size_t available() const;

    /**
     * @brief Pre-allocate additional slabs
     */
    void reserve(size_t count);

private:
    MemorySlabConfig config_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<MemorySlab>> available_slabs_;
    std::atomic<size_t> total_allocated_{0};
};

} // namespace psyne