#include "psyne/memory/memory_slab.hpp"
#include "logger.hpp"

#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/mman.h> // For MAP_HUGE_2MB
#include <numa.h>
#include <numaif.h>
#include <sys/syscall.h>
#endif

#ifdef PSYNE_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_map.h>
#include <mach/vm_page_size.h>
#endif

namespace psyne {

// Platform-specific huge page size
static constexpr size_t HUGE_PAGE_SIZE = 2 * 1024 * 1024; // 2MB

// Align value up to alignment
static inline size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

MemorySlab::MemorySlab(const MemorySlabConfig &config) {
    allocate_memory(config);
}

MemorySlab::~MemorySlab() {
    if (gpu_pinned_) {
        unpin_from_gpu();
    }
    deallocate_memory();
}

MemorySlab::MemorySlab(MemorySlab &&other) noexcept
    : base_(other.base_), size_(other.size_),
      huge_pages_enabled_(other.huge_pages_enabled_),
      gpu_accessible_(other.gpu_accessible_), gpu_pinned_(other.gpu_pinned_),
      numa_node_(other.numa_node_), alignment_(other.alignment_)
#ifdef __linux__
      ,
      memfd_(other.memfd_)
#endif
{
    other.base_ = nullptr;
    other.size_ = 0;
#ifdef __linux__
    other.memfd_ = -1;
#endif
}

MemorySlab &MemorySlab::operator=(MemorySlab &&other) noexcept {
    if (this != &other) {
        if (gpu_pinned_) {
            unpin_from_gpu();
        }
        deallocate_memory();

        base_ = other.base_;
        size_ = other.size_;
        huge_pages_enabled_ = other.huge_pages_enabled_;
        gpu_accessible_ = other.gpu_accessible_;
        gpu_pinned_ = other.gpu_pinned_;
        numa_node_ = other.numa_node_;
        alignment_ = other.alignment_;
#ifdef __linux__
        memfd_ = other.memfd_;
        other.memfd_ = -1;
#endif

        other.base_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MemorySlab::allocate_memory(const MemorySlabConfig &config) {
    size_ = align_up(config.size_bytes, config.alignment);
    alignment_ = config.alignment;
    numa_node_ = config.numa_node;
    gpu_accessible_ = config.gpu_accessible;

    // Align size to huge page boundary if requested
    if (config.use_huge_pages) {
        size_ = align_up(size_, HUGE_PAGE_SIZE);
    }

#ifdef __linux__
    // Linux implementation with huge page support
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;

    if (config.use_huge_pages && size_ >= HUGE_PAGE_SIZE) {
        // Try to use huge pages
        flags |= MAP_HUGETLB | MAP_HUGE_2MB;

        // Try mmap with huge pages first
        base_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, flags, -1, 0);

        if (base_ == MAP_FAILED) {
            // Fallback to transparent huge pages
            flags = MAP_PRIVATE | MAP_ANONYMOUS;
            base_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, flags, -1, 0);

            if (base_ != MAP_FAILED) {
                // Advise kernel to use huge pages
                madvise(base_, size_, MADV_HUGEPAGE);
                log_debug("Allocated ", size_ / 1024 / 1024,
                          "MB using transparent huge pages");
            }
        } else {
            huge_pages_enabled_ = true;
            log_debug("Allocated ", size_ / 1024 / 1024,
                      "MB using explicit huge pages");
        }
    } else {
        // Regular allocation
        base_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, flags, -1, 0);
    }

    if (base_ == MAP_FAILED) {
        throw std::runtime_error("Failed to allocate memory slab: " +
                                 std::string(strerror(errno)));
    }

    // Set NUMA node affinity if specified
    if (numa_node_ >= 0 && numa_available() >= 0) {
        mbind(base_, size_, MPOL_BIND, numa_get_mems_allowed()->maskp,
              numa_get_mems_allowed()->size + 1, MPOL_MF_STRICT);
    }

    // Lock memory to prevent swapping
    if (mlock(base_, size_) == 0) {
        log_trace("Locked ", size_ / 1024 / 1024, "MB in memory");
    }

#elif defined(__APPLE__)
    // macOS implementation
    kern_return_t kr;
    mach_vm_address_t addr = 0;

    int flags = VM_FLAGS_ANYWHERE;
    if (config.use_huge_pages && size_ >= HUGE_PAGE_SIZE) {
        flags |= VM_FLAGS_SUPERPAGE_SIZE_2MB;
    }

    kr = mach_vm_allocate(mach_task_self(), &addr, size_, flags);

    if (kr != KERN_SUCCESS) {
        // Fallback to regular allocation
        kr =
            mach_vm_allocate(mach_task_self(), &addr, size_, VM_FLAGS_ANYWHERE);
        if (kr != KERN_SUCCESS) {
            throw std::runtime_error("Failed to allocate memory slab");
        }
    } else if (flags & VM_FLAGS_SUPERPAGE_SIZE_2MB) {
        huge_pages_enabled_ = true;
        log_debug("Allocated ", size_ / 1024 / 1024, "MB using superpages");
    }

    base_ = reinterpret_cast<void *>(addr);

    // Wire memory to prevent paging
    kr = mach_vm_wire(mach_host_self(), mach_task_self(), addr, size_,
                      VM_PROT_READ | VM_PROT_WRITE);
    if (kr == KERN_SUCCESS) {
        log_trace("Wired ", size_ / 1024 / 1024, "MB in memory");
    }

#else
    // Generic POSIX implementation
    base_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (base_ == MAP_FAILED) {
        throw std::runtime_error("Failed to allocate memory slab");
    }
#endif

    // Zero-initialize the memory
    std::memset(base_, 0, size_);

    // Pin for GPU if requested
    if (gpu_accessible_) {
        try {
            pin_for_gpu();
        } catch (const std::exception &e) {
            log_warn("Failed to pin memory for GPU: ", e.what());
            gpu_accessible_ = false;
        }
    }
}

void MemorySlab::deallocate_memory() {
    if (!base_)
        return;

#ifdef __linux__
    munlock(base_, size_);
    munmap(base_, size_);
    if (memfd_ >= 0) {
        close(memfd_);
        memfd_ = -1;
    }
#elif defined(__APPLE__)
    mach_vm_address_t addr = reinterpret_cast<mach_vm_address_t>(base_);
    mach_vm_deallocate(mach_task_self(), addr, size_);
#else
    munmap(base_, size_);
#endif

    base_ = nullptr;
    size_ = 0;
}

void MemorySlab::pin_for_gpu() {
    if (gpu_pinned_ || !base_)
        return;

#ifdef PSYNE_CUDA_ENABLED
    cudaError_t err = cudaHostRegister(base_, size_, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to pin memory for CUDA: " +
                                 std::string(cudaGetErrorString(err)));
    }
    gpu_pinned_ = true;
    log_trace("Pinned ", size_ / 1024 / 1024, "MB for GPU access");
#endif
}

void MemorySlab::unpin_from_gpu() {
    if (!gpu_pinned_ || !base_)
        return;

#ifdef PSYNE_CUDA_ENABLED
    cudaError_t err = cudaHostUnregister(base_);
    if (err != cudaSuccess) {
        log_warn("Failed to unpin GPU memory: ", cudaGetErrorString(err));
    }
    gpu_pinned_ = false;
#endif
}

void MemorySlab::prefetch_to_gpu(int device_id, size_t offset, size_t size) {
    if (!gpu_accessible_ || !base_)
        return;

    if (size == 0)
        size = size_ - offset;

#ifdef PSYNE_CUDA_ENABLED
    cudaError_t err = cudaMemPrefetchAsync(static_cast<char *>(base_) + offset,
                                           size, device_id,
                                           0 // Default stream
    );
    if (err != cudaSuccess) {
        log_warn("Failed to prefetch to GPU: ", cudaGetErrorString(err));
    }
#endif
}

void MemorySlab::prefetch_to_cpu(size_t offset, size_t size) {
    if (!gpu_accessible_ || !base_)
        return;

    if (size == 0)
        size = size_ - offset;

#ifdef PSYNE_CUDA_ENABLED
    cudaError_t err = cudaMemPrefetchAsync(static_cast<char *>(base_) + offset,
                                           size, cudaCpuDeviceId,
                                           0 // Default stream
    );
    if (err != cudaSuccess) {
        log_warn("Failed to prefetch to CPU: ", cudaGetErrorString(err));
    }
#endif
}

// MemorySlabPool implementation

MemorySlabPool::MemorySlabPool(const MemorySlabConfig &config,
                               size_t initial_slabs)
    : config_(config) {
    reserve(initial_slabs);
}

std::unique_ptr<MemorySlab> MemorySlabPool::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (available_slabs_.empty()) {
        // Allocate new slab on demand
        auto slab = std::make_unique<MemorySlab>(config_);
        total_allocated_.fetch_add(1, std::memory_order_relaxed);
        return slab;
    }

    auto slab = std::move(available_slabs_.back());
    available_slabs_.pop_back();
    return slab;
}

void MemorySlabPool::release(std::unique_ptr<MemorySlab> slab) {
    if (!slab)
        return;

    // Zero out the slab for security/consistency
    std::memset(slab->data(), 0, slab->size());

    std::lock_guard<std::mutex> lock(mutex_);
    available_slabs_.push_back(std::move(slab));
}

size_t MemorySlabPool::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_slabs_.size();
}

void MemorySlabPool::reserve(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t i = 0; i < count; ++i) {
        available_slabs_.push_back(std::make_unique<MemorySlab>(config_));
        total_allocated_.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace psyne