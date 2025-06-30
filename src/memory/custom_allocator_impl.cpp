/**
 * @file custom_allocator_impl.cpp
 * @brief Implementation of custom memory allocator
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <psyne/psyne.hpp>
#include <unordered_map>

#ifdef __linux__
#include <numa.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace psyne {
namespace memory {

/**
 * @brief Implementation of CustomAllocator
 */
class CustomAllocator::Impl {
public:
    Impl() {
        // Initialize huge page size
        huge_page_size_ = get_system_huge_page_size();
        huge_pages_available_ = (huge_page_size_ > 0);
        numa_nodes_ = get_system_numa_nodes();
    }

    void *allocate(size_t size, AllocFlags flags) {
        std::lock_guard<std::mutex> lock(mutex_);

        void *ptr = nullptr;
        bool is_huge_page = false;

        // Handle alignment
        size_t alignment = 1;
        if (static_cast<uint32_t>(flags) &
            static_cast<uint32_t>(AllocFlags::Aligned64)) {
            alignment = 64;
        } else if (static_cast<uint32_t>(flags) &
                   static_cast<uint32_t>(AllocFlags::Aligned128)) {
            alignment = 128;
        } else if (static_cast<uint32_t>(flags) &
                   static_cast<uint32_t>(AllocFlags::Aligned256)) {
            alignment = 256;
        }

        // Try huge pages if requested and available
        if ((static_cast<uint32_t>(flags) &
             static_cast<uint32_t>(AllocFlags::HugePage)) &&
            huge_pages_available_ && size >= huge_page_size_) {
            ptr = allocate_huge_page(size);
            is_huge_page = (ptr != nullptr);
        }

        // Fall back to regular allocation
        if (!ptr) {
            if (alignment > 1) {
                ptr = aligned_alloc(alignment, size);
            } else {
                ptr = malloc(size);
            }
        }

        if (ptr) {
            // Zero-initialize if requested
            if (static_cast<uint32_t>(flags) &
                static_cast<uint32_t>(AllocFlags::Zeroed)) {
                memset(ptr, 0, size);
            }

            // Store allocation info
            BlockInfo info;
            info.size = size;
            info.is_huge_page = is_huge_page;
            info.numa_node = 0; // Simplified
            info.base_address = ptr;

            allocations_[ptr] = info;

            // Update stats
            stats_.total_allocated += size;
            stats_.current_usage += size;
            stats_.peak_usage =
                std::max(stats_.peak_usage, stats_.current_usage);
            stats_.allocation_count++;

            if (is_huge_page) {
                stats_.huge_page_count++;
            }
        }

        return ptr;
    }

    void deallocate(void *ptr) {
        if (!ptr)
            return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            const BlockInfo &info = it->second;

            if (info.is_huge_page) {
                deallocate_huge_page(ptr, info.size);
            } else {
                free(ptr);
            }

            // Update stats
            stats_.total_freed += info.size;
            stats_.current_usage -= info.size;
            stats_.free_count++;

            allocations_.erase(it);
        }
    }

    bool huge_pages_available() const {
        return huge_pages_available_;
    }

    size_t get_huge_page_size() const {
        return huge_page_size_;
    }

    int get_numa_nodes() const {
        return numa_nodes_;
    }

    AllocatorStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    std::optional<BlockInfo> get_block_info(void *ptr) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

private:
    void *allocate_huge_page(size_t size) {
#ifdef __linux__
        // Round up to huge page boundary
        size_t aligned_size =
            ((size + huge_page_size_ - 1) / huge_page_size_) * huge_page_size_;

        void *ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

        if (ptr == MAP_FAILED) {
            return nullptr;
        }

        return ptr;
#else
        (void)size;
        return nullptr;
#endif
    }

    void deallocate_huge_page(void *ptr, size_t size) {
#ifdef __linux__
        size_t aligned_size =
            ((size + huge_page_size_ - 1) / huge_page_size_) * huge_page_size_;
        munmap(ptr, aligned_size);
#else
        (void)ptr;
        (void)size;
#endif
    }

    size_t get_system_huge_page_size() {
#ifdef __linux__
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            // Assume 2MB huge pages on Linux
            return 2 * 1024 * 1024;
        }
#endif
        return 0;
    }

    int get_system_numa_nodes() {
#ifdef __linux__
        if (numa_available() == 0) {
            return numa_max_node() + 1;
        }
#endif
        return 1;
    }

    mutable std::mutex mutex_;
    std::unordered_map<void *, BlockInfo> allocations_;
    AllocatorStats stats_;
    bool huge_pages_available_ = false;
    size_t huge_page_size_ = 0;
    int numa_nodes_ = 1;
};

// CustomAllocator implementation
CustomAllocator &CustomAllocator::instance() {
    static CustomAllocator instance;
    return instance;
}

CustomAllocator::CustomAllocator() : impl_(std::make_unique<Impl>()) {}

void *CustomAllocator::allocate(size_t size, AllocFlags flags) {
    return impl_->allocate(size, flags);
}

void CustomAllocator::deallocate(void *ptr) {
    impl_->deallocate(ptr);
}

bool CustomAllocator::huge_pages_available() const {
    return impl_->huge_pages_available();
}

size_t CustomAllocator::get_huge_page_size() const {
    return impl_->get_huge_page_size();
}

int CustomAllocator::get_numa_nodes() const {
    return impl_->get_numa_nodes();
}

AllocatorStats CustomAllocator::stats() const {
    return impl_->stats();
}

std::optional<BlockInfo> CustomAllocator::get_block_info(void *ptr) const {
    return impl_->get_block_info(ptr);
}

// UniqueAlloc implementation
UniqueAlloc::UniqueAlloc(size_t size, AllocFlags flags) : size_(size) {
    ptr_ = CustomAllocator::instance().allocate(size, flags);
}

UniqueAlloc::~UniqueAlloc() {
    reset();
}

UniqueAlloc::UniqueAlloc(UniqueAlloc &&other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

UniqueAlloc &UniqueAlloc::operator=(UniqueAlloc &&other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void UniqueAlloc::reset() {
    if (ptr_) {
        CustomAllocator::instance().deallocate(ptr_);
        ptr_ = nullptr;
        size_ = 0;
    }
}

// Helper functions
void *allocate_tensor(size_t size) {
    return CustomAllocator::instance().allocate(size, AllocFlags::Aligned64 |
                                                          AllocFlags::Zeroed);
}

void deallocate_tensor(void *ptr) {
    CustomAllocator::instance().deallocate(ptr);
}

} // namespace memory
} // namespace psyne