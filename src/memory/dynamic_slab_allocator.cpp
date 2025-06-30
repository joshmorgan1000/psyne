/**
 * @file dynamic_slab_allocator.cpp
 * @brief Dynamic slab allocator implementation
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include "dynamic_slab_allocator.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace psyne {
namespace memory {

// Slab implementation

Slab::Slab(size_t size)
    : memory_(static_cast<uint8_t *>(std::aligned_alloc(4096, size)),
              &std::free),
      size_(size) {
    if (!memory_) {
        throw std::bad_alloc();
    }

    // Initialize memory to zero
    std::memset(memory_.get(), 0, size_);
}

Slab::~Slab() = default;

void *Slab::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);

    // First try free list
    void *ptr = allocate_from_free_list(size, alignment);
    if (ptr) {
        return ptr;
    }

    // Align current offset
    size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

    // Check if we have enough space
    if (aligned_offset + size > size_) {
        return nullptr;
    }

    // Allocate from end of used region
    ptr = memory_.get() + aligned_offset;
    offset_ = aligned_offset + size;
    used_ += size;

    return ptr;
}

void Slab::deallocate(void *ptr, size_t size) {
    if (!ptr || !contains(ptr)) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Add to free list
    size_t offset = static_cast<uint8_t *>(ptr) - memory_.get();
    free_list_.push_back({offset, size});
    used_ -= size;

    // Coalesce adjacent free blocks
    coalesce_free_blocks();
}

bool Slab::contains(void *ptr) const {
    const uint8_t *p = static_cast<const uint8_t *>(ptr);
    return p >= memory_.get() && p < memory_.get() + size_;
}

void Slab::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    offset_ = 0;
    used_ = 0;
    free_list_.clear();
}

void Slab::coalesce_free_blocks() {
    if (free_list_.size() < 2) {
        return;
    }

    // Sort by offset
    std::sort(free_list_.begin(), free_list_.end());

    // Merge adjacent blocks
    std::vector<FreeBlock> merged;
    merged.reserve(free_list_.size());

    FreeBlock current = free_list_[0];
    for (size_t i = 1; i < free_list_.size(); ++i) {
        const auto &next = free_list_[i];
        if (current.offset + current.size == next.offset) {
            // Merge blocks
            current.size += next.size;
        } else {
            // Save current and start new
            merged.push_back(current);
            current = next;
        }
    }
    merged.push_back(current);

    free_list_ = std::move(merged);
}

void *Slab::allocate_from_free_list(size_t size, size_t alignment) {
    for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
        size_t aligned_offset = (it->offset + alignment - 1) & ~(alignment - 1);
        size_t padding = aligned_offset - it->offset;

        if (it->size >= size + padding) {
            void *ptr = memory_.get() + aligned_offset;

            // Update or remove the free block
            if (it->size == size + padding) {
                free_list_.erase(it);
            } else {
                it->offset = aligned_offset + size;
                it->size -= (size + padding);
            }

            used_ += size;
            return ptr;
        }
    }

    return nullptr;
}

// DynamicSlabAllocator implementation

DynamicSlabAllocator::DynamicSlabAllocator(const DynamicSlabConfig &config)
    : config_(config), last_shrink_check_(std::chrono::steady_clock::now()),
      last_growth_(std::chrono::steady_clock::now()) {
    // Create initial slab
    slabs_.emplace_back(std::make_unique<Slab>(config_.initial_slab_size));
}

DynamicSlabAllocator::~DynamicSlabAllocator() = default;

void *DynamicSlabAllocator::allocate(size_t size, size_t alignment) {
    if (size == 0 || size > config_.max_slab_size) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Find suitable slab
    Slab *slab = find_suitable_slab(size);

    if (!slab && slabs_.size() < config_.max_slabs) {
        // Need to grow
        grow_slabs();
        slab = find_suitable_slab(size);
    }

    if (!slab) {
        return nullptr;
    }

    void *ptr = slab->allocate(size, alignment);
    if (ptr) {
        total_allocations_++;

        // Check if we should grow
        if (should_grow()) {
            grow_slabs();
        }
    }

    return ptr;
}

void DynamicSlabAllocator::deallocate(void *ptr, size_t size) {
    if (!ptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Find containing slab
    for (auto &slab : slabs_) {
        if (slab->contains(ptr)) {
            slab->deallocate(ptr, size);
            total_deallocations_++;
            break;
        }
    }

    // Periodic maintenance
    perform_maintenance();
}

DynamicSlabStats DynamicSlabAllocator::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    DynamicSlabStats stats;
    stats.num_slabs = slabs_.size();
    stats.allocations = total_allocations_;
    stats.deallocations = total_deallocations_;
    stats.slab_growths = slab_growths_;
    stats.slab_shrinks = slab_shrinks_;

    for (const auto &slab : slabs_) {
        stats.total_capacity += slab->size();
        stats.total_used += slab->used();

        if (stats.largest_slab_size < slab->size()) {
            stats.largest_slab_size = slab->size();
        }
        if (stats.smallest_slab_size == 0 ||
            stats.smallest_slab_size > slab->size()) {
            stats.smallest_slab_size = slab->size();
        }
    }

    stats.usage_ratio =
        stats.total_capacity > 0
            ? static_cast<double>(stats.total_used) / stats.total_capacity
            : 0.0;

    return stats;
}

void DynamicSlabAllocator::perform_maintenance() {
    auto now = std::chrono::steady_clock::now();

    // Check for shrinking
    if (config_.enable_shrinking &&
        now - last_shrink_check_ > config_.shrink_delay) {
        if (should_shrink()) {
            shrink_slabs();
        }
        last_shrink_check_ = now;
    }
}

void DynamicSlabAllocator::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto &slab : slabs_) {
        slab->reset();
    }

    // Keep only the initial slab
    if (slabs_.size() > 1) {
        slabs_.resize(1);
        slabs_[0] = std::make_unique<Slab>(config_.initial_slab_size);
    }
}

void DynamicSlabAllocator::grow_slabs() {
    if (slabs_.size() >= config_.max_slabs) {
        return;
    }

    size_t new_size = calculate_next_slab_size();
    slabs_.emplace_back(std::make_unique<Slab>(new_size));
    slab_growths_++;
    last_growth_ = std::chrono::steady_clock::now();

    std::cout << "[DynamicSlabAllocator] Grew to " << slabs_.size()
              << " slabs, new slab size: " << (new_size / (1024 * 1024))
              << " MB" << std::endl;
}

void DynamicSlabAllocator::shrink_slabs() {
    // Find and remove empty slabs (keep at least one)
    auto it =
        std::remove_if(slabs_.begin() + 1, slabs_.end(),
                       [](const auto &slab) { return slab->used() == 0; });

    size_t removed = std::distance(it, slabs_.end());
    if (removed > 0) {
        slabs_.erase(it, slabs_.end());
        slab_shrinks_ += removed;

        std::cout << "[DynamicSlabAllocator] Shrank by " << removed
                  << " slabs, now " << slabs_.size() << " slabs" << std::endl;
    }
}

size_t DynamicSlabAllocator::calculate_next_slab_size() const {
    if (slabs_.empty()) {
        return config_.initial_slab_size;
    }

    // Use the largest existing slab size and grow by factor
    size_t largest = 0;
    for (const auto &slab : slabs_) {
        if (slab->size() > largest) {
            largest = slab->size();
        }
    }

    size_t next_size = static_cast<size_t>(largest * config_.growth_factor);
    next_size =
        std::clamp(next_size, config_.min_slab_size, config_.max_slab_size);

    return next_size;
}

Slab *DynamicSlabAllocator::find_suitable_slab(size_t size) {
    // Find slab with enough space, prefer fuller slabs to reduce fragmentation
    Slab *best = nullptr;
    double best_usage = 0.0;

    for (auto &slab : slabs_) {
        if (slab->available() >= size) {
            double usage = slab->usage_ratio();
            if (!best || usage > best_usage) {
                best = slab.get();
                best_usage = usage;
            }
        }
    }

    return best;
}

bool DynamicSlabAllocator::should_grow() const {
    if (slabs_.size() >= config_.max_slabs) {
        return false;
    }

    // Check overall usage ratio
    size_t total_capacity = 0;
    size_t total_used = 0;

    for (const auto &slab : slabs_) {
        total_capacity += slab->size();
        total_used += slab->used();
    }

    double usage_ratio = total_capacity > 0
                             ? static_cast<double>(total_used) / total_capacity
                             : 1.0;

    return usage_ratio > config_.high_water_mark;
}

bool DynamicSlabAllocator::should_shrink() const {
    if (slabs_.size() <= 1 || !config_.enable_shrinking) {
        return false;
    }

    // Check if we have empty slabs or very low usage
    size_t empty_slabs = 0;
    size_t total_capacity = 0;
    size_t total_used = 0;

    for (const auto &slab : slabs_) {
        if (slab->used() == 0) {
            empty_slabs++;
        }
        total_capacity += slab->size();
        total_used += slab->used();
    }

    double usage_ratio = total_capacity > 0
                             ? static_cast<double>(total_used) / total_capacity
                             : 0.0;

    return empty_slabs > 0 || usage_ratio < config_.low_water_mark;
}

// ThreadLocalSlabAllocator implementation

thread_local std::unique_ptr<DynamicSlabAllocator>
    ThreadLocalSlabAllocator::tls_allocator_;

ThreadLocalSlabAllocator::ThreadLocalSlabAllocator(
    const DynamicSlabConfig &config)
    : config_(config) {}

void *ThreadLocalSlabAllocator::allocate(size_t size, size_t alignment) {
    return get_allocator()->allocate(size, alignment);
}

void ThreadLocalSlabAllocator::deallocate(void *ptr, size_t size) {
    get_allocator()->deallocate(ptr, size);
}

DynamicSlabStats ThreadLocalSlabAllocator::get_stats() const {
    if (tls_allocator_) {
        return tls_allocator_->get_stats();
    }
    return {};
}

DynamicSlabAllocator *ThreadLocalSlabAllocator::get_allocator() {
    if (!tls_allocator_) {
        tls_allocator_ = std::make_unique<DynamicSlabAllocator>(config_);
    }
    return tls_allocator_.get();
}

// GlobalSlabAllocator implementation

GlobalSlabAllocator &GlobalSlabAllocator::instance() {
    static GlobalSlabAllocator instance;
    return instance;
}

GlobalSlabAllocator::GlobalSlabAllocator() {
    DynamicSlabConfig config;
    allocator_ = std::make_unique<DynamicSlabAllocator>(config);
}

GlobalSlabAllocator::~GlobalSlabAllocator() = default;

void *GlobalSlabAllocator::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocator_->allocate(size, alignment);
}

void GlobalSlabAllocator::deallocate(void *ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    allocator_->deallocate(ptr, size);
}

void GlobalSlabAllocator::configure(const DynamicSlabConfig &config) {
    std::lock_guard<std::mutex> lock(mutex_);
    allocator_ = std::make_unique<DynamicSlabAllocator>(config);
}

DynamicSlabStats GlobalSlabAllocator::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocator_->get_stats();
}

} // namespace memory
} // namespace psyne