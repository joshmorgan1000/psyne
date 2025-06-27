#pragma once

#if USE_VULKAN
#include <vulkan/vulkan.h>
#endif
#include "utils.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>
#include "gpu/VulkanContext.hpp"
#include "gpu/ShaderBuffer.hpp"
#include "gpu/ShaderBufferRing.hpp"

namespace psyne {

class VulkanContext {
public:
    static VulkanContext& getInstance() {
        static VulkanContext instance;
        return instance;
    }
    bool isGPUCapable() const { return false; }
    uint8_t* allocateGPUArena(size_t) { return nullptr; }
};

constexpr size_t DEFAULT_PAGE_SIZE = 4096;
constexpr size_t GPU_ARENA_SIZE = 4ULL * 1024 * 1024 * 1024; // 4GB

struct MemRef {
    uint32_t page_id;
    uint32_t offset;
    uint32_t length;

    bool is_valid() const { return length > 0; }
};

enum class MemVecBackend {
    CPU,
    GPU,
    SHARED,
};

struct Page {
    uint32_t id;
    uint8_t* data; // Either GPU-mapped pointer or CPU-allocated
    size_t capacity;
    size_t used;
    size_t gpu_offset;
    MemVecBackend backend;
    std::function<void(uint8_t*)> free_fn;
    std::atomic<uint32_t> ref_count{0};
    bool marked_free = false;

    bool has_space(size_t len, size_t align = alignof(std::max_align_t)) const {
        size_t aligned = (used + align - 1) & ~(align - 1);
        return aligned + len <= capacity;
    }

    MemRef allocate(size_t len, size_t align = alignof(std::max_align_t)) {
        size_t aligned = (used + align - 1) & ~(align - 1);
        if (aligned + len > capacity)
            throw std::bad_alloc();
        used = aligned + len;
        ref_count++;
        return MemRef{id, static_cast<uint32_t>(aligned), static_cast<uint32_t>(len)};
    }

    uint8_t* get_ptr(const MemRef& ref) {
        if (ref.offset + ref.length > capacity)
            throw std::out_of_range("Invalid MemRef");
        return data + ref.offset;
    }

    void reset_usage() {
        used = 0;
        ref_count = 0;
        marked_free = false;
    }
};

struct PageZone {
    size_t page_size;
    size_t start_offset;
    size_t end_offset;
    size_t current_offset;
    bool grows_upward;
    std::vector<size_t> free_offsets;

    PageZone(size_t ps, size_t start, size_t end, bool up)
        : page_size(ps), start_offset(start), end_offset(end), current_offset(up ? start : end),
          grows_upward(up) {}

    bool has_space() const {
        if (grows_upward)
            return current_offset + page_size <= end_offset;
        else
            return current_offset >= start_offset + page_size;
    }

    size_t alloc_offset() {
        if (!free_offsets.empty()) {
            size_t off = free_offsets.back();
            free_offsets.pop_back();
            return off;
        }
        if (!has_space())
            return SIZE_MAX;

        size_t off = current_offset;
        grows_upward ? current_offset += page_size : current_offset -= page_size;
        return off;
    }

    void free_offset(size_t offset) { free_offsets.push_back(offset); }
};

struct GpuArena {
    uint8_t* base_ptr = nullptr;
    size_t capacity = 0;
    size_t used = 0;
    std::vector<PageZone> zones;
    VulkanContext* ctx = nullptr;

    explicit GpuArena(VulkanContext* ctx) : ctx(ctx) {}

    void initialize(size_t size) {
        base_ptr = ctx->allocateGPUArena(size);
        if (!base_ptr)
            throw std::runtime_error("Failed to allocate GPU arena");
        capacity = size;
        used = 0;
        initialize_zones();
    }

    void initialize_zones() {
        zones.clear();
        size_t quarter = capacity / 4;

        zones.emplace_back(4096, 0 * quarter, 1 * quarter, true);       // 4K pages
        zones.emplace_back(16384, 2 * quarter - 1, 1 * quarter, false); // 16K pages
        zones.emplace_back(65536, 2 * quarter, 3 * quarter, true);      // 64K pages
        zones.emplace_back(131072, 4 * quarter - 1, 3 * quarter,
                           false); // 128K pages
    }

    PageZone* find_zone_for_size(size_t size) {
        for (auto& z : zones) {
            if (z.page_size >= size)
                return &z;
        }
        return nullptr; // fallback: allocation failed
    }

    void* alloc(size_t size, size_t align = alignof(std::max_align_t)) {
        PageZone* zone = find_zone_for_size(size);
        if (!zone)
            return nullptr;
        size_t offset = zone->alloc_offset();
        if (offset == SIZE_MAX || offset + zone->page_size > capacity)
            return nullptr;

        return base_ptr + offset;
    }

    void free(size_t offset, size_t size) {
        PageZone* zone = find_zone_for_size(size);
        if (zone) {
            zone->free_offset(offset);
        }
    }

    void reset() { used = 0; }

    void shutdown() {
        base_ptr = nullptr;
        used = 0;
        capacity = 0;
    }

    ~GpuArena() { shutdown(); }
};

class MemVec {
    std::vector<std::unique_ptr<Page>> pages;
    std::vector<Page*> free_pages;

    uint32_t next_page_id = 0;
    size_t default_page_size = DEFAULT_PAGE_SIZE;
    Page* current_page = nullptr;

    VulkanContext* ctx = nullptr;
    std::unique_ptr<GpuArena> gpu_arena;

    MemVec() {
        getGlobalContext().thread_context = "MemVec";
        ctx = &VulkanContext::getInstance();
        use_gpu = ctx->isGPUCapable();
        if (use_gpu) {
            gpu_arena = std::make_unique<GpuArena>(ctx);
            gpu_arena->initialize(GPU_ARENA_SIZE);
            log_info("Memory pool initialized; " + std::to_string(GPU_ARENA_SIZE) +
                     " bytes reserved on GPU.");
        } else {
            log_info("Memory pool initialized; GPU not available, using CPU memory.");
        }
    }

    Page* allocate_new_page(MemVecBackend backend, size_t min_size) {
        size_t size = std::max(min_size, default_page_size);
        size = ((size + 4095) / 4096) * 4096;

        uint8_t* buffer = nullptr;
        std::function<void(uint8_t*)> deleter;
        size_t offset_in_gpu = 0;

        if (backend == MemVecBackend::GPU && use_gpu) {
            void* gpu_ptr = gpu_arena->alloc(size, 64);
            if (!gpu_ptr) {
                log_error("GPU zone allocation failed.");
                return nullptr;
            }
            offset_in_gpu = static_cast<uint8_t*>(gpu_ptr) - gpu_arena->base_ptr;
            if (offset_in_gpu == SIZE_MAX) {
                log_error("GPU zone allocation failed.");
                return nullptr;
            }
            buffer = gpu_arena->base_ptr + offset_in_gpu;
            deleter = [](uint8_t*) {}; // Vulkan managed
        } else {
            buffer = static_cast<uint8_t*>(std::aligned_alloc(64, size));
            if (!buffer) {
                log_error("CPU memory allocation failed.");
                return nullptr;
            }
            deleter = [](uint8_t* ptr) { std::free(ptr); };
        }

        auto page = std::make_unique<Page>();
        page->id = next_page_id++;
        page->data = buffer;
        page->capacity = size;
        page->used = 0;
        page->gpu_offset = offset_in_gpu;
        page->backend = backend;
        page->free_fn = deleter;

        Page* raw = page.get();
        pages.push_back(std::move(page));
        return raw;
    }

    Page* find_reusable_page(MemVecBackend backend, size_t size, size_t align) {
        for (auto* page : free_pages) {
            if (page->backend == backend && page->capacity >= size &&
                page->has_space(size, align)) {
                page->reset_usage();
                free_pages.erase(std::remove(free_pages.begin(), free_pages.end(), page),
                                 free_pages.end());
                return page;
            }
        }
        return nullptr;
    }

    void dump_stats() const {
        if (gpu_arena) {
            for (const auto& z : gpu_arena->zones) {
                std::cout << "[GPU Zone] " << z.page_size
                          << " bytes: " << (z.grows_upward ? "↑" : "↓") << " used="
                          << (z.grows_upward ? z.current_offset - z.start_offset
                                             : z.end_offset - z.current_offset)
                          << " free=" << z.free_offsets.size() << "\n";
            }
        } else {
            std::cout << "GPU arena is not initialized.\n";
        }
    }

  public:
    bool use_gpu;
    MemVec(const MemVec&) = delete;
    MemVec& operator=(const MemVec&) = delete;

    static MemVec& instance() {
        static MemVec instance;
        return instance;
    }

    MemRef allocate(size_t size, size_t align = alignof(std::max_align_t),
                    MemVecBackend backend = MemVecBackend::CPU) {
        if (backend == MemVecBackend::CPU && use_gpu)
            backend = MemVecBackend::GPU;

        if (!current_page || !current_page->has_space(size, align) ||
            current_page->backend != backend) {
            current_page = find_reusable_page(backend, size, align);
            if (!current_page) {
                current_page = allocate_new_page(backend, size);
            }
        }
        return current_page->allocate(size, align);
    }

    void release(const MemRef& ref) {
        for (auto& page : pages) {
            if (page->id == ref.page_id) {
                if (--page->ref_count == 0) {
                    page->used = 0;
                    page->marked_free = true;
                    free_pages.push_back(page.get());
                }
                return;
            }
        }
        log_warn("Tried to release MemRef for unknown page_id: " + std::to_string(ref.page_id));
    }

    uint8_t* resolve(const MemRef& ref) {
        for (auto& page : pages) {
            if (page->id == ref.page_id) {
                return page->get_ptr(ref);
            }
        }
        throw std::runtime_error("Page not found for MemRef");
    }

    void reset() {
        for (auto& page : pages) {
            page->reset_usage();
        }
        current_page = !pages.empty() ? pages.front().get() : nullptr;
        free_pages.clear();
        if (gpu_arena)
            gpu_arena->reset();
    }
};

} // namespace psyne