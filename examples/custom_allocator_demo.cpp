/**
 * @file custom_allocator_demo.cpp
 * @brief Demonstrates custom memory allocator with huge page support
 */

// Custom allocator functionality should be available through psyne.hpp
#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <random>
#include <vector>

using namespace psyne;
using namespace psyne::memory;
using namespace std::chrono;

// Helper to measure allocation time
template <typename F>
double measure_time_ms(F &&func, const std::string &name) {
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << name << ": " << duration << " ms" << std::endl;
    return duration;
}

// Helper to format bytes
std::string format_bytes(size_t bytes) {
    const char *units[] = {"B", "KB", "MB", "GB"};
    int unit = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024 && unit < 3) {
        size /= 1024;
        unit++;
    }

    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit]);
    return std::string(buffer);
}

int main() {
    std::cout << "Custom Allocator Demo\n";
    std::cout << "====================\n\n";

    auto &allocator = CustomAllocator::instance();

    // Display system info
    std::cout << "System Information:\n";
    std::cout << "  Huge pages available: "
              << (allocator.huge_pages_available() ? "Yes" : "No") << "\n";
    std::cout << "  Huge page size: "
              << format_bytes(allocator.get_huge_page_size()) << "\n";
    std::cout << "  NUMA nodes: " << allocator.get_numa_nodes() << "\n\n";

    // Test 1: Compare allocation speeds
    std::cout << "1. Allocation Speed Test\n";
    const size_t ALLOC_SIZE = 16 * 1024 * 1024; // 16MB
    const size_t NUM_ALLOCS = 100;

    std::vector<void *> standard_ptrs;
    std::vector<void *> custom_ptrs;
    std::vector<void *> huge_ptrs;

    // Standard malloc
    auto malloc_time = measure_time_ms(
        [&]() {
            for (size_t i = 0; i < NUM_ALLOCS; ++i) {
                void *ptr = malloc(ALLOC_SIZE);
                if (ptr) {
                    standard_ptrs.push_back(ptr);
                    // Touch memory to ensure it's allocated
                    volatile char *p = static_cast<char *>(ptr);
                    p[0] = 1;
                    p[ALLOC_SIZE - 1] = 1;
                }
            }
        },
        "  Standard malloc");

    // Custom allocator (normal)
    auto custom_time = measure_time_ms(
        [&]() {
            for (size_t i = 0; i < NUM_ALLOCS; ++i) {
                void *ptr = allocator.allocate(ALLOC_SIZE, AllocFlags::Zeroed);
                if (ptr) {
                    custom_ptrs.push_back(ptr);
                    volatile char *p = static_cast<char *>(ptr);
                    p[0] = 1;
                    p[ALLOC_SIZE - 1] = 1;
                }
            }
        },
        "  Custom allocator");

    // Custom allocator (huge pages)
    auto huge_time = measure_time_ms(
        [&]() {
            for (size_t i = 0; i < NUM_ALLOCS; ++i) {
                void *ptr = allocator.allocate(
                    ALLOC_SIZE, AllocFlags::HugePage | AllocFlags::Zeroed);
                if (ptr) {
                    huge_ptrs.push_back(ptr);
                    volatile char *p = static_cast<char *>(ptr);
                    p[0] = 1;
                    p[ALLOC_SIZE - 1] = 1;
                }
            }
        },
        "  Custom (huge pages)");

    std::cout << "  Speedup vs malloc: " << malloc_time / custom_time << "x\n";
    if (!huge_ptrs.empty()) {
        std::cout << "  Huge page speedup: " << malloc_time / huge_time
                  << "x\n";
    }

    // Clean up
    for (void *ptr : standard_ptrs)
        free(ptr);
    for (void *ptr : custom_ptrs)
        allocator.deallocate(ptr);
    for (void *ptr : huge_ptrs)
        allocator.deallocate(ptr);

    std::cout << "\n2. Memory Pool Performance\n";

    // Test small allocations that use pools
    const size_t SMALL_SIZE = 512;
    const size_t SMALL_COUNT = 10000;

    std::vector<void *> small_ptrs;

    auto pool_time = measure_time_ms(
        [&]() {
            for (size_t i = 0; i < SMALL_COUNT; ++i) {
                void *ptr = allocator.allocate(SMALL_SIZE);
                if (ptr)
                    small_ptrs.push_back(ptr);
            }
        },
        "  Pool allocation");

    auto pool_free_time = measure_time_ms(
        [&]() {
            for (void *ptr : small_ptrs) {
                allocator.deallocate(ptr);
            }
        },
        "  Pool deallocation");

    std::cout << "  Total pool operations: "
              << (SMALL_COUNT * 2) / (pool_time + pool_free_time)
              << " ops/ms\n";

    // Test 3: STL container with custom allocator
    std::cout << "\n3. STL Container Integration\n";

    using CustomVector = std::vector<float, StlCustomAllocator<float>>;
    const size_t VECTOR_SIZE = 10'000'000; // 10M floats

    auto stl_time = measure_time_ms(
        [&]() {
            CustomVector vec;
            vec.reserve(VECTOR_SIZE);
            for (size_t i = 0; i < VECTOR_SIZE; ++i) {
                vec.push_back(static_cast<float>(i));
            }
        },
        "  Custom vector creation");

    auto std_time = measure_time_ms(
        [&]() {
            std::vector<float> vec;
            vec.reserve(VECTOR_SIZE);
            for (size_t i = 0; i < VECTOR_SIZE; ++i) {
                vec.push_back(static_cast<float>(i));
            }
        },
        "  Standard vector creation");

    std::cout << "  Custom allocator overhead: "
              << (stl_time / std_time - 1) * 100 << "%\n";

    // Test 4: RAII wrapper
    std::cout << "\n4. RAII Memory Management\n";
    {
        UniqueAlloc buffer(1024 * 1024,
                           AllocFlags::Zeroed | AllocFlags::Aligned64);
        std::cout << "  Allocated " << format_bytes(buffer.size())
                  << " with RAII\n";

        // Use the buffer
        float *data = static_cast<float *>(buffer.get());
        for (size_t i = 0; i < buffer.size() / sizeof(float); ++i) {
            data[i] = static_cast<float>(i);
        }

        std::cout << "  Buffer will be automatically freed\n";
    }

    // Display final statistics
    std::cout << "\n5. Allocation Statistics\n";
    const auto &stats = allocator.stats();
    std::cout << "  Total allocated: " << format_bytes(stats.total_allocated)
              << "\n";
    std::cout << "  Total freed: " << format_bytes(stats.total_freed) << "\n";
    std::cout << "  Current usage: " << format_bytes(stats.current_usage)
              << "\n";
    std::cout << "  Peak usage: " << format_bytes(stats.peak_usage) << "\n";
    std::cout << "  Allocations: " << stats.allocation_count << "\n";
    std::cout << "  Deallocations: " << stats.free_count << "\n";
    std::cout << "  Huge page allocations: " << stats.huge_page_count << "\n";

    // Test tensor allocation helper
    std::cout << "\n6. Tensor Allocation Helper\n";
    const size_t TENSOR_SIZE = 224 * 224 * 3 * sizeof(float); // ImageNet size

    void *tensor = allocate_tensor(TENSOR_SIZE);
    if (tensor) {
        std::cout << "  Allocated tensor: " << format_bytes(TENSOR_SIZE)
                  << "\n";
        auto block_info = allocator.get_block_info(tensor);
        if (block_info) {
            std::cout << "  Is huge page: "
                      << (block_info->is_huge_page ? "Yes" : "No") << "\n";
        }
        deallocate_tensor(tensor);
    }

    return 0;
}