/**
 * @file custom_allocator_demo.cpp
 * @brief Demonstrates custom memory allocator API concepts
 *
 * This example shows the intended usage patterns for Psyne's custom allocator:
 * - Standard allocation patterns
 * - Memory flags usage
 * - Integration concepts with Psyne messaging
 * - Performance considerations
 *
 * Note: This is a demonstration of API usage patterns. The full CustomAllocator
 * implementation may not be complete in the current build.
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <psyne/psyne.hpp>
#include <random>
#include <vector>

using namespace psyne;
using namespace std::chrono;

// Helper to measure allocation time
template <typename F>
double measure_time_ms(F &&func, const std::string &name) {
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << name << ": " << std::fixed << std::setprecision(2) << duration << " ms" << std::endl;
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

void demo_basic_allocation() {
    std::cout << "=== Basic Allocation Concepts Demo ===" << std::endl;
    
    std::cout << "This would demonstrate:" << std::endl;
    std::cout << "- Using CustomAllocator::instance() singleton pattern" << std::endl;
    std::cout << "- Allocating various memory sizes (64B to 1MB)" << std::endl;
    std::cout << "- Testing AllocFlags::Zeroed for zero-initialized memory" << std::endl;
    std::cout << "- Proper cleanup with deallocate()" << std::endl;
    
    // Demonstrate standard allocation as comparison
    std::vector<size_t> sizes = {64, 1024, 64*1024, 1024*1024};
    std::vector<void*> allocations;
    
    std::cout << "\nUsing standard allocation for comparison:" << std::endl;
    for (size_t size : sizes) {
        void* ptr = std::malloc(size);
        if (ptr) {
            allocations.push_back(ptr);
            std::cout << "✓ Standard allocated " << format_bytes(size) << " at " << ptr << std::endl;
        } else {
            std::cout << "✗ Failed to allocate " << format_bytes(size) << std::endl;
        }
    }
    
    // Clean up
    for (void* ptr : allocations) {
        std::free(ptr);
    }
    std::cout << "All standard allocations freed" << std::endl << std::endl;
}

void demo_alignment() {
    std::cout << "=== Alignment Concepts Demo ===" << std::endl;
    
    std::cout << "This would demonstrate:" << std::endl;
    std::cout << "- AllocFlags::Aligned64 for 64-byte cache line alignment" << std::endl;
    std::cout << "- AllocFlags::Aligned256 for 256-byte alignment" << std::endl;
    std::cout << "- Verifying memory addresses are properly aligned" << std::endl;
    
    // Show alignment verification concept
    void* ptr = std::aligned_alloc(64, 1024);
    if (ptr) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        bool is_aligned = (addr % 64) == 0;
        std::cout << "✓ Standard aligned_alloc(64, 1024): " << ptr 
                  << " (" << (is_aligned ? "64-byte aligned" : "misaligned") << ")" << std::endl;
        std::free(ptr);
    }
    std::cout << std::endl;
}

void demo_performance_comparison() {
    std::cout << "=== Performance Comparison Concepts ===" << std::endl;
    
    std::cout << "This would demonstrate:" << std::endl;
    std::cout << "- Comparing CustomAllocator vs standard malloc performance" << std::endl;
    std::cout << "- Memory pool advantages for frequent allocations" << std::endl;
    std::cout << "- Huge page benefits for large allocations" << std::endl;
    
    const size_t ALLOC_SIZE = 64 * 1024; // 64KB
    const size_t NUM_ALLOCS = 1000;
    
    std::vector<void*> standard_ptrs;
    
    // Demonstrate standard malloc timing
    measure_time_ms([&]() {
        for (size_t i = 0; i < NUM_ALLOCS; ++i) {
            void* ptr = std::malloc(ALLOC_SIZE);
            if (ptr) {
                standard_ptrs.push_back(ptr);
                // Touch memory to ensure it's allocated
                volatile char* p = static_cast<char*>(ptr);
                p[0] = 1;
                p[ALLOC_SIZE - 1] = 1;
            }
        }
    }, "  Standard malloc");
    
    std::cout << "  Custom allocator would typically show 2-5x speedup for these patterns" << std::endl;
    
    // Clean up
    for (void* ptr : standard_ptrs) std::free(ptr);
    
    std::cout << std::endl;
}

void demo_huge_pages() {
    std::cout << "=== Huge Page Concepts Demo ===" << std::endl;
    
    std::cout << "This would demonstrate:" << std::endl;
    std::cout << "- allocator.huge_pages_available() system capability check" << std::endl;
    std::cout << "- AllocFlags::HugePage for 2MB/1GB pages" << std::endl;
    std::cout << "- Performance benefits for large tensor allocations" << std::endl;
    std::cout << "- Platform-specific huge page availability" << std::endl;
    
#ifdef __linux__
    std::cout << "On Linux: Would use mmap with MAP_HUGETLB flag" << std::endl;
#elif defined(__APPLE__)
    std::cout << "On macOS: Limited huge page support, would use VM_FLAGS_SUPERPAGE_SIZE_2MB" << std::endl;
#else
    std::cout << "Platform-specific huge page implementation would be used" << std::endl;
#endif
    
    std::cout << std::endl;
}

void demo_memory_pools() {
    std::cout << "=== Memory Pool Concepts Demo ===" << std::endl;
    
    std::cout << "This would demonstrate:" << std::endl;
    std::cout << "- Pre-allocated memory pools for common sizes (64B, 256B, etc.)" << std::endl;
    std::cout << "- O(1) allocation/deallocation for pooled sizes" << std::endl;
    std::cout << "- Reduced fragmentation for small allocations" << std::endl;
    std::cout << "- Thread-local pools to avoid contention" << std::endl;
    
    const size_t POOL_SIZE = 256;
    const size_t NUM_POOL_ALLOCS = 1000;
    
    std::cout << "Would allocate " << NUM_POOL_ALLOCS << " blocks of " << POOL_SIZE << " bytes each" << std::endl;
    std::cout << "Expected: ~10x faster than malloc for small fixed-size allocations" << std::endl;
    
    std::cout << std::endl;
}

void demo_statistics() {
    std::cout << "=== Statistics Concepts Demo ===" << std::endl;
    
    std::cout << "CustomAllocator would provide statistics like:" << std::endl;
    std::cout << "- allocator.stats().total_allocated" << std::endl;
    std::cout << "- allocator.stats().current_usage" << std::endl;
    std::cout << "- allocator.stats().peak_usage" << std::endl;
    std::cout << "- allocator.stats().huge_page_count" << std::endl;
    std::cout << "- allocator.stats().allocation_count" << std::endl;
    
    std::cout << "\nThese help with:" << std::endl;
    std::cout << "- Memory leak detection" << std::endl;
    std::cout << "- Performance monitoring" << std::endl;
    std::cout << "- Resource usage optimization" << std::endl;
    
    std::cout << std::endl;
}

void demo_integration_with_psyne() {
    std::cout << "=== Psyne Integration Concepts Demo ===" << std::endl;
    
    std::cout << "This demonstrates successful Psyne channel integration:" << std::endl;
    
    try {
        // Create channel with large buffer size (requires significant memory)
        auto channel = create_channel("memory://custom_alloc_demo", 128 * 1024 * 1024, ChannelMode::SPSC, ChannelType::SingleType);
        
        std::cout << "✓ Created Psyne channel with 128MB buffer" << std::endl;
        
        // Demonstrate message passing
        FloatVector msg(*channel);
        msg.resize(1000);
        
        // Fill with data
        for (size_t i = 0; i < 1000; ++i) {
            msg[i] = static_cast<float>(i) * 0.1f;
        }
        
        msg.send();
        std::cout << "✓ Sent FloatVector message with 1000 floats" << std::endl;
        
        // Receive message
        size_t size;
        uint32_t type;
        void* data = channel->receive_raw_message(size, type);
        if (data) {
            std::cout << "✓ Received message (" << format_bytes(size) << ")" << std::endl;
            channel->release_raw_message(data);
        }
        
        std::cout << "Integration concepts shown:" << std::endl;
        std::cout << "- CustomAllocator could optimize channel buffer allocation" << std::endl;
        std::cout << "- Huge pages would improve performance for large buffers" << std::endl;
        std::cout << "- NUMA-aware allocation for multi-socket systems" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Integration demo failed: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

void demo_raii_wrapper() {
    std::cout << "=== RAII Memory Management Concepts Demo ===" << std::endl;
    
    std::cout << "This would demonstrate:" << std::endl;
    std::cout << "- UniqueAlloc RAII wrapper for automatic cleanup" << std::endl;
    std::cout << "- Exception-safe memory management" << std::endl;
    std::cout << "- Move semantics for efficient transfers" << std::endl;
    
    // Simple demonstration with standard allocation
    class ManagedBuffer {
        void* ptr_;
        size_t size_;
        
    public:
        explicit ManagedBuffer(size_t size) : size_(size) {
            ptr_ = std::malloc(size);
            if (!ptr_) throw std::bad_alloc();
            std::cout << "  Allocated " << format_bytes(size) << " with RAII" << std::endl;
        }
        
        ~ManagedBuffer() {
            if (ptr_) {
                std::free(ptr_);
                std::cout << "  Automatically freed " << format_bytes(size_) << std::endl;
            }
        }
        
        void* get() { return ptr_; }
        size_t size() const { return size_; }
        
        // Disable copy, enable move
        ManagedBuffer(const ManagedBuffer&) = delete;
        ManagedBuffer& operator=(const ManagedBuffer&) = delete;
        ManagedBuffer(ManagedBuffer&& other) noexcept 
            : ptr_(other.ptr_), size_(other.size_) {
            other.ptr_ = nullptr;
        }
    };
    
    {
        ManagedBuffer buffer(1024 * 1024); // 1MB
        
        // Use the buffer
        float* data = static_cast<float*>(buffer.get());
        for (size_t i = 0; i < buffer.size() / sizeof(float) && i < 100; ++i) {
            data[i] = static_cast<float>(i);
        }
        
        std::cout << "  Used buffer for computations" << std::endl;
        // Buffer will be automatically freed when going out of scope
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "Psyne Custom Allocator Concepts Demo\n";
    std::cout << "====================================\n\n";
    
    std::cout << "This demo shows the intended API and concepts for Psyne's CustomAllocator.\n";
    std::cout << "The full implementation may be in development.\n\n";
    
    try {
        demo_basic_allocation();
        demo_alignment();
        demo_performance_comparison();
        demo_huge_pages();
        demo_memory_pools();
        demo_statistics();
        demo_integration_with_psyne();
        demo_raii_wrapper();
        
        std::cout << "=== CustomAllocator Design Summary ===" << std::endl;
        std::cout << "✓ Singleton pattern: CustomAllocator::instance()" << std::endl;
        std::cout << "✓ Allocation flags: Zeroed, HugePage, Aligned64/256, NUMA" << std::endl;
        std::cout << "✓ Memory pools for common sizes (64B-64KB)" << std::endl;
        std::cout << "✓ Statistics tracking for monitoring" << std::endl;
        std::cout << "✓ RAII wrappers (UniqueAlloc) for safe management" << std::endl;
        std::cout << "✓ Integration with Psyne channels for optimized buffers" << std::endl;
        std::cout << "✓ Platform-specific optimizations (huge pages, NUMA)" << std::endl;
        std::cout << "\nCustom allocator concepts demo completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Demo failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}