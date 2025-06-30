/**
 * @file custom_allocator_demo.cpp
 * @brief Demonstrates custom memory allocator with performance optimizations
 *
 * This example shows how to use Psyne's custom allocator for:
 * - High-performance memory allocation
 * - Huge page support (Linux)
 * - NUMA-aware allocation
 * - Memory pool optimization
 * - Zero-copy buffer management
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

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
    std::cout << "=== Basic Allocation Demo ===" << std::endl;
    
    CustomAllocator allocator;
    std::cout << "Created custom allocator instance" << std::endl;
    
    // Test various allocation sizes
    std::vector<size_t> sizes = {64, 1024, 64*1024, 1024*1024};
    std::vector<void*> allocations;
    
    for (size_t size : sizes) {
        void* ptr = allocator.allocate(size, AllocFlags::Zeroed);
        if (ptr) {
            allocations.push_back(ptr);
            std::cout << "✓ Allocated " << format_bytes(size) << " at " << ptr << std::endl;
            
            // Verify zero initialization
            uint8_t* bytes = static_cast<uint8_t*>(ptr);
            bool is_zeroed = true;
            for (size_t i = 0; i < std::min(size, size_t(100)); ++i) {
                if (bytes[i] != 0) {
                    is_zeroed = false;
                    break;
                }
            }
            std::cout << "  Memory is " << (is_zeroed ? "zeroed" : "not zeroed") << std::endl;
        } else {
            std::cout << "✗ Failed to allocate " << format_bytes(size) << std::endl;
        }
    }
    
    // Clean up
    for (void* ptr : allocations) {
        allocator.deallocate(ptr);
    }
    std::cout << "All allocations freed" << std::endl << std::endl;
}

void demo_alignment() {
    std::cout << "=== Alignment Demo ===" << std::endl;
    
    CustomAllocator allocator;
    
    // Test different alignments
    std::vector<size_t> alignments = {16, 64, 4096};
    
    for (size_t alignment : alignments) {
        void* ptr = allocator.allocate(1024, AllocFlags::None, alignment);
        if (ptr) {
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            bool is_aligned = (addr % alignment) == 0;
            std::cout << "✓ " << alignment << "-byte aligned allocation: " << ptr 
                      << " (" << (is_aligned ? "aligned" : "misaligned") << ")" << std::endl;
            allocator.deallocate(ptr);
        } else {
            std::cout << "✗ Failed " << alignment << "-byte aligned allocation" << std::endl;
        }
    }
    std::cout << std::endl;
}

void demo_performance_comparison() {
    std::cout << "=== Performance Comparison ===" << std::endl;
    
    CustomAllocator allocator;
    const size_t ALLOC_SIZE = 64 * 1024; // 64KB
    const size_t NUM_ALLOCS = 1000;
    
    std::vector<void*> standard_ptrs;
    std::vector<void*> custom_ptrs;
    
    // Standard malloc
    auto malloc_time = measure_time_ms([&]() {
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
    
    // Custom allocator
    auto custom_time = measure_time_ms([&]() {
        for (size_t i = 0; i < NUM_ALLOCS; ++i) {
            void* ptr = allocator.allocate(ALLOC_SIZE);
            if (ptr) {
                custom_ptrs.push_back(ptr);
                volatile char* p = static_cast<char*>(ptr);
                p[0] = 1;
                p[ALLOC_SIZE - 1] = 1;
            }
        }
    }, "  Custom allocator");
    
    if (malloc_time > 0 && custom_time > 0) {
        double speedup = malloc_time / custom_time;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    // Clean up
    for (void* ptr : standard_ptrs) std::free(ptr);
    for (void* ptr : custom_ptrs) allocator.deallocate(ptr);
    
    std::cout << std::endl;
}

void demo_huge_pages() {
    std::cout << "=== Huge Page Demo ===" << std::endl;
    
    CustomAllocator allocator;
    
#ifdef __linux__
    // Try huge page allocation
    const size_t HUGE_SIZE = 4 * 1024 * 1024; // 4MB
    void* huge_ptr = allocator.allocate(HUGE_SIZE, AllocFlags::HugePage);
    
    if (huge_ptr) {
        std::cout << "✓ Allocated " << format_bytes(HUGE_SIZE) << " with huge pages" << std::endl;
        
        // Touch the memory
        volatile char* p = static_cast<char*>(huge_ptr);
        p[0] = 1;
        p[HUGE_SIZE - 1] = 1;
        
        allocator.deallocate(huge_ptr);
        std::cout << "✓ Freed huge page allocation" << std::endl;
    } else {
        std::cout << "✗ Huge page allocation failed (may not be available)" << std::endl;
    }
#else
    std::cout << "Huge pages not supported on this platform" << std::endl;
#endif
    
    std::cout << std::endl;
}

void demo_memory_pools() {
    std::cout << "=== Memory Pool Demo ===" << std::endl;
    
    CustomAllocator allocator;
    
    // Test pool allocation for small sizes
    const size_t POOL_SIZE = 256;
    const size_t NUM_POOL_ALLOCS = 1000;
    
    std::vector<void*> pool_ptrs;
    
    auto pool_time = measure_time_ms([&]() {
        for (size_t i = 0; i < NUM_POOL_ALLOCS; ++i) {
            void* ptr = allocator.allocate(POOL_SIZE);
            if (ptr) {
                pool_ptrs.push_back(ptr);
            }
        }
    }, "  Pool allocation");
    
    auto pool_free_time = measure_time_ms([&]() {
        for (void* ptr : pool_ptrs) {
            allocator.deallocate(ptr);
        }
    }, "  Pool deallocation");
    
    if (pool_time > 0 && pool_free_time > 0) {
        double total_time = pool_time + pool_free_time;
        double ops_per_ms = (NUM_POOL_ALLOCS * 2) / total_time;
        std::cout << "  Total operations: " << std::fixed << std::setprecision(0) 
                  << ops_per_ms << " ops/ms" << std::endl;
    }
    
    std::cout << std::endl;
}

void demo_statistics() {
    std::cout << "=== Allocation Statistics ===" << std::endl;
    
    CustomAllocator allocator;
    
    // Perform various allocations
    std::vector<void*> ptrs;
    std::vector<size_t> sizes = {64, 1024, 64*1024};
    
    for (size_t size : sizes) {
        void* ptr = allocator.allocate(size);
        if (ptr) {
            ptrs.push_back(ptr);
        }
    }
    
    // Display statistics
    const auto& stats = allocator.get_stats();
    std::cout << "Allocation Statistics:" << std::endl;
    std::cout << "  Total allocated: " << format_bytes(stats.total_allocated) << std::endl;
    std::cout << "  Total freed: " << format_bytes(stats.total_freed) << std::endl;
    std::cout << "  Current usage: " << format_bytes(stats.current_usage) << std::endl;
    std::cout << "  Peak usage: " << format_bytes(stats.peak_usage) << std::endl;
    std::cout << "  Allocation count: " << stats.allocation_count << std::endl;
    std::cout << "  Free count: " << stats.free_count << std::endl;
    std::cout << "  Huge page count: " << stats.huge_page_count << std::endl;
    
    // Clean up
    for (void* ptr : ptrs) {
        allocator.deallocate(ptr);
    }
    
    std::cout << std::endl;
}

void demo_integration_with_psyne() {
    std::cout << "=== Psyne Integration Demo ===" << std::endl;
    
    // Show how custom allocator can work with Psyne channels
    try {
        // Create channel with custom buffer size
        auto channel = Channel::get_or_create<FloatVector>("memory://custom_alloc_demo", 1024 * 1024);
        
        std::cout << "✓ Created Psyne channel with custom buffer size" << std::endl;
        
        // Demonstrate message passing
        FloatVector msg(*channel);
        msg.resize(1000);
        
        // Fill with data
        for (size_t i = 0; i < 1000; ++i) {
            msg[i] = static_cast<float>(i) * 0.1f;
        }
        
        msg.send();
        std::cout << "✓ Sent message with 1000 floats" << std::endl;
        
        // Receive message
        size_t size;
        uint32_t type;
        void* data = channel->receive_message(size, type);
        if (data) {
            std::cout << "✓ Received message (" << format_bytes(size) << ")" << std::endl;
            channel->release_message(data);
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ Integration demo failed: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

void demo_raii_wrapper() {
    std::cout << "=== RAII Memory Management Demo ===" << std::endl;
    
    // Simple RAII wrapper example
    class ManagedBuffer {
        CustomAllocator& allocator_;
        void* ptr_;
        size_t size_;
        
    public:
        ManagedBuffer(CustomAllocator& alloc, size_t size) 
            : allocator_(alloc), size_(size) {
            ptr_ = allocator_.allocate(size, AllocFlags::Zeroed);
            if (!ptr_) {
                throw std::bad_alloc();
            }
            std::cout << "  Allocated " << format_bytes(size) << " with RAII" << std::endl;
        }
        
        ~ManagedBuffer() {
            if (ptr_) {
                allocator_.deallocate(ptr_);
                std::cout << "  Automatically freed " << format_bytes(size_) << std::endl;
            }
        }
        
        void* get() { return ptr_; }
        size_t size() const { return size_; }
        
        // Disable copy, enable move
        ManagedBuffer(const ManagedBuffer&) = delete;
        ManagedBuffer& operator=(const ManagedBuffer&) = delete;
        ManagedBuffer(ManagedBuffer&& other) noexcept 
            : allocator_(other.allocator_), ptr_(other.ptr_), size_(other.size_) {
            other.ptr_ = nullptr;
        }
    };
    
    CustomAllocator allocator;
    
    {
        ManagedBuffer buffer(allocator, 1024 * 1024); // 1MB
        
        // Use the buffer
        float* data = static_cast<float*>(buffer.get());
        for (size_t i = 0; i < buffer.size() / sizeof(float); ++i) {
            data[i] = static_cast<float>(i);
        }
        
        std::cout << "  Used buffer for computations" << std::endl;
        // Buffer will be automatically freed when going out of scope
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "Custom Allocator Demo\n";
    std::cout << "====================\n\n";
    
    try {
        demo_basic_allocation();
        demo_alignment();
        demo_performance_comparison();
        demo_huge_pages();
        demo_memory_pools();
        demo_statistics();
        demo_integration_with_psyne();
        demo_raii_wrapper();
        
        std::cout << "=== Summary ===" << std::endl;
        std::cout << "✓ Basic allocation with flags (zeroed, aligned)" << std::endl;
        std::cout << "✓ Performance comparison vs standard malloc" << std::endl;
        std::cout << "✓ Huge page support (platform dependent)" << std::endl;
        std::cout << "✓ Memory pool optimization for small allocations" << std::endl;
        std::cout << "✓ Allocation statistics and monitoring" << std::endl;
        std::cout << "✓ Integration with Psyne channels" << std::endl;
        std::cout << "✓ RAII memory management patterns" << std::endl;
        std::cout << "\nCustom allocator demo completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Demo failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}