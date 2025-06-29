#include <psyne/psyne.hpp>
#include <psyne/memory/dynamic_slab_allocator.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <random>
#include <cstring>

using namespace psyne;
using namespace std::chrono_literals;

void demonstrate_dynamic_slab() {
    std::cout << "\n=== Dynamic Slab Allocator Demo ===" << std::endl;
    
    memory::DynamicSlabConfig config;
    config.initial_slab_size = 64 * 1024 * 1024;  // Start with 64MB
    config.max_slab_size = 1024 * 1024 * 1024;    // Max 1GB
    config.high_water_mark = 0.75;                // Grow at 75% usage
    config.low_water_mark = 0.25;                 // Shrink below 25% usage
    config.growth_factor = 2.0;                   // Double size when growing
    
    memory::DynamicSlabAllocator allocator(config);
    
    // Allocate increasingly larger chunks
    std::vector<std::pair<void*, size_t>> allocations;
    size_t sizes[] = {64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024};
    
    for (int round = 0; round < 3; ++round) {
        std::cout << "\nRound " << (round + 1) << ":" << std::endl;
        
        for (size_t size : sizes) {
            void* ptr = allocator.allocate(size);
            if (ptr) {
                allocations.push_back({ptr, size});
                auto stats = allocator.get_stats();
                std::cout << "  Allocated " << (size / (1024 * 1024)) << " MB"
                         << " (slabs: " << stats.num_slabs
                         << ", usage: " << std::fixed << std::setprecision(1) 
                         << (stats.usage_ratio * 100) << "%"
                         << ", capacity: " << (stats.total_capacity / (1024 * 1024)) << " MB)" << std::endl;
            }
        }
    }
    
    auto peak_stats = allocator.get_stats();
    std::cout << "\nPeak statistics:" << std::endl;
    std::cout << "  Total slabs: " << peak_stats.num_slabs << std::endl;
    std::cout << "  Total capacity: " << (peak_stats.total_capacity / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Total used: " << (peak_stats.total_used / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Smallest slab: " << (peak_stats.smallest_slab_size / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Largest slab: " << (peak_stats.largest_slab_size / (1024 * 1024)) << " MB" << std::endl;
    
    // Deallocate half to demonstrate shrinking
    std::cout << "\nDeallocating half of allocations..." << std::endl;
    for (size_t i = 0; i < allocations.size(); i += 2) {
        allocator.deallocate(allocations[i].first, allocations[i].second);
    }
    
    allocator.perform_maintenance();
    
    auto after_dealloc_stats = allocator.get_stats();
    std::cout << "\nAfter deallocation:" << std::endl;
    std::cout << "  Total slabs: " << after_dealloc_stats.num_slabs << std::endl;
    std::cout << "  Total capacity: " << (after_dealloc_stats.total_capacity / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Total used: " << (after_dealloc_stats.total_used / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Usage ratio: " << std::fixed << std::setprecision(1) 
              << (after_dealloc_stats.usage_ratio * 100) << "%" << std::endl;
    std::cout << "  Number of growths: " << after_dealloc_stats.slab_growths << std::endl;
    std::cout << "  Number of shrinks: " << after_dealloc_stats.slab_shrinks << std::endl;
}

void demonstrate_thread_local_allocator() {
    std::cout << "\n=== Thread-Local Allocator Demo ===" << std::endl;
    
    memory::DynamicSlabConfig config;
    config.initial_slab_size = 8 * 1024 * 1024;  // 8MB per thread
    config.high_water_mark = 0.8;
    config.enable_shrinking = false;  // Disable shrinking for thread-local
    
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::atomic<size_t> total_allocated{0};
    
    auto worker = [&config, &total_allocated](int thread_id) {
        memory::ThreadLocalSlabAllocator allocator(config);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> size_dist(1024, 64 * 1024);
        
        size_t local_allocated = 0;
        std::vector<std::pair<void*, size_t>> allocations;
        
        // Perform allocations
        for (int i = 0; i < 100; ++i) {
            size_t size = size_dist(gen);
            void* ptr = allocator.allocate(size);
            if (ptr) {
                allocations.push_back({ptr, size});
                local_allocated += size;
                // Write pattern
                std::memset(ptr, thread_id, size);
            }
            
            // Occasionally deallocate
            if (i % 10 == 5 && !allocations.empty()) {
                size_t idx = gen() % allocations.size();
                allocator.deallocate(allocations[idx].first, allocations[idx].second);
                allocations.erase(allocations.begin() + idx);
            }
        }
        
        auto stats = allocator.get_stats();
        std::cout << "  Thread " << thread_id << ": "
                  << "allocated " << (local_allocated / 1024) << " KB, "
                  << "slabs: " << stats.num_slabs << ", "
                  << "usage: " << std::fixed << std::setprecision(1)
                  << (stats.usage_ratio * 100) << "%" << std::endl;
        
        total_allocated += local_allocated;
        
        // Clean up
        for (auto& [ptr, size] : allocations) {
            allocator.deallocate(ptr, size);
        }
    };
    
    // Launch worker threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "\nTotal allocated across all threads: " 
              << (total_allocated.load() / (1024 * 1024)) << " MB" << std::endl;
}

void demonstrate_scoped_allocator() {
    std::cout << "\n=== Scoped Allocator Demo ===" << std::endl;
    std::cout << "Demonstrates automatic cleanup with RAII pattern" << std::endl;
    
    memory::DynamicSlabConfig config;
    config.initial_slab_size = 16 * 1024 * 1024;  // 16MB
    
    memory::DynamicSlabAllocator main_allocator(config);
    
    {
        // Scoped allocator automatically cleans up
        memory::ScopedSlabAllocator scoped(main_allocator);
        
        std::cout << "\nAllocating within scope..." << std::endl;
        
        // Allocate various sizes
        void* small = scoped.allocate(1024);
        void* medium = scoped.allocate(64 * 1024);
        void* large = scoped.allocate(1024 * 1024);
        
        if (small && medium && large) {
            std::cout << "  Successfully allocated 1KB, 64KB, and 1MB" << std::endl;
            
            // Use the memory
            std::memset(small, 0xAA, 1024);
            std::memset(medium, 0xBB, 64 * 1024);
            std::memset(large, 0xCC, 1024 * 1024);
        }
        
        auto stats = main_allocator.get_stats();
        std::cout << "  Main allocator usage: " << (stats.total_used / 1024) << " KB" << std::endl;
        
        // Manual deallocation is possible but not required
        if (medium) {
            scoped.deallocate(medium, 64 * 1024);
            std::cout << "  Manually deallocated 64KB" << std::endl;
        }
        
    } // Automatic cleanup happens here
    
    auto final_stats = main_allocator.get_stats();
    std::cout << "\nAfter scope exit:" << std::endl;
    std::cout << "  Main allocator usage: " << final_stats.total_used << " bytes" << std::endl;
    std::cout << "  All memory automatically cleaned up!" << std::endl;
}

void demonstrate_global_allocator() {
    std::cout << "\n=== Global Allocator Demo ===" << std::endl;
    
    // Configure the global allocator
    memory::DynamicSlabConfig config;
    config.initial_slab_size = 64 * 1024 * 1024;  // 64MB
    config.high_water_mark = 0.75;
    config.low_water_mark = 0.25;
    
    memory::GlobalSlabAllocator::instance().configure(config);
    
    // Multiple threads using the global allocator
    const int num_threads = 3;
    std::vector<std::thread> threads;
    
    auto worker = [](int thread_id) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> size_dist(1024, 256 * 1024);
        
        for (int i = 0; i < 50; ++i) {
            size_t size = size_dist(gen);
            void* ptr = memory::GlobalSlabAllocator::instance().allocate(size);
            
            if (ptr) {
                // Use the memory
                std::memset(ptr, thread_id, size);
                
                // Simulate work
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                
                // Deallocate
                memory::GlobalSlabAllocator::instance().deallocate(ptr, size);
            }
            
            if (i % 10 == 0) {
                auto stats = memory::GlobalSlabAllocator::instance().get_stats();
                std::cout << "  Thread " << thread_id << " at iteration " << i 
                          << ": " << stats.num_slabs << " slabs, "
                          << (stats.usage_ratio * 100) << "% usage" << std::endl;
            }
        }
    };
    
    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }
    
    auto final_stats = memory::GlobalSlabAllocator::instance().get_stats();
    std::cout << "\nFinal global allocator statistics:" << std::endl;
    std::cout << "  Total allocations: " << final_stats.allocations << std::endl;
    std::cout << "  Total deallocations: " << final_stats.deallocations << std::endl;
    std::cout << "  Slab growths: " << final_stats.slab_growths << std::endl;
    std::cout << "  Slab shrinks: " << final_stats.slab_shrinks << std::endl;
}

int main() {
    std::cout << "Dynamic Memory Allocation Demo" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "Demonstrating Psyne's adaptive slab allocator with 64MB-1GB growth" << std::endl;
    
    try {
        demonstrate_dynamic_slab();
        demonstrate_thread_local_allocator();
        demonstrate_scoped_allocator();
        demonstrate_global_allocator();
        
        std::cout << "\nDemo completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}