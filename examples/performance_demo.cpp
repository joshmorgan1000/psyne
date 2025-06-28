// Performance optimization demonstration for Psyne
// Shows how to use SIMD, huge pages, CPU affinity, and prefetching

#include <psyne/psyne.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

using namespace psyne;
using namespace psyne::perf;

// Custom message type for performance testing
class PerformanceMessage : public Message<PerformanceMessage> {
public:
    static constexpr uint32_t message_type = 100;
    
    using Message<PerformanceMessage>::Message;
    
    static size_t calculate_size() {
        return 64 * 1024; // 64KB message
    }
    
    // Get data as float array for SIMD operations
    float* get_data() { 
        return reinterpret_cast<float*>(data()); 
    }
    
    size_t get_element_count() const {
        return size() / sizeof(float);
    }
};

void demonstrate_simd_operations() {
    std::cout << "\n=== SIMD Operations Demo ===\n";
    
    // Check CPU capabilities
    auto features = get_cpu_features();
    std::cout << "CPU Features:\n";
    std::cout << "  AVX2: " << (features.has_avx2 ? "Yes" : "No") << "\n";
    std::cout << "  AVX: " << (features.has_avx ? "Yes" : "No") << "\n";
    std::cout << "  SSE4.1: " << (features.has_sse41 ? "Yes" : "No") << "\n";
    std::cout << "  NEON: " << (features.has_neon ? "Yes" : "No") << "\n";
    
    // Create test vectors
    constexpr size_t size = 1024;
    std::vector<float> a(size, 1.5f);
    std::vector<float> b(size, 2.5f);
    std::vector<float> result(size);
    
    // Benchmark SIMD vs scalar operations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        vector_add_f32(a.data(), b.data(), result.data(), size);
    }
    auto simd_time = std::chrono::high_resolution_clock::now() - start;
    
    // Scalar baseline
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        for (size_t j = 0; j < size; ++j) {
            result[j] = a[j] + b[j];
        }
    }
    auto scalar_time = std::chrono::high_resolution_clock::now() - start;
    
    auto simd_ms = std::chrono::duration_cast<std::chrono::microseconds>(simd_time).count();
    auto scalar_ms = std::chrono::duration_cast<std::chrono::microseconds>(scalar_time).count();
    
    std::cout << "Vector addition (1000 iterations, " << size << " elements):\n";
    std::cout << "  SIMD time: " << simd_ms << " μs\n";
    std::cout << "  Scalar time: " << scalar_ms << " μs\n";
    std::cout << "  Speedup: " << (double)scalar_ms / simd_ms << "x\n";
    
    // Test dot product
    float dot_result = vector_dot_f32(a.data(), b.data(), size);
    std::cout << "Dot product result: " << dot_result << "\n";
}

void demonstrate_huge_pages() {
    std::cout << "\n=== Huge Pages Demo ===\n";
    
    // Get system huge page info
    auto info = get_huge_page_info();
    std::cout << "Huge page support: " << (info.supported ? "Yes" : "No") << "\n";
    std::cout << "Transparent huge pages: " << (info.transparent_enabled ? "Yes" : "No") << "\n";
    std::cout << "Free 2MB pages: " << info.free_2mb_pages << "\n";
    std::cout << "Free 1GB pages: " << info.free_1gb_pages << "\n";
    
    if (!info.supported) {
        std::cout << "Huge pages not supported on this system\n";
        return;
    }
    
    // Allocate memory with huge pages
    size_t buffer_size = 64 * 1024 * 1024; // 64MB
    auto allocation = allocate_huge_pages(buffer_size, HugePagePolicy::TryBest);
    
    if (allocation.ptr) {
        std::cout << "Allocated " << allocation.size << " bytes\n";
        std::cout << "Using huge pages: " << (allocation.is_huge_page ? "Yes" : "No") << "\n";
        std::cout << "Page size: " << allocation.actual_page_size << " bytes\n";
        std::cout << "NUMA local: " << (allocation.is_numa_local ? "Yes" : "No") << "\n";
        
        // Test memory performance
        auto* data = static_cast<float*>(allocation.ptr);
        size_t float_count = allocation.size / sizeof(float);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < float_count; ++i) {
            data[i] = static_cast<float>(i);
        }
        auto write_time = std::chrono::high_resolution_clock::now() - start;
        
        start = std::chrono::high_resolution_clock::now();
        volatile float sum = 0;
        for (size_t i = 0; i < float_count; ++i) {
            sum += data[i];
        }
        auto read_time = std::chrono::high_resolution_clock::now() - start;
        
        auto write_ms = std::chrono::duration_cast<std::chrono::milliseconds>(write_time).count();
        auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(read_time).count();
        
        std::cout << "Memory performance:\n";
        std::cout << "  Write time: " << write_ms << " ms\n";
        std::cout << "  Read time: " << read_ms << " ms\n";
        std::cout << "  Sum: " << sum << "\n";
        
        free_huge_pages(allocation);
    } else {
        std::cout << "Failed to allocate huge pages: " << allocation.error_message << "\n";
    }
}

void demonstrate_cpu_affinity() {
    std::cout << "\n=== CPU Affinity Demo ===\n";
    
    // Get CPU topology
    auto topology = get_cpu_topology();
    std::cout << "CPU Topology:\n";
    std::cout << "  Physical cores: " << topology.physical_cores << "\n";
    std::cout << "  Logical cores: " << topology.logical_cores << "\n";
    std::cout << "  NUMA nodes: " << topology.numa_nodes << "\n";
    std::cout << "  L3 cache size: " << topology.l3_cache_size << " bytes\n";
    
    // Get current thread affinity
    auto current_affinity = get_current_thread_affinity();
    if (!current_affinity.empty()) {
        std::cout << "Current thread runs on cores: ";
        for (int core : current_affinity) {
            std::cout << core << " ";
        }
        std::cout << "\n";
    }
    
    // Get recommended affinity for high-throughput work
    auto config = get_recommended_affinity(ThreadType::HighThroughput);
    std::cout << "Recommended cores for high-throughput: ";
    for (int core : config.core_ids) {
        std::cout << core << " ";
    }
    std::cout << "\n";
    
    // Demonstrate thread pool with affinity
    std::cout << "Creating affinity-aware thread pool...\n";
    AffinityThreadPool pool(4);
    
    // Submit work to specific core types
    auto future1 = pool.submit(ThreadType::HighThroughput, []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return "High-throughput task completed";
    });
    
    auto future2 = pool.submit(ThreadType::LowLatency, []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return "Low-latency task completed";
    });
    
    std::cout << future1.get() << "\n";
    std::cout << future2.get() << "\n";
}

void demonstrate_prefetching() {
    std::cout << "\n=== Memory Prefetching Demo ===\n";
    
    // Get cache information
    auto cache_info = get_cache_hierarchy();
    std::cout << "Cache Hierarchy:\n";
    std::cout << "  L1 data cache: " << cache_info.l1_data.size << " bytes\n";
    std::cout << "  L2 cache: " << cache_info.l2.size << " bytes\n";
    std::cout << "  L3 cache: " << cache_info.l3.size << " bytes\n";
    std::cout << "  Cache line size: " << get_cache_line_size() << " bytes\n";
    
    // Create test data
    constexpr size_t array_size = 1024 * 1024; // 1M elements
    std::vector<int> data(array_size);
    std::iota(data.begin(), data.end(), 0);
    
    // Benchmark with and without prefetching
    auto benchmark_access = [&](bool use_prefetch) {
        auto start = std::chrono::high_resolution_clock::now();
        volatile int sum = 0;
        
        for (size_t i = 0; i < array_size - 64; i += 64) {
            if (use_prefetch) {
                prefetch(&data[i + 64], PrefetchHint::Read, CacheLevel::L2);
            }
            
            // Access current cache line
            for (int j = 0; j < 16; ++j) { // 16 ints per cache line (64 bytes)
                sum += data[i + j];
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    };
    
    auto time_without = benchmark_access(false);
    auto time_with = benchmark_access(true);
    
    std::cout << "Sequential access benchmark (1M elements):\n";
    std::cout << "  Without prefetch: " << time_without << " μs\n";
    std::cout << "  With prefetch: " << time_with << " μs\n";
    std::cout << "  Improvement: " << (double)time_without / time_with << "x\n";
    
    // Demonstrate adaptive prefetcher
    AdaptivePrefetcher prefetcher;
    
    // Train the prefetcher with access pattern
    std::cout << "Training adaptive prefetcher...\n";
    for (size_t i = 0; i < 1000; i += 8) {
        prefetcher.record_access(&data[i]);
    }
    
    // Use prefetcher predictions
    auto predictions = prefetcher.predict_next_accesses(4);
    std::cout << "Predicted next " << predictions.size() << " accesses\n";
    
    auto stats = prefetcher.get_stats();
    std::cout << "Prefetcher stats:\n";
    std::cout << "  Total accesses: " << stats.total_accesses << "\n";
    std::cout << "  Confidence: " << stats.confidence << "\n";
}

void demonstrate_performance_manager() {
    std::cout << "\n=== Performance Manager Demo ===\n";
    
    // Get global performance manager
    auto& manager = get_performance_manager();
    
    // Get system performance analysis
    auto advice = manager.analyze_system_performance();
    std::cout << "Performance Analysis:\n";
    std::cout << "  Optimal configuration: " << (advice.optimal_configuration ? "Yes" : "No") << "\n";
    std::cout << "  Estimated gain: " << advice.estimated_performance_gain << "x\n";
    
    std::cout << "Recommendations:\n";
    for (const auto& rec : advice.recommendations) {
        std::cout << "  • " << rec << "\n";
    }
    
    if (!advice.warnings.empty()) {
        std::cout << "Warnings:\n";
        for (const auto& warning : advice.warnings) {
            std::cout << "  ⚠ " << warning << "\n";
        }
    }
    
    // Measure current system performance
    auto perf = manager.measure_system_performance();
    std::cout << "Current Performance:\n";
    std::cout << "  Memory bandwidth: " << perf.memory_bandwidth_gbps << " GB/s\n";
    std::cout << "  Cache hit rate: " << perf.cache_hit_rate << "\n";
    std::cout << "  CPU utilization: " << perf.cpu_utilization << "\n";
    std::cout << "  Using huge pages: " << (perf.using_huge_pages ? "Yes" : "No") << "\n";
    std::cout << "  SIMD accelerated: " << (perf.simd_accelerated ? "Yes" : "No") << "\n";
}

int main() {
    std::cout << "Psyne Performance Optimization Demo\n";
    std::cout << "===================================\n";
    
    try {
        // Initialize performance optimizations
        PerformanceManager::Config config;
        config.enable_simd = true;
        config.enable_huge_pages = true;
        config.enable_numa_affinity = true;
        config.enable_cpu_affinity = true;
        config.enable_prefetching = true;
        config.auto_tune = true;
        
        if (enable_performance_optimizations(config)) {
            std::cout << "Performance optimizations enabled successfully!\n";
        } else {
            std::cout << "Some performance optimizations failed to initialize\n";
        }
        
        // Run demonstrations
        demonstrate_simd_operations();
        demonstrate_huge_pages();
        demonstrate_cpu_affinity();
        demonstrate_prefetching();
        demonstrate_performance_manager();
        
        std::cout << "\n=== Performance Summary ===\n";
        std::cout << get_performance_summary() << "\n";
        
        auto recommendations = get_performance_recommendations();
        if (!recommendations.empty()) {
            std::cout << "\nFinal Recommendations:\n";
            for (const auto& rec : recommendations) {
                std::cout << "• " << rec << "\n";
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}