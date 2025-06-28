#pragma once

// Psyne Performance Optimization Suite
// Comprehensive performance optimizations for zero-copy messaging

#include "simd_operations.hpp"
#include "huge_pages.hpp"
#include "cpu_affinity.hpp"
#include "prefetch.hpp"
#include "../psyne.hpp"

namespace psyne {
namespace perf {

// Performance optimization manager
class PerformanceManager {
public:
    struct Config {
        bool enable_simd = true;
        bool enable_huge_pages = true;
        bool enable_numa_affinity = true;
        bool enable_cpu_affinity = true;
        bool enable_prefetching = true;
        bool auto_tune = true;
        
        // SIMD configuration
        bool force_simd_fallback = false;
        
        // Memory configuration
        HugePagePolicy huge_page_policy = HugePagePolicy::TryBest;
        HugePageSize preferred_page_size = HugePageSize::Auto;
        
        // CPU configuration
        ThreadType default_thread_type = ThreadType::HighThroughput;
        bool isolate_performance_threads = false;
        
        // Prefetch configuration
        MessagePrefetchStrategy prefetch_strategy = MessagePrefetchStrategy::Sequential;
        CacheLevel prefetch_target = CacheLevel::Auto;
    };

public:
    PerformanceManager();
    explicit PerformanceManager(const Config& config);
    ~PerformanceManager();
    
    // Initialize performance optimizations
    bool initialize();
    void shutdown();
    
    // Component access
    const CPUFeatures& get_cpu_features() const { return cpu_features_; }
    const HugePageInfo& get_huge_page_info() const { return huge_page_info_; }
    const CPUTopology& get_cpu_topology() const { return cpu_topology_; }
    
    // Auto-tuning
    void auto_tune_for_workload(const std::string& workload_description);
    void benchmark_and_optimize();
    
    // Performance advice
    struct PerformanceAdvice {
        std::vector<std::string> recommendations;
        std::vector<std::string> warnings;
        bool optimal_configuration;
        double estimated_performance_gain;
    };
    
    PerformanceAdvice analyze_system_performance();
    
    // Configuration management
    const Config& get_config() const { return config_; }
    void update_config(const Config& config);
    
    // Performance monitoring
    struct SystemPerformance {
        double memory_bandwidth_gbps;
        double cache_hit_rate;
        double cpu_utilization;
        size_t context_switches_per_sec;
        double numa_efficiency;
        bool using_huge_pages;
        bool simd_accelerated;
    };
    
    SystemPerformance measure_system_performance();

private:
    Config config_;
    bool initialized_;
    
    // System information
    CPUFeatures cpu_features_;
    HugePageInfo huge_page_info_;
    CPUTopology cpu_topology_;
    CacheHierarchy cache_hierarchy_;
    
    // Performance components
    std::unique_ptr<HugePagePool> huge_page_pool_;
    std::unique_ptr<AffinityThreadPool> thread_pool_;
    std::unique_ptr<AdaptivePrefetcher> prefetcher_;
    
    // Internal methods
    void detect_system_capabilities();
    void configure_huge_pages();
    void configure_cpu_affinity();
    void configure_prefetching();
    Config generate_optimal_config();
};

// Global performance manager instance
PerformanceManager& get_performance_manager();

// Convenience functions for quick optimization
bool enable_performance_optimizations(const PerformanceManager::Config& config = {});
void disable_performance_optimizations();

// Quick system analysis
std::string get_performance_summary();
std::vector<std::string> get_performance_recommendations();

// Benchmark utilities
struct BenchmarkResult {
    std::string test_name;
    double baseline_performance;
    double optimized_performance;
    double improvement_ratio;
    std::string units;
    std::vector<std::string> optimizations_used;
};

std::vector<BenchmarkResult> run_performance_benchmarks();

// Message-specific optimization helpers
template<typename MessageType>
class OptimizedMessage {
public:
    explicit OptimizedMessage(Channel& channel) 
        : message_(channel), prefetcher_() {
        
        // Auto-configure based on message type and system capabilities
        configure_for_message_type();
    }
    
    MessageType& message() { return message_; }
    const MessageType& message() const { return message_; }
    
    // Optimized send with prefetching and SIMD
    void send_optimized() {
        // Apply SIMD optimizations if applicable
        apply_simd_optimizations();
        
        // Send the message
        message_.send();
    }
    
    // Get optimization statistics
    struct OptimizationStats {
        bool simd_used;
        bool prefetch_active;
        bool huge_pages_used;
        double estimated_speedup;
    };
    
    OptimizationStats get_optimization_stats() const;

private:
    MessageType message_;
    MessagePrefetcher prefetcher_;
    
    void configure_for_message_type();
    void apply_simd_optimizations();
};

// Factory function for optimized messages
template<typename MessageType>
OptimizedMessage<MessageType> create_optimized_message(Channel& channel) {
    return OptimizedMessage<MessageType>(channel);
}

// Memory-optimized containers
template<typename T>
using OptimizedVector = PrefetchVector<T>;

template<typename T>
using CacheAlignedVector = std::vector<T, CacheAlignedAllocator<T>>;

// High-performance memory allocator
class PerformanceAllocator {
public:
    static void* allocate(size_t size, size_t alignment = 0);
    static void deallocate(void* ptr, size_t size);
    static void* reallocate(void* ptr, size_t old_size, size_t new_size);
    
    // Statistics
    static size_t get_total_allocated();
    static size_t get_allocation_count();
    static double get_allocation_efficiency();
};

} // namespace perf
} // namespace psyne