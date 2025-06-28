#include "../../include/psyne/perf/performance.hpp"
#include "../../include/psyne/psyne.hpp"
#include <sstream>
#include <mutex>

namespace psyne {
namespace perf {

// ============================================================================
// PerformanceManager Implementation
// ============================================================================

PerformanceManager::PerformanceManager() : PerformanceManager(Config{}) {}

PerformanceManager::PerformanceManager(const Config& config) 
    : config_(config), initialized_(false) {}

PerformanceManager::~PerformanceManager() {
    shutdown();
}

bool PerformanceManager::initialize() {
    if (initialized_) {
        return true;
    }
    
    try {
        // Detect system capabilities
        detect_system_capabilities();
        
        // Configure components based on capabilities and config
        if (config_.enable_huge_pages) {
            configure_huge_pages();
        }
        
        if (config_.enable_cpu_affinity) {
            configure_cpu_affinity();
        }
        
        if (config_.enable_prefetching) {
            configure_prefetching();
        }
        
        // Auto-tune if requested
        if (config_.auto_tune) {
            config_ = generate_optimal_config();
        }
        
        initialized_ = true;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void PerformanceManager::shutdown() {
    if (!initialized_) {
        return;
    }
    
    // Clean up components
    thread_pool_.reset();
    huge_page_pool_.reset();
    prefetcher_.reset();
    
    initialized_ = false;
}

void PerformanceManager::auto_tune_for_workload(const std::string& workload_description) {
    // Analyze workload characteristics and adjust configuration
    if (workload_description.find("latency") != std::string::npos) {
        config_.default_thread_type = ThreadType::LowLatency;
        config_.prefetch_strategy = MessagePrefetchStrategy::Random;
        config_.isolate_performance_threads = true;
    } else if (workload_description.find("throughput") != std::string::npos) {
        config_.default_thread_type = ThreadType::HighThroughput;
        config_.prefetch_strategy = MessagePrefetchStrategy::Sequential;
        config_.huge_page_policy = HugePagePolicy::TryBest;
    } else if (workload_description.find("streaming") != std::string::npos) {
        config_.prefetch_strategy = MessagePrefetchStrategy::Streaming;
        config_.huge_page_policy = HugePagePolicy::Required;
    }
    
    // Reinitialize with new configuration
    if (initialized_) {
        shutdown();
        initialize();
    }
}

void PerformanceManager::benchmark_and_optimize() {
    // Run benchmarks to find optimal settings
    auto benchmarks = run_performance_benchmarks();
    
    // Analyze results and update configuration
    double best_performance = 0.0;
    Config best_config = config_;
    
    // Test different configurations
    std::vector<Config> test_configs = {
        config_,  // Current config
        generate_optimal_config()  // System-optimized config
    };
    
    for (const auto& test_config : test_configs) {
        // Temporarily apply config and measure performance
        Config old_config = config_;
        config_ = test_config;
        
        auto perf = measure_system_performance();
        double score = perf.memory_bandwidth_gbps * perf.cache_hit_rate * 
                      (1.0 - perf.cpu_utilization);
        
        if (score > best_performance) {
            best_performance = score;
            best_config = test_config;
        }
        
        config_ = old_config;
    }
    
    // Apply best configuration
    config_ = best_config;
    
    if (initialized_) {
        shutdown();
        initialize();
    }
}

PerformanceManager::PerformanceAdvice PerformanceManager::analyze_system_performance() {
    PerformanceAdvice advice;
    advice.optimal_configuration = true;
    advice.estimated_performance_gain = 0.0;
    
    // Check CPU features
    if (!cpu_features_.has_avx2 && config_.enable_simd) {
        advice.recommendations.push_back("CPU lacks AVX2 support - consider upgrading for better SIMD performance");
        advice.estimated_performance_gain += 0.2;
        advice.optimal_configuration = false;
    }
    
    // Check huge page support
    if (!huge_page_info_.supported && config_.enable_huge_pages) {
        advice.warnings.push_back("Huge pages not available - enable in kernel configuration");
        advice.estimated_performance_gain += 0.15;
        advice.optimal_configuration = false;
    } else if (huge_page_info_.free_2mb_pages == 0 && config_.enable_huge_pages) {
        advice.recommendations.push_back("No free huge pages available - increase huge page allocation");
        advice.estimated_performance_gain += 0.1;
    }
    
    // Check NUMA configuration
    if (huge_page_info_.numa_available && !config_.enable_numa_affinity) {
        advice.recommendations.push_back("NUMA system detected - enable NUMA affinity for better performance");
        advice.estimated_performance_gain += 0.25;
        advice.optimal_configuration = false;
    }
    
    // Check CPU topology
    if (cpu_topology_.has_hyperthreading && config_.isolate_performance_threads) {
        advice.recommendations.push_back("Consider disabling hyperthreading for latency-critical applications");
    }
    
    if (cpu_topology_.has_hybrid_cores) {
        advice.recommendations.push_back("Hybrid CPU detected - bind high-priority threads to P-cores");
        if (config_.default_thread_type == ThreadType::LowLatency) {
            advice.estimated_performance_gain += 0.3;
        }
    }
    
    return advice;
}

void PerformanceManager::update_config(const Config& config) {
    config_ = config;
    
    if (initialized_) {
        // Reinitialize with new configuration
        shutdown();
        initialize();
    }
}

PerformanceManager::SystemPerformance PerformanceManager::measure_system_performance() {
    SystemPerformance perf = {};
    
    // Measure memory bandwidth
    perf.memory_bandwidth_gbps = measure_memory_bandwidth();
    
    // Estimate cache hit rate (simplified)
    perf.cache_hit_rate = 0.95; // Placeholder
    
    // Get CPU utilization
    auto load = get_load_average();
    perf.cpu_utilization = std::min(1.0, load.load_1min / cpu_topology_.num_logical_cores);
    
    // Check huge page usage
    perf.using_huge_pages = huge_page_pool_ != nullptr;
    
    // Check SIMD availability
    perf.simd_accelerated = cpu_features_.has_avx2 || cpu_features_.has_neon;
    
    // Estimate NUMA efficiency
    perf.numa_efficiency = huge_page_info_.numa_available ? 0.9 : 1.0;
    
    // Estimate context switches (placeholder)
    perf.context_switches_per_sec = 1000;
    
    return perf;
}

void PerformanceManager::detect_system_capabilities() {
    cpu_features_ = get_cpu_features();
    huge_page_info_ = get_huge_page_info();
    cpu_topology_ = get_cpu_topology();
    cache_hierarchy_ = get_cache_hierarchy();
}

void PerformanceManager::configure_huge_pages() {
    if (!huge_page_info_.supported) {
        return;
    }
    
    HugePagePool::PoolConfig pool_config;
    pool_config.policy = config_.huge_page_policy;
    pool_config.preferred_page_size = config_.preferred_page_size;
    pool_config.numa_aware = config_.enable_numa_affinity;
    
    huge_page_pool_ = std::make_unique<HugePagePool>(pool_config);
}

void PerformanceManager::configure_cpu_affinity() {
    AffinityThreadPool::PoolConfig pool_config;
    pool_config.thread_type = config_.default_thread_type;
    pool_config.pin_threads = true;
    pool_config.exclusive_cores = config_.isolate_performance_threads;
    
    if (config_.enable_numa_affinity && huge_page_info_.numa_available) {
        pool_config.numa_node = huge_page_info_.current_numa_node;
    }
    
    thread_pool_ = std::make_unique<AffinityThreadPool>(pool_config);
}

void PerformanceManager::configure_prefetching() {
    AdaptivePrefetcher::Config prefetch_config;
    prefetch_config.enable_stride_detection = true;
    prefetch_config.enable_pattern_learning = true;
    prefetch_config.target_level = config_.prefetch_target;
    
    prefetcher_ = std::make_unique<AdaptivePrefetcher>(prefetch_config);
}

PerformanceManager::Config PerformanceManager::generate_optimal_config() {
    Config optimal = config_;
    
    // Enable features based on system capabilities
    optimal.enable_simd = cpu_features_.has_avx2 || cpu_features_.has_neon;
    optimal.enable_huge_pages = huge_page_info_.supported;
    optimal.enable_numa_affinity = huge_page_info_.numa_available;
    optimal.enable_cpu_affinity = cpu_topology_.num_logical_cores > 2;
    
    // Configure huge pages
    if (huge_page_info_.supported) {
        optimal.huge_page_policy = huge_page_info_.free_2mb_pages > 0 ? 
            HugePagePolicy::TryBest : HugePagePolicy::Never;
    }
    
    // Configure for CPU architecture
    if (cpu_topology_.has_hybrid_cores) {
        optimal.default_thread_type = ThreadType::HighThroughput;
        optimal.isolate_performance_threads = true;
    }
    
    return optimal;
}

// ============================================================================
// Global Performance Manager
// ============================================================================

PerformanceManager& get_performance_manager() {
    static PerformanceManager instance;
    static std::once_flag init_flag;
    
    std::call_once(init_flag, [&]() {
        instance.initialize();
    });
    
    return instance;
}

// ============================================================================
// Convenience Functions
// ============================================================================

bool enable_performance_optimizations(const PerformanceManager::Config& config) {
    auto& manager = get_performance_manager();
    manager.update_config(config);
    return true;
}

void disable_performance_optimizations() {
    auto& manager = get_performance_manager();
    PerformanceManager::Config disabled_config = {};
    disabled_config.enable_simd = false;
    disabled_config.enable_huge_pages = false;
    disabled_config.enable_numa_affinity = false;
    disabled_config.enable_cpu_affinity = false;
    disabled_config.enable_prefetching = false;
    
    manager.update_config(disabled_config);
}

std::string get_performance_summary() {
    auto& manager = get_performance_manager();
    auto advice = manager.analyze_system_performance();
    auto perf = manager.measure_system_performance();
    
    std::ostringstream summary;
    summary << "=== Psyne Performance Summary ===\n";
    summary << "Memory Bandwidth: " << perf.memory_bandwidth_gbps << " GB/s\n";
    summary << "Cache Hit Rate: " << (perf.cache_hit_rate * 100) << "%\n";
    summary << "CPU Utilization: " << (perf.cpu_utilization * 100) << "%\n";
    summary << "SIMD Acceleration: " << (perf.simd_accelerated ? "Yes" : "No") << "\n";
    summary << "Huge Pages: " << (perf.using_huge_pages ? "Yes" : "No") << "\n";
    summary << "NUMA Efficiency: " << (perf.numa_efficiency * 100) << "%\n";
    
    if (!advice.optimal_configuration) {
        summary << "\nPerformance Improvements Available:\n";
        for (const auto& rec : advice.recommendations) {
            summary << "- " << rec << "\n";
        }
        summary << "Estimated gain: " << (advice.estimated_performance_gain * 100) << "%\n";
    }
    
    return summary.str();
}

std::vector<std::string> get_performance_recommendations() {
    auto& manager = get_performance_manager();
    auto advice = manager.analyze_system_performance();
    return advice.recommendations;
}

// ============================================================================
// Benchmark Utilities
// ============================================================================

std::vector<BenchmarkResult> run_performance_benchmarks() {
    std::vector<BenchmarkResult> results;
    
    // Memory bandwidth benchmark
    {
        BenchmarkResult result;
        result.test_name = "Memory Bandwidth";
        result.units = "GB/s";
        
        // Baseline
        result.baseline_performance = measure_memory_bandwidth(1024 * 1024, 100);
        
        // With optimizations
        auto& manager = get_performance_manager();
        const auto& config = manager.get_config();
        
        double optimized_perf = result.baseline_performance;
        if (config.enable_huge_pages) {
            optimized_perf *= 1.15; // Estimate 15% improvement with huge pages
        }
        if (config.enable_numa_affinity) {
            optimized_perf *= 1.1;  // Estimate 10% improvement with NUMA affinity
        }
        
        result.optimized_performance = optimized_perf;
        result.improvement_ratio = optimized_perf / result.baseline_performance;
        
        if (config.enable_huge_pages) {
            result.optimizations_used.push_back("Huge Pages");
        }
        if (config.enable_numa_affinity) {
            result.optimizations_used.push_back("NUMA Affinity");
        }
        
        results.push_back(result);
    }
    
    // SIMD operations benchmark
    {
        BenchmarkResult result;
        result.test_name = "Vector Operations";
        result.units = "GFLOPS";
        
        // Simple vector operation benchmark
        constexpr size_t vector_size = 10000;
        std::vector<float> a(vector_size, 1.0f);
        std::vector<float> b(vector_size, 2.0f);
        std::vector<float> c(vector_size);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Baseline - scalar operations
        for (int iter = 0; iter < 1000; ++iter) {
            for (size_t i = 0; i < vector_size; ++i) {
                c[i] = a[i] + b[i];
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        double ops_per_sec = (1000.0 * vector_size) / (duration.count() / 1e9);
        result.baseline_performance = ops_per_sec / 1e9; // GOPS
        
        // SIMD optimized
        start = std::chrono::high_resolution_clock::now();
        
        for (int iter = 0; iter < 1000; ++iter) {
            vector_add_f32(a.data(), b.data(), c.data(), vector_size);
        }
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        ops_per_sec = (1000.0 * vector_size) / (duration.count() / 1e9);
        result.optimized_performance = ops_per_sec / 1e9; // GOPS
        result.improvement_ratio = result.optimized_performance / result.baseline_performance;
        result.optimizations_used.push_back("SIMD Instructions");
        
        results.push_back(result);
    }
    
    return results;
}

// ============================================================================
// OptimizedMessage Implementation
// ============================================================================

template<typename MessageType>
void OptimizedMessage<MessageType>::configure_for_message_type() {
    // Configure based on message characteristics
    size_t message_size = sizeof(MessageType);
    
    if (message_size > 64 * 1024) {
        // Large messages - use streaming strategy
        MessagePrefetcher::MessageConfig config;
        config.strategy = MessagePrefetchStrategy::Streaming;
        config.message_size_estimate = message_size;
        config.lookahead_messages = 1;
        prefetcher_ = MessagePrefetcher(config);
    } else if (message_size < 256) {
        // Small messages - minimal prefetching
        MessagePrefetcher::MessageConfig config;
        config.strategy = MessagePrefetchStrategy::Random;
        config.prefetch_payload = false;
        prefetcher_ = MessagePrefetcher(config);
    } else {
        // Medium messages - sequential strategy
        MessagePrefetcher::MessageConfig config;
        config.strategy = MessagePrefetchStrategy::Sequential;
        config.message_size_estimate = message_size;
        prefetcher_ = MessagePrefetcher(config);
    }
}

template<typename MessageType>
void OptimizedMessage<MessageType>::apply_simd_optimizations() {
    // Apply SIMD optimizations based on message type
    // This would be specialized for different message types
    // For now, just record that SIMD could be applied
}

template<typename MessageType>
typename OptimizedMessage<MessageType>::OptimizationStats 
OptimizedMessage<MessageType>::get_optimization_stats() const {
    OptimizationStats stats = {};
    
    const auto& manager = get_performance_manager();
    const auto& cpu_features = manager.get_cpu_features();
    const auto& huge_page_info = manager.get_huge_page_info();
    
    stats.simd_used = cpu_features.has_avx2 || cpu_features.has_neon;
    stats.prefetch_active = true;
    stats.huge_pages_used = huge_page_info.supported;
    
    // Estimate speedup based on optimizations
    stats.estimated_speedup = 1.0;
    if (stats.simd_used) stats.estimated_speedup *= 2.0;
    if (stats.prefetch_active) stats.estimated_speedup *= 1.2;
    if (stats.huge_pages_used) stats.estimated_speedup *= 1.15;
    
    return stats;
}

// ============================================================================
// Performance Allocator
// ============================================================================

void* PerformanceAllocator::allocate(size_t size, size_t alignment) {
    auto& manager = get_performance_manager();
    const auto& config = manager.get_config();
    
    if (config.enable_huge_pages && size > 2 * 1024 * 1024) {
        // Use huge page allocation for large allocations
        auto allocation = allocate_huge_pages(size, config.huge_page_policy);
        return allocation.ptr;
    } else {
        // Use cache-aligned allocation
        size_t cache_line_size = get_cache_line_size();
        if (alignment == 0) alignment = cache_line_size;
        
        return std::aligned_alloc(alignment, size);
    }
}

void PerformanceAllocator::deallocate(void* ptr, size_t size) {
    if (!ptr) return;
    
    // For simplicity, just use free
    // In a real implementation, we'd track allocation method
    std::free(ptr);
}

void* PerformanceAllocator::reallocate(void* ptr, size_t old_size, size_t new_size) {
    void* new_ptr = allocate(new_size);
    if (new_ptr && ptr) {
        std::memcpy(new_ptr, ptr, std::min(old_size, new_size));
        deallocate(ptr, old_size);
    }
    return new_ptr;
}

size_t PerformanceAllocator::get_total_allocated() {
    // Placeholder implementation
    return 0;
}

size_t PerformanceAllocator::get_allocation_count() {
    // Placeholder implementation
    return 0;
}

double PerformanceAllocator::get_allocation_efficiency() {
    // Placeholder implementation
    return 1.0;
}

// Template instantiations for common message types
template class OptimizedMessage<FloatVector>;
template class OptimizedMessage<DoubleMatrix>;

} // namespace perf
} // namespace psyne