#pragma once

// Memory prefetching hints for improved cache performance
// Strategic prefetching can significantly reduce memory latency

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>

namespace psyne {
namespace perf {

// Prefetch hint types
enum class PrefetchHint {
    Read,           // Prefetch for reading
    Write,          // Prefetch for writing
    ReadWrite,      // Prefetch for both reading and writing
    Temporal,       // Data will be used again soon (keep in cache)
    NonTemporal,    // Data will not be used again (don't pollute cache)
    Stream          // Sequential streaming access pattern
};

// Cache level targets for prefetching
enum class CacheLevel {
    L1 = 1,         // Prefetch to L1 cache
    L2 = 2,         // Prefetch to L2 cache  
    L3 = 3,         // Prefetch to L3 cache
    Auto = 0        // Let CPU decide optimal cache level
};

// Prefetch distance strategies
enum class PrefetchDistance {
    Near = 64,      // 1 cache line ahead
    Medium = 256,   // 4 cache lines ahead
    Far = 1024,     // 16 cache lines ahead
    Auto = 0        // Automatically determine optimal distance
};

// ============================================================================
// Basic Prefetch Operations
// ============================================================================

// Prefetch a single memory location
void prefetch(const void* addr, PrefetchHint hint = PrefetchHint::Read, 
              CacheLevel level = CacheLevel::Auto);

// Prefetch a memory range
void prefetch_range(const void* start, size_t size, 
                   PrefetchHint hint = PrefetchHint::Read,
                   CacheLevel level = CacheLevel::Auto,
                   size_t stride = 64);

// Prefetch with specific distance ahead
void prefetch_ahead(const void* current_addr, size_t distance,
                   PrefetchHint hint = PrefetchHint::Read);

// Prefetch multiple non-contiguous addresses
void prefetch_scatter(const std::vector<const void*>& addresses,
                     PrefetchHint hint = PrefetchHint::Read);

// ============================================================================
// Adaptive Prefetching
// ============================================================================

// Adaptive prefetcher that learns access patterns
class AdaptivePrefetcher {
public:
    struct Config {
        size_t history_size = 16;           // Number of addresses to track
        double confidence_threshold = 0.7;  // Minimum confidence for prefetch
        size_t max_prefetch_distance = 2048; // Maximum bytes to prefetch ahead
        bool enable_stride_detection = true;
        bool enable_pattern_learning = true;
        CacheLevel target_level = CacheLevel::Auto;
    };

public:
    AdaptivePrefetcher();
    explicit AdaptivePrefetcher(const Config& config);
    ~AdaptivePrefetcher() = default;
    
    // Record a memory access for pattern learning
    void record_access(const void* addr, PrefetchHint hint = PrefetchHint::Read);
    
    // Predict and prefetch next likely accesses
    size_t prefetch_predicted(size_t max_prefetches = 4);
    
    // Get predicted next addresses (without prefetching)
    std::vector<const void*> predict_next_accesses(size_t count = 4);
    
    // Reset learning state
    void reset();
    
    // Get performance statistics
    struct Stats {
        uint64_t total_accesses;
        uint64_t prefetches_issued;
        uint64_t prefetch_hits;     // Prefetched data that was actually used
        uint64_t prefetch_misses;   // Prefetched data that was not used
        double hit_rate;
        double confidence;
    };
    
    Stats get_stats() const;

private:
    Config config_;
    
    struct AccessEntry {
        const void* addr;
        uint64_t timestamp;
        PrefetchHint hint;
    };
    
    std::vector<AccessEntry> access_history_;
    size_t history_index_;
    
    // Stride detection
    struct StridePattern {
        intptr_t stride;
        size_t confidence;
        uint64_t last_seen;
    };
    
    std::vector<StridePattern> stride_patterns_;
    
    // Performance tracking
    mutable Stats stats_;
    
    void detect_stride_patterns();
    void update_statistics();
    double calculate_confidence(const StridePattern& pattern) const;
};

// ============================================================================
// Message-Specific Prefetching
// ============================================================================

// Prefetch strategies for different message types
enum class MessagePrefetchStrategy {
    Sequential,     // Linear sequential access
    Random,         // Random access pattern
    Structured,     // Structured data with known layout
    Streaming,      // High-bandwidth streaming
    Sparse          // Sparse access pattern
};

// Message prefetcher optimized for zero-copy messaging
class MessagePrefetcher {
public:
    struct MessageConfig {
        MessagePrefetchStrategy strategy = MessagePrefetchStrategy::Sequential;
        size_t message_size_estimate = 1024;
        size_t lookahead_messages = 2;      // Number of messages to prefetch ahead
        bool prefetch_headers = true;
        bool prefetch_payload = true;
        CacheLevel target_level = CacheLevel::L2;
    };

public:
    MessagePrefetcher();
    explicit MessagePrefetcher(const MessageConfig& config);
    
    // Prefetch for message reading
    void prefetch_message_read(const void* message_ptr, size_t message_size);
    
    // Prefetch for message writing
    void prefetch_message_write(void* message_ptr, size_t message_size);
    
    // Prefetch ring buffer region
    void prefetch_ring_buffer(const void* buffer_start, size_t buffer_size,
                             size_t read_pos, size_t write_pos);
    
    // Prefetch based on message queue state
    void prefetch_queue_region(const void* queue_base, size_t element_size,
                              size_t current_index, size_t queue_size);
    
    // Update configuration based on observed patterns
    void adapt_to_pattern(const std::vector<size_t>& message_sizes);

private:
    MessageConfig config_;
    
    void prefetch_with_strategy(const void* ptr, size_t size, PrefetchHint hint);
    size_t calculate_optimal_stride(size_t message_size) const;
};

// ============================================================================
// Cache-Aware Data Structures
// ============================================================================

// Cache-line aligned allocator
template<typename T>
class CacheAlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = CacheAlignedAllocator<U>;
    };
    
    CacheAlignedAllocator() = default;
    
    template<typename U>
    CacheAlignedAllocator(const CacheAlignedAllocator<U>&) {}
    
    pointer allocate(size_type n);
    void deallocate(pointer p, size_type n);
    
    template<typename U, typename... Args>
    void construct(U* p, Args&&... args);
    
    template<typename U>
    void destroy(U* p);
    
    size_type max_size() const noexcept;
    
    bool operator==(const CacheAlignedAllocator& other) const { return true; }
    bool operator!=(const CacheAlignedAllocator& other) const { return false; }
};

// Cache-friendly vector with prefetching
template<typename T>
class PrefetchVector {
public:
    using value_type = T;
    using allocator_type = CacheAlignedAllocator<T>;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;
    
private:
    std::vector<T, allocator_type> data_;
    mutable AdaptivePrefetcher prefetcher_;
    
public:
    explicit PrefetchVector(const AdaptivePrefetcher::Config& prefetch_config = {});
    
    // Standard vector interface with prefetching
    T& at(size_type pos);
    const T& at(size_type pos) const;
    
    T& operator[](size_type pos);
    const T& operator[](size_type pos) const;
    
    void push_back(const T& value);
    void push_back(T&& value);
    
    template<typename... Args>
    void emplace_back(Args&&... args);
    
    void reserve(size_type new_capacity);
    void resize(size_type new_size);
    
    size_type size() const { return data_.size(); }
    size_type capacity() const { return data_.capacity(); }
    bool empty() const { return data_.empty(); }
    
    iterator begin() { return data_.data(); }
    iterator end() { return data_.data() + data_.size(); }
    const_iterator begin() const { return data_.data(); }
    const_iterator end() const { return data_.data() + data_.size(); }
    
    // Prefetch-aware operations
    void prefetch_range(size_type start, size_type count);
    void sequential_access_hint(size_type start_pos);
    void random_access_hint();
};

// ============================================================================
// Hardware Performance Counters Integration
// ============================================================================

// Cache performance monitoring
struct CachePerformance {
    uint64_t l1_hits;
    uint64_t l1_misses;
    uint64_t l2_hits;
    uint64_t l2_misses;
    uint64_t l3_hits;
    uint64_t l3_misses;
    uint64_t memory_accesses;
    uint64_t prefetch_requests;
    
    double l1_hit_rate() const { return static_cast<double>(l1_hits) / (l1_hits + l1_misses); }
    double l2_hit_rate() const { return static_cast<double>(l2_hits) / (l2_hits + l2_misses); }
    double l3_hit_rate() const { return static_cast<double>(l3_hits) / (l3_hits + l3_misses); }
};

// Start cache performance monitoring
bool start_cache_monitoring();

// Stop monitoring and get results
CachePerformance stop_cache_monitoring();

// Get current cache performance counters
CachePerformance get_cache_performance();

// ============================================================================
// Prefetch Pattern Analysis
// ============================================================================

// Analyze memory access patterns for optimization
struct AccessPattern {
    enum class Type {
        Sequential,
        Strided,
        Random,
        Hotspot,
        Unknown
    } type;
    
    double stride_average;
    double stride_variance;
    size_t hotspot_regions;
    double temporal_locality;
    double spatial_locality;
    std::string description;
};

// Analyze access pattern from a sequence of addresses
AccessPattern analyze_access_pattern(const std::vector<const void*>& addresses);

// Get prefetch recommendations based on access pattern
struct PrefetchRecommendation {
    PrefetchDistance distance;
    CacheLevel target_level;
    size_t stride;
    bool enable_hardware_prefetcher;
    std::string reasoning;
};

PrefetchRecommendation get_prefetch_recommendation(const AccessPattern& pattern);

// ============================================================================
// Utility Functions
// ============================================================================

// Get cache line size for optimal prefetch alignment
size_t get_cache_line_size();

// Get cache sizes and latencies
struct CacheHierarchy {
    struct CacheInfo {
        size_t size;
        size_t line_size;
        size_t associativity;
        double latency_cycles;
    };
    
    CacheInfo l1_data;
    CacheInfo l1_instruction;
    CacheInfo l2;
    CacheInfo l3;
    double memory_latency_cycles;
};

CacheHierarchy get_cache_hierarchy();

// Calculate optimal prefetch distance based on memory latency
size_t calculate_optimal_prefetch_distance(double memory_bandwidth_gbps,
                                          double access_frequency_hz);

// Check if hardware prefetcher is enabled
bool is_hardware_prefetcher_enabled();

// Enable/disable hardware prefetcher (requires privileges)
bool set_hardware_prefetcher_enabled(bool enabled);

// Memory bandwidth measurement
double measure_memory_bandwidth(size_t buffer_size = 64 * 1024 * 1024,
                               size_t iterations = 100);

// Prefetch effectiveness measurement
struct PrefetchEffectiveness {
    double performance_improvement;  // Percentage improvement
    double cache_hit_improvement;    // Cache hit rate improvement
    double bandwidth_utilization;    // Memory bandwidth utilization
    size_t optimal_distance;         // Optimal prefetch distance found
};

PrefetchEffectiveness measure_prefetch_effectiveness(
    std::function<void()> workload_without_prefetch,
    std::function<void()> workload_with_prefetch);

} // namespace perf
} // namespace psyne