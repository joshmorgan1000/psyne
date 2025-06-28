#pragma once

// CPU affinity and NUMA-aware thread management for optimal performance
// Helps bind threads to specific cores and manage CPU topology

#include <vector>
#include <string>
#include <cstdint>
#include <thread>
#include <functional>
#include <future>
#include <queue>

namespace psyne {
namespace perf {

// CPU core information
struct CoreInfo {
    int core_id;                // Logical core ID
    int physical_id;            // Physical CPU ID (socket)
    int numa_node;              // NUMA node this core belongs to
    bool is_hyperthread;        // True if this is a hyperthread core
    int sibling_core_id;        // ID of sibling hyperthread core (-1 if none)
    double frequency_mhz;       // Current frequency in MHz
    double max_frequency_mhz;   // Maximum frequency in MHz
    bool is_performance_core;   // True for P-cores on hybrid architectures
    bool is_efficiency_core;    // True for E-cores on hybrid architectures
};

// CPU topology information
struct CPUTopology {
    std::vector<CoreInfo> cores;
    int num_physical_cpus;      // Number of physical CPU packages/sockets
    int num_cores;              // Number of physical cores
    int num_logical_cores;      // Number of logical cores (including hyperthreads)
    int num_numa_nodes;         // Number of NUMA nodes
    bool has_hyperthreading;    // True if hyperthreading is enabled
    bool has_hybrid_cores;      // True if system has P/E core architecture
    std::string cpu_model;      // CPU model string
    std::string architecture;   // Architecture (x86_64, aarch64, etc.)
};

// Thread affinity configuration
struct AffinityConfig {
    std::vector<int> allowed_cores;     // Specific cores to bind to
    int numa_node = -1;                 // NUMA node preference (-1 = no preference)
    bool prefer_performance_cores = true;  // Prefer P-cores on hybrid systems
    bool avoid_hyperthreads = false;    // Avoid hyperthread siblings if possible
    bool exclusive_cores = false;       // Request exclusive core access
    int thread_priority = 0;            // Thread priority adjustment
};

// Thread performance hints
enum class ThreadType {
    HighThroughput,     // High-throughput data processing
    LowLatency,         // Low-latency messaging
    Background,         // Background/maintenance work
    Compute,            // CPU-intensive computation
    IO,                 // I/O bound operations
    Network             // Network processing
};

// ============================================================================
// CPU Topology Discovery
// ============================================================================

// Get detailed CPU topology information
CPUTopology get_cpu_topology();

// Get simplified core count information
struct CoreCounts {
    int physical_cores;
    int logical_cores;
    int performance_cores;  // P-cores (0 if not hybrid)
    int efficiency_cores;   // E-cores (0 if not hybrid)
};

CoreCounts get_core_counts();

// Check if the system has hybrid (P/E) core architecture
bool has_hybrid_architecture();

// Get list of performance cores (P-cores)
std::vector<int> get_performance_cores();

// Get list of efficiency cores (E-cores) 
std::vector<int> get_efficiency_cores();

// Get cores belonging to a specific NUMA node
std::vector<int> get_numa_node_cores(int numa_node);

// ============================================================================
// Thread Affinity Management
// ============================================================================

// Set thread affinity to specific cores
bool set_thread_affinity(std::thread::id thread_id, const std::vector<int>& core_ids);
bool set_thread_affinity(const std::vector<int>& core_ids); // Current thread

// Set thread affinity based on configuration
bool set_thread_affinity(std::thread::id thread_id, const AffinityConfig& config);
bool set_thread_affinity(const AffinityConfig& config); // Current thread

// Get current thread affinity
std::vector<int> get_thread_affinity(std::thread::id thread_id);
std::vector<int> get_thread_affinity(); // Current thread

// Remove thread affinity restrictions (allow all cores)
bool clear_thread_affinity(std::thread::id thread_id);
bool clear_thread_affinity(); // Current thread

// ============================================================================
// Automatic Affinity Assignment
// ============================================================================

// Get recommended affinity configuration for a thread type
AffinityConfig get_recommended_affinity(ThreadType thread_type);

// Automatically assign optimal cores for a thread type
bool set_optimal_affinity(ThreadType thread_type);
bool set_optimal_affinity(std::thread::id thread_id, ThreadType thread_type);

// Distribute threads across cores optimally
std::vector<AffinityConfig> distribute_threads_across_cores(
    size_t num_threads, 
    ThreadType thread_type = ThreadType::HighThroughput,
    bool avoid_core_sharing = true);

// ============================================================================
// Thread Priority Management
// ============================================================================

// Thread priority levels (platform-independent)
enum class ThreadPriority {
    Idle = -3,
    BelowNormal = -1,
    Normal = 0,
    AboveNormal = 1,
    High = 2,
    Realtime = 3        // Use with caution!
};

// Set thread priority
bool set_thread_priority(std::thread::id thread_id, ThreadPriority priority);
bool set_thread_priority(ThreadPriority priority); // Current thread

// Get current thread priority
ThreadPriority get_thread_priority(std::thread::id thread_id);
ThreadPriority get_thread_priority(); // Current thread

// ============================================================================
// Performance Optimization Helpers
// ============================================================================

// CPU isolation and performance settings
struct PerformanceConfig {
    bool disable_cpu_frequency_scaling = false;
    bool set_cpu_governor_performance = false;
    bool disable_irq_balancing = false;
    bool isolate_cores = false;
    std::vector<int> isolated_cores;
    bool disable_smt = false;           // Disable simultaneous multithreading
};

// Apply system-wide performance optimizations (requires privileges)
bool apply_performance_config(const PerformanceConfig& config);

// Get current CPU frequency for a core
double get_cpu_frequency(int core_id);

// Set CPU frequency scaling governor (requires privileges)
bool set_cpu_governor(const std::string& governor); // "performance", "powersave", etc.

// ============================================================================
// Thread Pool with Affinity Management
// ============================================================================

class AffinityThreadPool {
public:
    struct PoolConfig {
        size_t num_threads = 0;     // 0 = auto-detect optimal count
        ThreadType thread_type = ThreadType::HighThroughput;
        bool pin_threads = true;
        bool exclusive_cores = false;
        ThreadPriority priority = ThreadPriority::Normal;
        int numa_node = -1;         // -1 = distribute across nodes
    };

public:
    AffinityThreadPool();
    explicit AffinityThreadPool(const PoolConfig& config);
    ~AffinityThreadPool();
    
    // Non-copyable, movable
    AffinityThreadPool(const AffinityThreadPool&) = delete;
    AffinityThreadPool& operator=(const AffinityThreadPool&) = delete;
    AffinityThreadPool(AffinityThreadPool&&) noexcept;
    AffinityThreadPool& operator=(AffinityThreadPool&&) noexcept;
    
    // Submit work to the thread pool
    template<typename Func, typename... Args>
    auto submit(Func&& func, Args&&... args) -> std::future<decltype(func(args...))>;
    
    // Get the number of worker threads
    size_t get_thread_count() const { return threads_.size(); }
    
    // Get affinity configuration for each thread
    std::vector<AffinityConfig> get_thread_affinities() const;
    
    // Reconfigure the thread pool
    void reconfigure(const PoolConfig& config);
    
    // Wait for all pending work to complete
    void wait_for_completion();
    
    // Get pool statistics
    struct PoolStats {
        size_t tasks_submitted;
        size_t tasks_completed;
        size_t tasks_pending;
        double average_task_time_ms;
        std::vector<double> per_thread_utilization;
    };
    
    PoolStats get_stats() const;

private:
    PoolConfig config_;
    std::vector<std::thread> threads_;
    std::vector<AffinityConfig> thread_affinities_;
    
    // Work queue and synchronization
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    PoolStats stats_;
    
    void worker_thread(size_t thread_index);
    void setup_thread_affinity(size_t thread_index);
    size_t calculate_optimal_thread_count() const;
    std::vector<AffinityConfig> calculate_thread_affinities() const;
};

// ============================================================================
// Utility Functions
// ============================================================================

// Get the current thread's CPU core
int get_current_cpu_core();

// Yield current thread to allow other threads on the same core to run
void yield_to_hyperthread_sibling();

// Busy-wait for a short duration (CPU-bound spin)
void cpu_pause(int cycles = 1);

// Get CPU cache line size
size_t get_cpu_cache_line_size();

// Get CPU cache sizes (L1, L2, L3)
struct CacheSizes {
    size_t l1_instruction;
    size_t l1_data;
    size_t l2;
    size_t l3;
};

CacheSizes get_cpu_cache_sizes();

// Check if two memory addresses are likely on the same cache line
bool same_cache_line(const void* addr1, const void* addr2);

// Align address to cache line boundary
void* align_to_cache_line(void* addr);
size_t align_size_to_cache_line(size_t size);

// ============================================================================
// Monitoring and Diagnostics
// ============================================================================

// Thread performance monitoring
struct ThreadPerfCounters {
    uint64_t cpu_cycles;
    uint64_t instructions;
    uint64_t cache_misses;
    uint64_t branch_misses;
    uint64_t context_switches;
    double cpu_utilization;
    int current_core;
    int migrations;         // Number of core migrations
};

// Get performance counters for a thread (if available)
ThreadPerfCounters get_thread_perf_counters(std::thread::id thread_id);

// Start/stop performance monitoring for current thread
bool start_perf_monitoring();
bool stop_perf_monitoring();

// Get system-wide CPU utilization per core
std::vector<double> get_cpu_utilization_per_core();

// Get current system load average
struct LoadAverage {
    double load_1min;
    double load_5min;
    double load_15min;
};

LoadAverage get_load_average();

// Diagnose thread affinity and performance issues
std::string diagnose_thread_performance();

} // namespace perf
} // namespace psyne