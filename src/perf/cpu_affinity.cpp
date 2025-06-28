#include "../../include/psyne/perf/cpu_affinity.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <future>
#include <queue>

// Platform-specific includes
#ifdef __linux__
    #include <sched.h>
    #include <unistd.h>
    #include <sys/syscall.h>
    #include <numa.h>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
    #include <mach/mach.h>
    #include <mach/thread_policy.h>
#elif defined(_WIN32)
    #include <windows.h>
    #include <processthreadsapi.h>
#endif

namespace psyne {
namespace perf {

// ============================================================================
// CPU Topology Discovery - Simplified Implementation
// ============================================================================

CPUTopology get_cpu_topology() {
    CPUTopology topology = {};
    
#ifdef __linux__
    // Read from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    
    int max_core_id = -1;
    int max_physical_id = -1;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("processor") == 0) {
            topology.num_logical_cores++;
        } else if (line.find("core id") == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                int core_id = std::stoi(line.substr(colon + 1));
                max_core_id = std::max(max_core_id, core_id);
            }
        } else if (line.find("physical id") == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                int phys_id = std::stoi(line.substr(colon + 1));
                max_physical_id = std::max(max_physical_id, phys_id);
            }
        } else if (line.find("model name") == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos && topology.cpu_model.empty()) {
                topology.cpu_model = line.substr(colon + 2);
            }
        }
    }
    
    topology.num_physical_cpus = max_physical_id + 1;
    topology.num_cores = max_core_id + 1;
    topology.has_hyperthreading = topology.num_logical_cores > topology.num_cores;
    topology.architecture = "x86_64";
    
#elif defined(__APPLE__)
    size_t size;
    
    // Get number of logical cores
    size = sizeof(topology.num_logical_cores);
    sysctlbyname("hw.logicalcpu", &topology.num_logical_cores, &size, nullptr, 0);
    
    // Get number of physical cores
    size = sizeof(topology.num_cores);
    sysctlbyname("hw.physicalcpu", &topology.num_cores, &size, nullptr, 0);
    
    topology.num_physical_cpus = 1; // Assume single CPU package
    topology.has_hyperthreading = topology.num_logical_cores > topology.num_cores;
    
    // Check for Apple Silicon
    char cpu_brand[256];
    size = sizeof(cpu_brand);
    if (sysctlbyname("machdep.cpu.brand_string", cpu_brand, &size, nullptr, 0) == 0) {
        topology.cpu_model = cpu_brand;
        if (topology.cpu_model.find("Apple") != std::string::npos) {
            topology.has_hybrid_cores = true;
            topology.architecture = "aarch64";
        } else {
            topology.architecture = "x86_64";
        }
    }
    
#else
    // Default fallback
    topology.num_logical_cores = std::thread::hardware_concurrency();
    topology.num_cores = topology.num_logical_cores;
    topology.num_physical_cpus = 1;
    topology.has_hyperthreading = false;
    topology.has_hybrid_cores = false;
    topology.cpu_model = "Unknown";
    topology.architecture = "unknown";
#endif

    // Populate core information
    for (int i = 0; i < topology.num_logical_cores; ++i) {
        CoreInfo core;
        core.core_id = i;
        core.physical_id = i / (topology.num_logical_cores / topology.num_physical_cpus);
        core.numa_node = 0; // Simplified
        core.is_hyperthread = topology.has_hyperthreading && (i >= topology.num_cores);
        core.sibling_core_id = topology.has_hyperthreading ? 
            (i < topology.num_cores ? i + topology.num_cores : i - topology.num_cores) : -1;
        core.frequency_mhz = 2400.0; // Placeholder
        core.max_frequency_mhz = 3200.0; // Placeholder
        
        // For hybrid architectures, assume first half are P-cores
        if (topology.has_hybrid_cores) {
            core.is_performance_core = i < topology.num_cores / 2;
            core.is_efficiency_core = !core.is_performance_core;
        } else {
            core.is_performance_core = true;
            core.is_efficiency_core = false;
        }
        
        topology.cores.push_back(core);
    }
    
    return topology;
}

CoreCounts get_core_counts() {
    auto topology = get_cpu_topology();
    
    CoreCounts counts = {};
    counts.physical_cores = topology.num_cores;
    counts.logical_cores = topology.num_logical_cores;
    
    for (const auto& core : topology.cores) {
        if (core.is_performance_core) counts.performance_cores++;
        if (core.is_efficiency_core) counts.efficiency_cores++;
    }
    
    return counts;
}

bool has_hybrid_architecture() {
    return get_cpu_topology().has_hybrid_cores;
}

std::vector<int> get_performance_cores() {
    auto topology = get_cpu_topology();
    std::vector<int> p_cores;
    
    for (const auto& core : topology.cores) {
        if (core.is_performance_core) {
            p_cores.push_back(core.core_id);
        }
    }
    
    return p_cores;
}

std::vector<int> get_efficiency_cores() {
    auto topology = get_cpu_topology();
    std::vector<int> e_cores;
    
    for (const auto& core : topology.cores) {
        if (core.is_efficiency_core) {
            e_cores.push_back(core.core_id);
        }
    }
    
    return e_cores;
}

std::vector<int> get_numa_node_cores(int numa_node) {
    auto topology = get_cpu_topology();
    std::vector<int> node_cores;
    
    for (const auto& core : topology.cores) {
        if (core.numa_node == numa_node) {
            node_cores.push_back(core.core_id);
        }
    }
    
    return node_cores;
}

// ============================================================================
// Thread Affinity Management - Simplified Implementation
// ============================================================================

bool set_thread_affinity(std::thread::id thread_id, const std::vector<int>& core_ids) {
    // For current thread implementation
    return set_thread_affinity(core_ids);
}

bool set_thread_affinity(const std::vector<int>& core_ids) {
    if (core_ids.empty()) return false;
    
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    for (int core_id : core_ids) {
        CPU_SET(core_id, &cpuset);
    }
    
    return sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0;
    
#elif defined(__APPLE__)
    // macOS thread affinity is more limited
    // For now, just return success
    return true;
    
#elif defined(_WIN32)
    DWORD_PTR mask = 0;
    for (int core_id : core_ids) {
        mask |= (1ULL << core_id);
    }
    
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
    
#else
    return false;
#endif
}

bool set_thread_affinity(std::thread::id thread_id, const AffinityConfig& config) {
    return set_thread_affinity(config.allowed_cores);
}

bool set_thread_affinity(const AffinityConfig& config) {
    return set_thread_affinity(config.allowed_cores);
}

std::vector<int> get_thread_affinity(std::thread::id thread_id) {
    return get_thread_affinity();
}

std::vector<int> get_thread_affinity() {
    std::vector<int> cores;
    
#ifdef __linux__
    cpu_set_t cpuset;
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &cpuset)) {
                cores.push_back(i);
            }
        }
    }
#endif
    
    return cores;
}

bool clear_thread_affinity(std::thread::id thread_id) {
    return clear_thread_affinity();
}

bool clear_thread_affinity() {
    auto topology = get_cpu_topology();
    std::vector<int> all_cores;
    
    for (int i = 0; i < topology.num_logical_cores; ++i) {
        all_cores.push_back(i);
    }
    
    return set_thread_affinity(all_cores);
}

// ============================================================================
// Automatic Affinity Assignment - Simplified Implementation
// ============================================================================

AffinityConfig get_recommended_affinity(ThreadType thread_type) {
    AffinityConfig config;
    
    auto topology = get_cpu_topology();
    
    switch (thread_type) {
        case ThreadType::LowLatency:
            config.allowed_cores = get_performance_cores();
            config.prefer_performance_cores = true;
            config.avoid_hyperthreads = true;
            config.exclusive_cores = true;
            break;
            
        case ThreadType::HighThroughput:
            for (int i = 0; i < topology.num_logical_cores; ++i) {
                config.allowed_cores.push_back(i);
            }
            config.prefer_performance_cores = false;
            config.avoid_hyperthreads = false;
            break;
            
        case ThreadType::Background:
            config.allowed_cores = get_efficiency_cores();
            if (config.allowed_cores.empty()) {
                // Fallback to last few cores
                for (int i = topology.num_logical_cores / 2; i < topology.num_logical_cores; ++i) {
                    config.allowed_cores.push_back(i);
                }
            }
            break;
            
        default:
            // Default configuration
            for (int i = 0; i < topology.num_logical_cores; ++i) {
                config.allowed_cores.push_back(i);
            }
            break;
    }
    
    return config;
}

bool set_optimal_affinity(ThreadType thread_type) {
    auto config = get_recommended_affinity(thread_type);
    return set_thread_affinity(config);
}

bool set_optimal_affinity(std::thread::id thread_id, ThreadType thread_type) {
    auto config = get_recommended_affinity(thread_type);
    return set_thread_affinity(thread_id, config);
}

std::vector<AffinityConfig> distribute_threads_across_cores(
    size_t num_threads, ThreadType thread_type, bool avoid_core_sharing) {
    
    std::vector<AffinityConfig> configs;
    auto topology = get_cpu_topology();
    
    std::vector<int> available_cores;
    if (thread_type == ThreadType::LowLatency) {
        available_cores = get_performance_cores();
    } else {
        for (int i = 0; i < topology.num_logical_cores; ++i) {
            available_cores.push_back(i);
        }
    }
    
    if (avoid_core_sharing && available_cores.size() >= num_threads) {
        // Assign one core per thread
        for (size_t i = 0; i < num_threads; ++i) {
            AffinityConfig config;
            config.allowed_cores = {available_cores[i % available_cores.size()]};
            config.exclusive_cores = true;
            configs.push_back(config);
        }
    } else {
        // Distribute threads across available cores
        for (size_t i = 0; i < num_threads; ++i) {
            AffinityConfig config;
            config.allowed_cores = available_cores;
            configs.push_back(config);
        }
    }
    
    return configs;
}

// ============================================================================
// Thread Priority Management - Simplified Implementation
// ============================================================================

bool set_thread_priority(std::thread::id thread_id, ThreadPriority priority) {
    // For current thread implementation
    return set_thread_priority(priority);
}

bool set_thread_priority(ThreadPriority priority) {
#ifdef __linux__
    int policy = SCHED_OTHER;
    struct sched_param param = {};
    
    switch (priority) {
        case ThreadPriority::Idle:
            policy = SCHED_IDLE;
            param.sched_priority = 0;
            break;
        case ThreadPriority::BelowNormal:
            param.sched_priority = -5;
            break;
        case ThreadPriority::Normal:
            param.sched_priority = 0;
            break;
        case ThreadPriority::AboveNormal:
            param.sched_priority = 5;
            break;
        case ThreadPriority::High:
            param.sched_priority = 10;
            break;
        case ThreadPriority::Realtime:
            policy = SCHED_RR;
            param.sched_priority = 50;
            break;
    }
    
    return pthread_setschedparam(pthread_self(), policy, &param) == 0;
    
#elif defined(_WIN32)
    int win_priority = THREAD_PRIORITY_NORMAL;
    
    switch (priority) {
        case ThreadPriority::Idle:
            win_priority = THREAD_PRIORITY_IDLE;
            break;
        case ThreadPriority::BelowNormal:
            win_priority = THREAD_PRIORITY_BELOW_NORMAL;
            break;
        case ThreadPriority::Normal:
            win_priority = THREAD_PRIORITY_NORMAL;
            break;
        case ThreadPriority::AboveNormal:
            win_priority = THREAD_PRIORITY_ABOVE_NORMAL;
            break;
        case ThreadPriority::High:
            win_priority = THREAD_PRIORITY_HIGHEST;
            break;
        case ThreadPriority::Realtime:
            win_priority = THREAD_PRIORITY_TIME_CRITICAL;
            break;
    }
    
    return SetThreadPriority(GetCurrentThread(), win_priority) != 0;
    
#else
    return false;
#endif
}

ThreadPriority get_thread_priority(std::thread::id thread_id) {
    return get_thread_priority();
}

ThreadPriority get_thread_priority() {
    // Simplified implementation
    return ThreadPriority::Normal;
}

// ============================================================================
// Utility Functions - Simplified Implementation
// ============================================================================

int get_current_cpu_core() {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}

void yield_to_hyperthread_sibling() {
    std::this_thread::yield();
}

void cpu_pause(int cycles) {
    for (int i = 0; i < cycles; ++i) {
#ifdef __x86_64__
        __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory");
#else
        // Fallback
        std::this_thread::yield();
#endif
    }
}

size_t get_cpu_cache_line_size() {
    return 64; // Common cache line size
}

CacheSizes get_cpu_cache_sizes() {
    CacheSizes sizes = {};
    
    // Default values - real implementation would query hardware
    sizes.l1_instruction = 32 * 1024;
    sizes.l1_data = 32 * 1024;
    sizes.l2 = 256 * 1024;
    sizes.l3 = 8 * 1024 * 1024;
    
    return sizes;
}

bool same_cache_line(const void* addr1, const void* addr2) {
    size_t cache_line_size = get_cpu_cache_line_size();
    uintptr_t line1 = reinterpret_cast<uintptr_t>(addr1) / cache_line_size;
    uintptr_t line2 = reinterpret_cast<uintptr_t>(addr2) / cache_line_size;
    return line1 == line2;
}

void* align_to_cache_line(void* addr) {
    size_t cache_line_size = get_cpu_cache_line_size();
    uintptr_t aligned = (reinterpret_cast<uintptr_t>(addr) + cache_line_size - 1) & 
                       ~(cache_line_size - 1);
    return reinterpret_cast<void*>(aligned);
}

size_t align_size_to_cache_line(size_t size) {
    size_t cache_line_size = get_cpu_cache_line_size();
    return (size + cache_line_size - 1) & ~(cache_line_size - 1);
}

// ============================================================================
// Performance Monitoring - Simplified Implementation
// ============================================================================

ThreadPerfCounters get_thread_perf_counters(std::thread::id thread_id) {
    ThreadPerfCounters counters = {};
    
    // Placeholder values - real implementation would use hardware counters
    counters.cpu_cycles = 1000000;
    counters.instructions = 800000;
    counters.cache_misses = 1000;
    counters.branch_misses = 100;
    counters.context_switches = 10;
    counters.cpu_utilization = 0.5;
    counters.current_core = get_current_cpu_core();
    counters.migrations = 2;
    
    return counters;
}

bool start_perf_monitoring() {
    // Placeholder - would initialize performance monitoring
    return true;
}

bool stop_perf_monitoring() {
    // Placeholder - would stop performance monitoring
    return true;
}

std::vector<double> get_cpu_utilization_per_core() {
    auto topology = get_cpu_topology();
    std::vector<double> utilization(topology.num_logical_cores, 0.5); // Placeholder
    return utilization;
}

LoadAverage get_load_average() {
    LoadAverage load = {};
    
#ifdef __linux__
    std::ifstream loadavg("/proc/loadavg");
    if (loadavg.is_open()) {
        loadavg >> load.load_1min >> load.load_5min >> load.load_15min;
    }
#else
    // Default values
    load.load_1min = 1.0;
    load.load_5min = 1.0;
    load.load_15min = 1.0;
#endif
    
    return load;
}

std::string diagnose_thread_performance() {
    std::ostringstream diagnosis;
    
    auto topology = get_cpu_topology();
    auto affinity = get_thread_affinity();
    auto load = get_load_average();
    
    diagnosis << "=== Thread Performance Diagnosis ===\n";
    diagnosis << "CPU Topology:\n";
    diagnosis << "  Logical cores: " << topology.num_logical_cores << "\n";
    diagnosis << "  Physical cores: " << topology.num_cores << "\n";
    diagnosis << "  Hyperthreading: " << (topology.has_hyperthreading ? "Yes" : "No") << "\n";
    diagnosis << "  Hybrid cores: " << (topology.has_hybrid_cores ? "Yes" : "No") << "\n";
    
    diagnosis << "\nThread Affinity:\n";
    diagnosis << "  Allowed cores: ";
    for (int core : affinity) {
        diagnosis << core << " ";
    }
    diagnosis << "\n";
    
    diagnosis << "\nSystem Load:\n";
    diagnosis << "  1-minute: " << load.load_1min << "\n";
    diagnosis << "  5-minute: " << load.load_5min << "\n";
    diagnosis << "  15-minute: " << load.load_15min << "\n";
    
    diagnosis << "\nRecommendations:\n";
    if (load.load_1min > topology.num_logical_cores) {
        diagnosis << "- System is overloaded, consider reducing thread count\n";
    }
    
    if (affinity.size() == topology.num_logical_cores) {
        diagnosis << "- Consider pinning threads to specific cores for better performance\n";
    }
    
    if (topology.has_hybrid_cores) {
        diagnosis << "- Use P-cores for latency-critical threads\n";
    }
    
    return diagnosis.str();
}

// ============================================================================
// AffinityThreadPool - Simplified Stub Implementation
// ============================================================================

AffinityThreadPool::AffinityThreadPool() : AffinityThreadPool(PoolConfig{}) {}

AffinityThreadPool::AffinityThreadPool(const PoolConfig& config) 
    : config_(config), shutdown_(false) {
    
    size_t num_threads = config_.num_threads;
    if (num_threads == 0) {
        num_threads = calculate_optimal_thread_count();
    }
    
    thread_affinities_ = calculate_thread_affinities();
    
    // Create worker threads
    for (size_t i = 0; i < num_threads; ++i) {
        threads_.emplace_back(&AffinityThreadPool::worker_thread, this, i);
    }
}

AffinityThreadPool::~AffinityThreadPool() {
    shutdown_ = true;
    queue_cv_.notify_all();
    
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

AffinityThreadPool::AffinityThreadPool(AffinityThreadPool&& other) noexcept
    : config_(std::move(other.config_)),
      threads_(std::move(other.threads_)),
      thread_affinities_(std::move(other.thread_affinities_)),
      shutdown_(other.shutdown_.load()) {
    other.shutdown_ = true;
}

AffinityThreadPool& AffinityThreadPool::operator=(AffinityThreadPool&& other) noexcept {
    if (this != &other) {
        // Clean up current state
        shutdown_ = true;
        queue_cv_.notify_all();
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        // Move from other
        config_ = std::move(other.config_);
        threads_ = std::move(other.threads_);
        thread_affinities_ = std::move(other.thread_affinities_);
        shutdown_ = other.shutdown_.load();
        other.shutdown_ = true;
    }
    return *this;
}

std::vector<AffinityConfig> AffinityThreadPool::get_thread_affinities() const {
    return thread_affinities_;
}

void AffinityThreadPool::reconfigure(const PoolConfig& config) {
    // Simplified - would need to rebuild thread pool
    config_ = config;
}

void AffinityThreadPool::wait_for_completion() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return task_queue_.empty(); });
}

AffinityThreadPool::PoolStats AffinityThreadPool::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void AffinityThreadPool::worker_thread(size_t thread_index) {
    // Set thread affinity
    setup_thread_affinity(thread_index);
    
    while (!shutdown_) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return shutdown_ || !task_queue_.empty(); });
            
            if (shutdown_) break;
            
            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
        }
        
        if (task) {
            task();
            
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.tasks_completed++;
            }
        }
    }
}

void AffinityThreadPool::setup_thread_affinity(size_t thread_index) {
    if (config_.pin_threads && thread_index < thread_affinities_.size()) {
        set_thread_affinity(thread_affinities_[thread_index]);
    }
    
    set_thread_priority(config_.priority);
}

size_t AffinityThreadPool::calculate_optimal_thread_count() const {
    auto topology = get_cpu_topology();
    
    switch (config_.thread_type) {
        case ThreadType::LowLatency:
            return get_performance_cores().size();
        case ThreadType::HighThroughput:
            return topology.num_logical_cores;
        case ThreadType::Background:
            return std::max(1, topology.num_logical_cores / 4);
        default:
            return topology.num_logical_cores;
    }
}

std::vector<AffinityConfig> AffinityThreadPool::calculate_thread_affinities() const {
    size_t num_threads = config_.num_threads;
    if (num_threads == 0) {
        num_threads = calculate_optimal_thread_count();
    }
    
    return distribute_threads_across_cores(num_threads, config_.thread_type, 
                                          config_.exclusive_cores);
}

// ============================================================================
// Performance Config and Optimization - Simplified Implementation
// ============================================================================

bool apply_performance_config(const PerformanceConfig& config) {
    // Placeholder - would require privileged access to apply system-wide settings
    (void)config;
    return false;
}

double get_cpu_frequency(int core_id) {
    // Placeholder implementation
    (void)core_id;
    return 2400.0; // MHz
}

bool set_cpu_governor(const std::string& governor) {
    // Placeholder - would require root access
    (void)governor;
    return false;
}

} // namespace perf
} // namespace psyne