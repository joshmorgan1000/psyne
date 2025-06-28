#pragma once

// Huge page support for improved memory performance
// Large pages can significantly reduce TLB misses for large message buffers

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

namespace psyne {
namespace perf {

// Huge page sizes (platform dependent)
enum class HugePageSize : size_t {
    None = 0,
    Page2MB = 2 * 1024 * 1024,      // 2MB pages (common on x86_64)
    Page1GB = 1024 * 1024 * 1024,   // 1GB pages (x86_64 with 1GB support)
    Page64KB = 64 * 1024,           // 64KB pages (ARM64)
    Page2MB_ARM = 2 * 1024 * 1024,  // 2MB pages (ARM64)
    Auto = SIZE_MAX                  // Automatically select best size
};

// Huge page allocation policies
enum class HugePagePolicy {
    Never,          // Never use huge pages
    TryBest,        // Try to use huge pages, fallback to normal if failed
    Required,       // Require huge pages, fail if not available
    Transparent     // Use transparent huge pages (if available)
};

// Huge page allocation result
struct HugePageAllocation {
    void* ptr = nullptr;
    size_t size = 0;
    size_t actual_page_size = 0;
    HugePageSize page_type = HugePageSize::None;
    bool is_huge_page = false;
    bool is_numa_local = false;
    std::string error_message;
};

// Huge page system information
struct HugePageInfo {
    bool supported = false;
    bool transparent_enabled = false;
    size_t total_2mb_pages = 0;
    size_t free_2mb_pages = 0;
    size_t total_1gb_pages = 0;
    size_t free_1gb_pages = 0;
    size_t default_page_size = 4096;
    
    // NUMA information
    bool numa_available = false;
    int numa_node_count = 0;
    int current_numa_node = -1;
};

// ============================================================================
// Huge Page Management
// ============================================================================

// Get system huge page information
HugePageInfo get_huge_page_info();

// Check if huge pages are available for a given size
bool is_huge_page_size_available(HugePageSize page_size);

// Get the optimal huge page size for a given allocation size
HugePageSize get_optimal_huge_page_size(size_t allocation_size);

// Allocate memory with huge pages
HugePageAllocation allocate_huge_pages(size_t size, 
                                      HugePagePolicy policy = HugePagePolicy::TryBest,
                                      HugePageSize preferred_size = HugePageSize::Auto,
                                      int numa_node = -1);

// Free huge page allocation
void free_huge_pages(const HugePageAllocation& allocation);

// Reallocate with huge pages (may move memory)
HugePageAllocation reallocate_huge_pages(const HugePageAllocation& old_allocation,
                                        size_t new_size,
                                        HugePagePolicy policy = HugePagePolicy::TryBest);

// ============================================================================
// Memory Advisor for Message Buffers
// ============================================================================

// Recommendations for message buffer allocation
struct BufferAllocationAdvice {
    size_t recommended_size;        // Recommended buffer size (aligned to page boundaries)
    HugePageSize page_size;         // Recommended page size
    HugePagePolicy policy;          // Recommended policy
    bool use_numa_affinity;         // Whether to use NUMA affinity
    int numa_node;                  // Recommended NUMA node (-1 for current)
    size_t alignment;               // Recommended memory alignment
    std::string reasoning;          // Human-readable explanation
};

// Get allocation advice for message buffers
BufferAllocationAdvice get_buffer_allocation_advice(size_t requested_size,
                                                   size_t message_count_estimate = 0,
                                                   bool high_throughput = true);

// ============================================================================
// NUMA Awareness
// ============================================================================

// NUMA node information
struct NumaNodeInfo {
    int node_id;
    size_t total_memory;
    size_t free_memory;
    size_t cpu_count;
    bool is_local;          // True if this is the current thread's node
    double memory_bandwidth; // Estimated memory bandwidth (GB/s)
};

// Get NUMA topology information
std::vector<NumaNodeInfo> get_numa_topology();

// Get current thread's NUMA node
int get_current_numa_node();

// Set thread affinity to a specific NUMA node
bool set_numa_affinity(int numa_node);

// Allocate memory on a specific NUMA node
void* allocate_numa_memory(size_t size, int numa_node);

// Free NUMA-allocated memory
void free_numa_memory(void* ptr, size_t size);

// ============================================================================
// Memory Pool with Huge Pages
// ============================================================================

class HugePagePool {
public:
    struct PoolConfig {
        size_t initial_size = 64 * 1024 * 1024;  // 64MB initial pool
        size_t max_size = 1024 * 1024 * 1024;    // 1GB maximum pool size
        HugePagePolicy policy = HugePagePolicy::TryBest;
        HugePageSize preferred_page_size = HugePageSize::Auto;
        bool numa_aware = true;
        int numa_node = -1;  // -1 for current node
        size_t alignment = 64;  // Cache line alignment
    };

public:
    HugePagePool();
    explicit HugePagePool(const PoolConfig& config);
    ~HugePagePool();
    
    // Non-copyable, movable
    HugePagePool(const HugePagePool&) = delete;
    HugePagePool& operator=(const HugePagePool&) = delete;
    HugePagePool(HugePagePool&&) noexcept;
    HugePagePool& operator=(HugePagePool&&) noexcept;
    
    // Allocate memory from the pool
    void* allocate(size_t size, size_t alignment = 0);
    
    // Deallocate memory back to the pool
    void deallocate(void* ptr, size_t size);
    
    // Pool management
    void expand_pool(size_t additional_size);
    void shrink_pool(size_t target_size);
    void reset_pool();  // Free all allocations and reset
    
    // Pool statistics
    struct PoolStats {
        size_t total_size;
        size_t allocated_size;
        size_t free_size;
        size_t allocation_count;
        size_t deallocation_count;
        size_t huge_page_size;
        bool using_huge_pages;
        int numa_node;
    };
    
    PoolStats get_stats() const;
    
    // Configuration
    const PoolConfig& get_config() const { return config_; }
    void set_config(const PoolConfig& config);

private:
    struct PoolBlock {
        void* ptr;
        size_t size;
        HugePageAllocation allocation;
    };
    
    struct FreeBlock {
        void* ptr;
        size_t size;
        size_t offset;
    };
    
    PoolConfig config_;
    std::vector<std::unique_ptr<PoolBlock>> blocks_;
    std::vector<FreeBlock> free_blocks_;
    
    mutable std::mutex mutex_;
    PoolStats stats_;
    
    void* allocate_new_block(size_t size);
    void merge_free_blocks();
    size_t align_size(size_t size, size_t alignment) const;
};

// ============================================================================
// Utility Functions
// ============================================================================

// Check if an address is on a huge page
bool is_address_on_huge_page(const void* ptr);

// Get the page size for a given address
size_t get_page_size_for_address(const void* ptr);

// Calculate memory overhead for huge pages vs normal pages
double calculate_huge_page_overhead(size_t allocation_size, HugePageSize page_size);

// Benchmark memory performance with different page configurations
struct MemoryBenchmarkResult {
    double sequential_read_bandwidth;   // GB/s
    double sequential_write_bandwidth;  // GB/s
    double random_read_latency;         // nanoseconds
    double random_write_latency;        // nanoseconds
    size_t tlb_misses_per_mb;          // Estimated TLB misses per MB
};

MemoryBenchmarkResult benchmark_memory_performance(void* ptr, size_t size,
                                                   size_t iterations = 1000);

// Get system-specific huge page configuration recommendations
std::string get_huge_page_configuration_advice();

} // namespace perf
} // namespace psyne