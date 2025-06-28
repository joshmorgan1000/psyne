#include "../../include/psyne/perf/huge_pages.hpp"
#include <cstring>
#include <mutex>
#include <algorithm>
#include <sstream>

// Platform-specific includes
#ifdef __linux__
    #include <sys/mman.h>
    #include <unistd.h>
    #include <fstream>
    #include <numa.h>
#elif defined(__APPLE__)
    #include <sys/mman.h>
    #include <unistd.h>
    #include <sys/sysctl.h>
#elif defined(_WIN32)
    #include <windows.h>
    #include <memoryapi.h>
#endif

namespace psyne {
namespace perf {

// ============================================================================
// System Information
// ============================================================================

HugePageInfo get_huge_page_info() {
    HugePageInfo info;
    
#ifdef __linux__
    // Check /proc/meminfo for huge page information
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    
    while (std::getline(meminfo, line)) {
        if (line.find("HugePages_Total:") == 0) {
            info.total_2mb_pages = std::stoul(line.substr(line.find_last_of(' ') + 1));
            info.supported = true;
        } else if (line.find("HugePages_Free:") == 0) {
            info.free_2mb_pages = std::stoul(line.substr(line.find_last_of(' ') + 1));
        } else if (line.find("Hugepagesize:") == 0) {
            // This gives us the default huge page size
            std::string size_str = line.substr(line.find_last_of(' ') + 1);
            if (size_str.find("kB") != std::string::npos) {
                info.default_page_size = std::stoul(size_str) * 1024;
            }
        }
    }
    
    // Check for transparent huge pages
    std::ifstream thp("/sys/kernel/mm/transparent_hugepage/enabled");
    if (thp.is_open()) {
        std::string thp_status;
        std::getline(thp, thp_status);
        info.transparent_enabled = thp_status.find("[always]") != std::string::npos ||
                                  thp_status.find("[madvise]") != std::string::npos;
    }
    
    // Check NUMA support
    info.numa_available = numa_available() != -1;
    if (info.numa_available) {
        info.numa_node_count = numa_num_configured_nodes();
        info.current_numa_node = numa_node_of_cpu(sched_getcpu());
    }
    
#elif defined(__APPLE__)
    // macOS doesn't have traditional huge pages, but has large pages
    size_t page_size = static_cast<size_t>(getpagesize());
    info.default_page_size = page_size;
    
    // Check for large page support (not commonly available on macOS)
    info.supported = false;
    info.transparent_enabled = false;
    
#elif defined(_WIN32)
    // Windows large page support
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    info.default_page_size = si.dwPageSize;
    
    // Check for large page privilege
    HANDLE token;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
        LUID luid;
        if (LookupPrivilegeValue(nullptr, SE_LOCK_MEMORY_NAME, &luid)) {
            PRIVILEGE_SET ps;
            ps.PrivilegeCount = 1;
            ps.Control = PRIVILEGE_SET_ALL_NECESSARY;
            ps.Privilege[0].Luid = luid;
            ps.Privilege[0].Attributes = SE_PRIVILEGE_ENABLED;
            
            BOOL result;
            if (PrivilegeCheck(token, &ps, &result)) {
                info.supported = result != FALSE;
            }
        }
        CloseHandle(token);
    }
#else
    // Default fallback
    info.default_page_size = 4096;
    info.supported = false;
#endif
    
    return info;
}

bool is_huge_page_size_available(HugePageSize page_size) {
    if (page_size == HugePageSize::None || page_size == HugePageSize::Auto) {
        return true;
    }
    
    const auto info = get_huge_page_info();
    if (!info.supported) {
        return false;
    }
    
#ifdef __linux__
    // Check if the specific page size is available
    size_t size_bytes = static_cast<size_t>(page_size);
    
    // Common huge page sizes on Linux
    return size_bytes == 2 * 1024 * 1024 || // 2MB
           size_bytes == 1024 * 1024 * 1024; // 1GB
#else
    return false;
#endif
}

HugePageSize get_optimal_huge_page_size(size_t allocation_size) {
    const auto info = get_huge_page_info();
    if (!info.supported) {
        return HugePageSize::None;
    }
    
    // For very large allocations, prefer 1GB pages if available
    if (allocation_size >= 512 * 1024 * 1024 && 
        is_huge_page_size_available(HugePageSize::Page1GB)) {
        return HugePageSize::Page1GB;
    }
    
    // For medium to large allocations, use 2MB pages
    if (allocation_size >= 4 * 1024 * 1024 &&
        is_huge_page_size_available(HugePageSize::Page2MB)) {
        return HugePageSize::Page2MB;
    }
    
    // For ARM64, consider 64KB pages for smaller allocations
#ifdef __aarch64__
    if (allocation_size >= 256 * 1024 &&
        is_huge_page_size_available(HugePageSize::Page64KB)) {
        return HugePageSize::Page64KB;
    }
#endif
    
    return HugePageSize::None;
}

HugePageAllocation allocate_huge_pages(size_t size, HugePagePolicy policy,
                                      HugePageSize preferred_size, int numa_node) {
    HugePageAllocation result;
    result.size = size;
    
    if (policy == HugePagePolicy::Never) {
        // Standard allocation
        result.ptr = std::aligned_alloc(4096, size);
        if (result.ptr) {
            result.actual_page_size = 4096;
            result.is_huge_page = false;
        } else {
            result.error_message = "Standard allocation failed";
        }
        return result;
    }
    
    // Determine page size to use
    HugePageSize page_size = preferred_size;
    if (page_size == HugePageSize::Auto) {
        page_size = get_optimal_huge_page_size(size);
    }
    
    size_t page_size_bytes = static_cast<size_t>(page_size);
    
    // Align size to page boundary
    if (page_size_bytes > 0) {
        size_t aligned_size = ((size + page_size_bytes - 1) / page_size_bytes) * page_size_bytes;
        result.size = aligned_size;
    }
    
#ifdef __linux__
    // Try to allocate with huge pages
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    
    if (page_size == HugePageSize::Page2MB) {
        flags |= MAP_HUGETLB | MAP_HUGE_2MB;
    } else if (page_size == HugePageSize::Page1GB) {
        flags |= MAP_HUGETLB | MAP_HUGE_1GB;
    }
    
    result.ptr = mmap(nullptr, result.size, PROT_READ | PROT_WRITE, flags, -1, 0);
    
    if (result.ptr != MAP_FAILED) {
        result.is_huge_page = (page_size != HugePageSize::None);
        result.actual_page_size = page_size_bytes;
        result.page_type = page_size;
        
        // Set NUMA affinity if requested
        if (numa_node >= 0 && numa_node < numa_num_configured_nodes()) {
            if (mbind(result.ptr, result.size, MPOL_BIND, 
                     numa_get_membind_compat(), numa_num_configured_nodes(), 0) == 0) {
                result.is_numa_local = true;
            }
        }
    } else {
        // Huge page allocation failed
        if (policy == HugePagePolicy::Required) {
            result.error_message = "Huge page allocation required but failed";
            return result;
        }
        
        // Fall back to standard allocation
        result.ptr = mmap(nullptr, result.size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        
        if (result.ptr != MAP_FAILED) {
            result.is_huge_page = false;
            result.actual_page_size = 4096;
            result.page_type = HugePageSize::None;
        } else {
            result.error_message = "Both huge page and standard allocation failed";
        }
    }
    
#elif defined(__APPLE__)
    // macOS doesn't support huge pages in the same way
    result.ptr = mmap(nullptr, result.size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (result.ptr != MAP_FAILED) {
        result.is_huge_page = false;
        result.actual_page_size = getpagesize();
        result.page_type = HugePageSize::None;
    } else {
        result.error_message = "Memory allocation failed";
    }
    
#elif defined(_WIN32)
    // Windows large page support
    SIZE_T large_page_size = GetLargePageMinimum();
    
    if (policy != HugePagePolicy::Never && large_page_size > 0 && 
        result.size >= large_page_size) {
        
        // Try large page allocation
        result.ptr = VirtualAlloc(nullptr, result.size, 
                                 MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                                 PAGE_READWRITE);
        
        if (result.ptr) {
            result.is_huge_page = true;
            result.actual_page_size = large_page_size;
            result.page_type = HugePageSize::Page2MB; // Approximate
        }
    }
    
    if (!result.ptr) {
        // Fall back to standard allocation
        if (policy == HugePagePolicy::Required) {
            result.error_message = "Large page allocation required but failed";
            return result;
        }
        
        result.ptr = VirtualAlloc(nullptr, result.size, MEM_COMMIT | MEM_RESERVE,
                                 PAGE_READWRITE);
        
        if (result.ptr) {
            result.is_huge_page = false;
            result.actual_page_size = 4096;
            result.page_type = HugePageSize::None;
        } else {
            result.error_message = "Memory allocation failed";
        }
    }
    
#else
    // Fallback for unsupported platforms
    result.ptr = std::aligned_alloc(4096, result.size);
    if (result.ptr) {
        result.is_huge_page = false;
        result.actual_page_size = 4096;
        result.page_type = HugePageSize::None;
    } else {
        result.error_message = "Memory allocation failed";
    }
#endif
    
    return result;
}

void free_huge_pages(const HugePageAllocation& allocation) {
    if (!allocation.ptr) {
        return;
    }
    
#ifdef __linux__
    munmap(allocation.ptr, allocation.size);
#elif defined(__APPLE__)
    munmap(allocation.ptr, allocation.size);
#elif defined(_WIN32)
    VirtualFree(allocation.ptr, 0, MEM_RELEASE);
#else
    std::free(allocation.ptr);
#endif
}

HugePageAllocation reallocate_huge_pages(const HugePageAllocation& old_allocation,
                                        size_t new_size, HugePagePolicy policy) {
    // Allocate new memory
    auto new_allocation = allocate_huge_pages(new_size, policy, 
                                            old_allocation.page_type);
    
    if (new_allocation.ptr && old_allocation.ptr) {
        // Copy old data
        size_t copy_size = std::min(old_allocation.size, new_size);
        std::memcpy(new_allocation.ptr, old_allocation.ptr, copy_size);
    }
    
    return new_allocation;
}

// ============================================================================
// Buffer Allocation Advice
// ============================================================================

BufferAllocationAdvice get_buffer_allocation_advice(size_t requested_size,
                                                   size_t message_count_estimate,
                                                   bool high_throughput) {
    BufferAllocationAdvice advice;
    
    const auto info = get_huge_page_info();
    
    // Calculate recommended size based on usage patterns
    size_t base_size = requested_size;
    
    // Add overhead for message headers and alignment
    if (message_count_estimate > 0) {
        base_size += message_count_estimate * 64; // Estimate 64 bytes overhead per message
    }
    
    // Align to cache line boundaries
    advice.alignment = 64;
    
    // Round up to huge page boundaries if beneficial
    advice.page_size = get_optimal_huge_page_size(base_size);
    
    if (advice.page_size != HugePageSize::None) {
        size_t page_size_bytes = static_cast<size_t>(advice.page_size);
        advice.recommended_size = ((base_size + page_size_bytes - 1) / page_size_bytes) * page_size_bytes;
        advice.policy = high_throughput ? HugePagePolicy::TryBest : HugePagePolicy::TryBest;
    } else {
        // Align to regular page boundaries
        advice.recommended_size = ((base_size + 4095) / 4096) * 4096;
        advice.policy = HugePagePolicy::Never;
    }
    
    // NUMA recommendations
    advice.use_numa_affinity = info.numa_available && (base_size > 1024 * 1024);
    advice.numa_node = info.current_numa_node;
    
    // Generate reasoning
    std::ostringstream reasoning;
    reasoning << "For " << (base_size / 1024 / 1024) << "MB allocation: ";
    
    if (advice.page_size != HugePageSize::None) {
        reasoning << "Using " << (static_cast<size_t>(advice.page_size) / 1024 / 1024) 
                 << "MB huge pages to reduce TLB misses. ";
    } else {
        reasoning << "Using standard 4KB pages (allocation too small for huge pages). ";
    }
    
    if (advice.use_numa_affinity) {
        reasoning << "NUMA affinity recommended for large allocation. ";
    }
    
    advice.reasoning = reasoning.str();
    
    return advice;
}

// ============================================================================
// NUMA Support (stubs for non-Linux platforms)
// ============================================================================

std::vector<NumaNodeInfo> get_numa_topology() {
    std::vector<NumaNodeInfo> nodes;
    
#ifdef __linux__
    if (numa_available() == -1) {
        return nodes;
    }
    
    int num_nodes = numa_num_configured_nodes();
    for (int i = 0; i < num_nodes; ++i) {
        NumaNodeInfo node;
        node.node_id = i;
        node.total_memory = numa_node_size64(i, nullptr);
        node.free_memory = 0; // Would need to parse /proc/meminfo per node
        node.cpu_count = 0;   // Would need to count CPUs per node
        node.is_local = (i == numa_node_of_cpu(sched_getcpu()));
        node.memory_bandwidth = 100.0; // Placeholder
        nodes.push_back(node);
    }
#endif
    
    return nodes;
}

int get_current_numa_node() {
#ifdef __linux__
    if (numa_available() != -1) {
        return numa_node_of_cpu(sched_getcpu());
    }
#endif
    return -1;
}

bool set_numa_affinity(int numa_node) {
#ifdef __linux__
    if (numa_available() != -1 && numa_node >= 0) {
        struct bitmask* mask = numa_allocate_nodemask();
        numa_bitmask_setbit(mask, numa_node);
        int result = numa_set_membind(mask);
        numa_free_nodemask(mask);
        return result == 0;
    }
#else
    (void)numa_node;
#endif
    return false;
}

void* allocate_numa_memory(size_t size, int numa_node) {
#ifdef __linux__
    if (numa_available() != -1 && numa_node >= 0) {
        return numa_alloc_onnode(size, numa_node);
    }
#else
    (void)numa_node;
#endif
    return std::aligned_alloc(64, size);
}

void free_numa_memory(void* ptr, size_t size) {
#ifdef __linux__
    if (numa_available() != -1) {
        numa_free(ptr, size);
        return;
    }
#else
    (void)size;
#endif
    std::free(ptr);
}

// ============================================================================
// Utility Functions
// ============================================================================

bool is_address_on_huge_page(const void* ptr) {
#ifdef __linux__
    // Check /proc/self/smaps for huge page information
    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    while (std::getline(smaps, line)) {
        if (line.find("-") != std::string::npos) {
            // Parse address range
            size_t dash_pos = line.find("-");
            uintptr_t start = std::stoull(line.substr(0, dash_pos), nullptr, 16);
            uintptr_t end = std::stoull(line.substr(dash_pos + 1), nullptr, 16);
            
            if (addr >= start && addr < end) {
                // Found the mapping, check for huge pages in following lines
                while (std::getline(smaps, line) && !line.empty()) {
                    if (line.find("KernelPageSize:") == 0) {
                        std::string size_str = line.substr(line.find_last_of(' ') + 1);
                        size_t page_size = std::stoul(size_str) * 1024;
                        return page_size > 4096;
                    }
                }
                break;
            }
        }
    }
#else
    (void)ptr;
#endif
    return false;
}

size_t get_page_size_for_address(const void* ptr) {
#ifdef __linux__
    if (is_address_on_huge_page(ptr)) {
        return 2 * 1024 * 1024; // Assume 2MB if huge page detected
    }
#else
    (void)ptr;
#endif
    return 4096;
}

double calculate_huge_page_overhead(size_t allocation_size, HugePageSize page_size) {
    if (page_size == HugePageSize::None) {
        return 0.0;
    }
    
    size_t page_size_bytes = static_cast<size_t>(page_size);
    size_t aligned_size = ((allocation_size + page_size_bytes - 1) / page_size_bytes) * page_size_bytes;
    
    return static_cast<double>(aligned_size - allocation_size) / allocation_size;
}

MemoryBenchmarkResult benchmark_memory_performance(void* ptr, size_t size, size_t iterations) {
    MemoryBenchmarkResult result = {};
    
    // This is a placeholder implementation
    // A real implementation would perform actual memory benchmarks
    result.sequential_read_bandwidth = 50.0;  // GB/s
    result.sequential_write_bandwidth = 45.0; // GB/s
    result.random_read_latency = 100.0;       // nanoseconds
    result.random_write_latency = 120.0;      // nanoseconds
    result.tlb_misses_per_mb = is_address_on_huge_page(ptr) ? 1 : 512;
    
    (void)size;
    (void)iterations;
    
    return result;
}

std::string get_huge_page_configuration_advice() {
    const auto info = get_huge_page_info();
    std::ostringstream advice;
    
    advice << "Huge Page Configuration Advice:\n";
    
    if (!info.supported) {
        advice << "- Huge pages are not supported or not configured on this system\n";
        advice << "- Consider enabling huge pages in kernel configuration\n";
    } else {
        advice << "- Huge pages are available\n";
        advice << "- Available 2MB pages: " << info.free_2mb_pages << "/" << info.total_2mb_pages << "\n";
        
        if (info.transparent_enabled) {
            advice << "- Transparent huge pages are enabled\n";
        } else {
            advice << "- Consider enabling transparent huge pages for automatic management\n";
        }
    }
    
    if (info.numa_available) {
        advice << "- NUMA is available with " << info.numa_node_count << " nodes\n";
        advice << "- Current node: " << info.current_numa_node << "\n";
    } else {
        advice << "- NUMA is not available (single node system)\n";
    }
    
    return advice.str();
}

// ============================================================================
// HugePagePool Implementation (simplified stub)
// ============================================================================

HugePagePool::HugePagePool() : HugePagePool(PoolConfig{}) {}

HugePagePool::HugePagePool(const PoolConfig& config) : config_(config) {
    // Initialize pool with initial allocation
    if (config_.initial_size > 0) {
        expand_pool(config_.initial_size);
    }
}

HugePagePool::~HugePagePool() {
    reset_pool();
}

HugePagePool::HugePagePool(HugePagePool&& other) noexcept 
    : config_(std::move(other.config_)),
      blocks_(std::move(other.blocks_)),
      free_blocks_(std::move(other.free_blocks_)),
      stats_(std::move(other.stats_)) {
}

HugePagePool& HugePagePool::operator=(HugePagePool&& other) noexcept {
    if (this != &other) {
        reset_pool();
        config_ = std::move(other.config_);
        blocks_ = std::move(other.blocks_);
        free_blocks_ = std::move(other.free_blocks_);
        stats_ = std::move(other.stats_);
    }
    return *this;
}

void* HugePagePool::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Simplified implementation - just allocate directly for now
    // A real implementation would manage a pool of pre-allocated huge page memory
    
    auto allocation = allocate_huge_pages(size, config_.policy, 
                                        config_.preferred_page_size, 
                                        config_.numa_node);
    
    if (allocation.ptr) {
        stats_.allocated_size += allocation.size;
        stats_.allocation_count++;
        return allocation.ptr;
    }
    
    return nullptr;
}

void HugePagePool::deallocate(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (ptr) {
        // Simplified - in a real implementation, we'd return memory to the pool
        HugePageAllocation allocation;
        allocation.ptr = ptr;
        allocation.size = size;
        free_huge_pages(allocation);
        
        stats_.allocated_size -= size;
        stats_.deallocation_count++;
    }
}

void HugePagePool::expand_pool(size_t additional_size) {
    // Placeholder implementation
    stats_.total_size += additional_size;
}

void HugePagePool::shrink_pool(size_t target_size) {
    // Placeholder implementation
    if (target_size < stats_.total_size) {
        stats_.total_size = target_size;
    }
}

void HugePagePool::reset_pool() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Free all blocks
    blocks_.clear();
    free_blocks_.clear();
    
    stats_ = {};
}

HugePagePool::PoolStats HugePagePool::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void HugePagePool::set_config(const PoolConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

void* HugePagePool::allocate_new_block(size_t size) {
    // Placeholder implementation
    return allocate_huge_pages(size, config_.policy).ptr;
}

void HugePagePool::merge_free_blocks() {
    // Placeholder implementation
}

size_t HugePagePool::align_size(size_t size, size_t alignment) const {
    if (alignment == 0) alignment = config_.alignment;
    return ((size + alignment - 1) / alignment) * alignment;
}

} // namespace perf
} // namespace psyne