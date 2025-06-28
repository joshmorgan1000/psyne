#include "../../include/psyne/perf/prefetch.hpp"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <thread>
#include <unordered_map>

// Platform-specific prefetch intrinsics
#ifdef __x86_64__
    #include <xmmintrin.h>  // For _mm_prefetch
    #include <emmintrin.h>
#elif defined(__aarch64__)
    #include <arm_acle.h>
#endif

namespace psyne {
namespace perf {

// ============================================================================
// Basic Prefetch Operations
// ============================================================================

void prefetch(const void* addr, PrefetchHint hint, CacheLevel level) {
    if (!addr) return;
    
#ifdef __x86_64__
    // Use x86 prefetch instructions
    switch (hint) {
        case PrefetchHint::Read:
        case PrefetchHint::ReadWrite:
            switch (level) {
                case CacheLevel::L1:
                    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
                    break;
                case CacheLevel::L2:
                    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T1);
                    break;
                case CacheLevel::L3:
                    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T2);
                    break;
                case CacheLevel::Auto:
                default:
                    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
                    break;
            }
            break;
        case PrefetchHint::Write:
            // x86 doesn't have dedicated write prefetch, use read prefetch
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
            break;
        case PrefetchHint::NonTemporal:
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_NTA);
            break;
        case PrefetchHint::Stream:
        case PrefetchHint::Temporal:
        default:
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
            break;
    }
    
#elif defined(__aarch64__)
    // Use ARM prefetch instructions
    switch (hint) {
        case PrefetchHint::Read:
        case PrefetchHint::ReadWrite:
        case PrefetchHint::Temporal:
            __builtin_prefetch(addr, 0, 3); // Read, high temporal locality
            break;
        case PrefetchHint::Write:
            __builtin_prefetch(addr, 1, 3); // Write, high temporal locality
            break;
        case PrefetchHint::NonTemporal:
        case PrefetchHint::Stream:
            __builtin_prefetch(addr, 0, 1); // Read, low temporal locality
            break;
    }
    
#else
    // Generic builtin prefetch
    int rw = (hint == PrefetchHint::Write) ? 1 : 0;
    int locality = (hint == PrefetchHint::NonTemporal) ? 0 : 3;
    __builtin_prefetch(addr, rw, locality);
#endif
}

void prefetch_range(const void* start, size_t size, PrefetchHint hint, 
                   CacheLevel level, size_t stride) {
    if (!start || size == 0) return;
    
    // Ensure stride is at least cache line size
    if (stride == 0) stride = 64;
    
    const char* ptr = static_cast<const char*>(start);
    const char* end = ptr + size;
    
    while (ptr < end) {
        prefetch(ptr, hint, level);
        ptr += stride;
    }
}

void prefetch_ahead(const void* current_addr, size_t distance, PrefetchHint hint) {
    if (!current_addr) return;
    
    const char* prefetch_addr = static_cast<const char*>(current_addr) + distance;
    prefetch(prefetch_addr, hint);
}

void prefetch_scatter(const std::vector<const void*>& addresses, PrefetchHint hint) {
    for (const void* addr : addresses) {
        if (addr) {
            prefetch(addr, hint);
        }
    }
}

// ============================================================================
// Adaptive Prefetcher Implementation
// ============================================================================

AdaptivePrefetcher::AdaptivePrefetcher() : AdaptivePrefetcher(Config{}) {}

AdaptivePrefetcher::AdaptivePrefetcher(const Config& config) 
    : config_(config), history_index_(0) {
    access_history_.resize(config_.history_size);
    stats_ = {};
}

void AdaptivePrefetcher::record_access(const void* addr, PrefetchHint hint) {
    if (!addr) return;
    
    auto now = std::chrono::steady_clock::now();
    uint64_t timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    
    // Record the access
    AccessEntry& entry = access_history_[history_index_];
    entry.addr = addr;
    entry.timestamp = timestamp;
    entry.hint = hint;
    
    history_index_ = (history_index_ + 1) % config_.history_size;
    stats_.total_accesses++;
    
    // Update stride patterns
    if (config_.enable_stride_detection) {
        detect_stride_patterns();
    }
}

size_t AdaptivePrefetcher::prefetch_predicted(size_t max_prefetches) {
    auto predictions = predict_next_accesses(max_prefetches);
    
    size_t prefetched = 0;
    for (const void* addr : predictions) {
        if (addr) {
            prefetch(addr, PrefetchHint::Read, config_.target_level);
            prefetched++;
            stats_.prefetches_issued++;
        }
    }
    
    return prefetched;
}

std::vector<const void*> AdaptivePrefetcher::predict_next_accesses(size_t count) {
    std::vector<const void*> predictions;
    predictions.reserve(count);
    
    if (access_history_.empty() || stats_.total_accesses < 2) {
        return predictions;
    }
    
    // Find the most confident stride pattern
    StridePattern* best_pattern = nullptr;
    double best_confidence = 0.0;
    
    for (auto& pattern : stride_patterns_) {
        double confidence = calculate_confidence(pattern);
        if (confidence > best_confidence && confidence >= config_.confidence_threshold) {
            best_confidence = confidence;
            best_pattern = &pattern;
        }
    }
    
    if (best_pattern) {
        // Predict based on stride pattern
        size_t current_idx = (history_index_ - 1 + config_.history_size) % config_.history_size;
        const void* last_addr = access_history_[current_idx].addr;
        
        const char* base_addr = static_cast<const char*>(last_addr);
        
        for (size_t i = 0; i < count; ++i) {
            const char* predicted_addr = base_addr + (i + 1) * best_pattern->stride;
            
            // Bounds check - don't prefetch too far ahead
            if (std::abs((predicted_addr - base_addr)) <= static_cast<intptr_t>(config_.max_prefetch_distance)) {
                predictions.push_back(predicted_addr);
            }
        }
    }
    
    return predictions;
}

void AdaptivePrefetcher::reset() {
    access_history_.clear();
    access_history_.resize(config_.history_size);
    stride_patterns_.clear();
    history_index_ = 0;
    stats_ = {};
}

AdaptivePrefetcher::Stats AdaptivePrefetcher::get_stats() const {
    Stats current_stats = stats_;
    
    if (current_stats.prefetches_issued > 0) {
        current_stats.hit_rate = static_cast<double>(current_stats.prefetch_hits) / 
                                 current_stats.prefetches_issued;
    }
    
    // Calculate overall confidence based on pattern strength
    double total_confidence = 0.0;
    for (const auto& pattern : stride_patterns_) {
        total_confidence += calculate_confidence(pattern);
    }
    
    current_stats.confidence = stride_patterns_.empty() ? 0.0 : 
        total_confidence / stride_patterns_.size();
    
    return current_stats;
}

void AdaptivePrefetcher::detect_stride_patterns() {
    if (stats_.total_accesses < 3) return;
    
    // Look at recent access history to detect stride patterns
    size_t current_idx = (history_index_ - 1 + config_.history_size) % config_.history_size;
    size_t prev_idx = (current_idx - 1 + config_.history_size) % config_.history_size;
    
    const void* current_addr = access_history_[current_idx].addr;
    const void* prev_addr = access_history_[prev_idx].addr;
    
    if (!current_addr || !prev_addr) return;
    
    intptr_t stride = static_cast<const char*>(current_addr) - 
                     static_cast<const char*>(prev_addr);
    
    // Look for existing stride pattern or create new one
    StridePattern* found_pattern = nullptr;
    for (auto& pattern : stride_patterns_) {
        if (pattern.stride == stride) {
            found_pattern = &pattern;
            break;
        }
    }
    
    auto now = std::chrono::steady_clock::now();
    uint64_t timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    
    if (found_pattern) {
        found_pattern->confidence++;
        found_pattern->last_seen = timestamp;
    } else {
        // Create new stride pattern
        StridePattern new_pattern;
        new_pattern.stride = stride;
        new_pattern.confidence = 1;
        new_pattern.last_seen = timestamp;
        stride_patterns_.push_back(new_pattern);
    }
    
    // Remove old patterns
    stride_patterns_.erase(
        std::remove_if(stride_patterns_.begin(), stride_patterns_.end(),
            [timestamp](const StridePattern& p) {
                return (timestamp - p.last_seen) > 1000000000ULL; // 1 second
            }),
        stride_patterns_.end());
}

double AdaptivePrefetcher::calculate_confidence(const StridePattern& pattern) const {
    // Simple confidence calculation based on frequency and recency
    double frequency_confidence = std::min(1.0, pattern.confidence / 10.0);
    
    auto now = std::chrono::steady_clock::now();
    uint64_t timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    
    double age_seconds = (timestamp - pattern.last_seen) / 1e9;
    double recency_confidence = std::exp(-age_seconds / 10.0); // Decay over 10 seconds
    
    return frequency_confidence * recency_confidence;
}

// ============================================================================
// Message Prefetcher Implementation
// ============================================================================

MessagePrefetcher::MessagePrefetcher() : MessagePrefetcher(MessageConfig{}) {}

MessagePrefetcher::MessagePrefetcher(const MessageConfig& config) : config_(config) {}

void MessagePrefetcher::prefetch_message_read(const void* message_ptr, size_t message_size) {
    if (!message_ptr) return;
    
    if (config_.prefetch_headers) {
        // Prefetch message header (typically first 64 bytes)
        prefetch(message_ptr, PrefetchHint::Read, config_.target_level);
    }
    
    if (config_.prefetch_payload && message_size > 64) {
        // Prefetch payload with appropriate strategy
        const char* payload_start = static_cast<const char*>(message_ptr) + 64;
        size_t payload_size = message_size - 64;
        
        prefetch_with_strategy(payload_start, payload_size, PrefetchHint::Read);
    }
}

void MessagePrefetcher::prefetch_message_write(void* message_ptr, size_t message_size) {
    if (!message_ptr) return;
    
    prefetch_with_strategy(message_ptr, message_size, PrefetchHint::Write);
}

void MessagePrefetcher::prefetch_ring_buffer(const void* buffer_start, size_t buffer_size,
                                           size_t read_pos, size_t write_pos) {
    if (!buffer_start) return;
    
    const char* base = static_cast<const char*>(buffer_start);
    
    // Prefetch around read position
    size_t prefetch_size = std::min(buffer_size / 4, config_.message_size_estimate * config_.lookahead_messages);
    prefetch_range(base + read_pos, prefetch_size, PrefetchHint::Read, config_.target_level);
    
    // Prefetch around write position  
    prefetch_range(base + write_pos, prefetch_size, PrefetchHint::Write, config_.target_level);
}

void MessagePrefetcher::prefetch_queue_region(const void* queue_base, size_t element_size,
                                            size_t current_index, size_t queue_size) {
    if (!queue_base) return;
    
    const char* base = static_cast<const char*>(queue_base);
    
    // Prefetch ahead based on lookahead setting
    for (size_t i = 1; i <= config_.lookahead_messages; ++i) {
        size_t next_index = (current_index + i) % queue_size;
        const char* next_element = base + (next_index * element_size);
        prefetch(next_element, PrefetchHint::Read, config_.target_level);
    }
}

void MessagePrefetcher::adapt_to_pattern(const std::vector<size_t>& message_sizes) {
    if (message_sizes.empty()) return;
    
    // Calculate average message size
    size_t total_size = 0;
    for (size_t size : message_sizes) {
        total_size += size;
    }
    
    config_.message_size_estimate = total_size / message_sizes.size();
    
    // Adapt strategy based on size variance
    size_t min_size = *std::min_element(message_sizes.begin(), message_sizes.end());
    size_t max_size = *std::max_element(message_sizes.begin(), message_sizes.end());
    
    if (max_size > min_size * 4) {
        // High variance - use random strategy
        config_.strategy = MessagePrefetchStrategy::Random;
    } else if (config_.message_size_estimate > 64 * 1024) {
        // Large messages - use streaming
        config_.strategy = MessagePrefetchStrategy::Streaming;
    } else {
        // Consistent size - use sequential
        config_.strategy = MessagePrefetchStrategy::Sequential;
    }
}

void MessagePrefetcher::prefetch_with_strategy(const void* ptr, size_t size, PrefetchHint hint) {
    switch (config_.strategy) {
        case MessagePrefetchStrategy::Sequential:
            prefetch_range(ptr, size, hint, config_.target_level, 64);
            break;
            
        case MessagePrefetchStrategy::Streaming:
            prefetch_range(ptr, size, PrefetchHint::Stream, config_.target_level, 128);
            break;
            
        case MessagePrefetchStrategy::Random:
            // Prefetch first few cache lines only
            prefetch_range(ptr, std::min(size, size_t(256)), hint, config_.target_level, 64);
            break;
            
        case MessagePrefetchStrategy::Structured:
            // Prefetch header and known hot regions
            prefetch(ptr, hint, config_.target_level);
            if (size > 1024) {
                prefetch(static_cast<const char*>(ptr) + size - 64, hint, config_.target_level);
            }
            break;
            
        case MessagePrefetchStrategy::Sparse:
            // Minimal prefetching for sparse access
            prefetch(ptr, hint, config_.target_level);
            break;
    }
}

size_t MessagePrefetcher::calculate_optimal_stride(size_t message_size) const {
    // Calculate stride based on cache line size and message characteristics
    size_t cache_line_size = 64; // Assume 64-byte cache lines
    
    if (message_size <= cache_line_size) {
        return cache_line_size;
    } else if (message_size <= 1024) {
        return cache_line_size * 2;
    } else {
        return cache_line_size * 4;
    }
}

// ============================================================================
// Cache-Aligned Allocator Implementation
// ============================================================================

template<typename T>
typename CacheAlignedAllocator<T>::pointer 
CacheAlignedAllocator<T>::allocate(size_type n) {
    size_t cache_line_size = 64; // Typical cache line size
    size_t total_size = n * sizeof(T);
    size_t aligned_size = ((total_size + cache_line_size - 1) / cache_line_size) * cache_line_size;
    
    void* ptr = std::aligned_alloc(cache_line_size, aligned_size);
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    return static_cast<pointer>(ptr);
}

template<typename T>
void CacheAlignedAllocator<T>::deallocate(pointer p, size_type n) {
    std::free(p);
}

template<typename T>
template<typename U, typename... Args>
void CacheAlignedAllocator<T>::construct(U* p, Args&&... args) {
    new(p) U(std::forward<Args>(args)...);
}

template<typename T>
template<typename U>
void CacheAlignedAllocator<T>::destroy(U* p) {
    p->~U();
}

template<typename T>
typename CacheAlignedAllocator<T>::size_type 
CacheAlignedAllocator<T>::max_size() const noexcept {
    return std::numeric_limits<size_type>::max() / sizeof(T);
}

// ============================================================================
// Utility Functions
// ============================================================================

size_t get_cache_line_size() {
#ifdef __x86_64__
    // Use CPUID to get cache line size
    uint32_t eax = 1, ebx, ecx, edx;
    __asm__("cpuid" : "+a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx));
    return ((ebx >> 8) & 0xFF) * 8; // Extract cache line size
#else
    // Default to common cache line size
    return 64;
#endif
}

CacheHierarchy get_cache_hierarchy() {
    CacheHierarchy hierarchy = {};
    
    // Default values - real implementation would query hardware
    hierarchy.l1_data = {32 * 1024, 64, 8, 4.0};
    hierarchy.l1_instruction = {32 * 1024, 64, 8, 4.0};
    hierarchy.l2 = {256 * 1024, 64, 8, 12.0};
    hierarchy.l3 = {8 * 1024 * 1024, 64, 16, 40.0};
    hierarchy.memory_latency_cycles = 300.0;
    
    return hierarchy;
}

size_t calculate_optimal_prefetch_distance(double memory_bandwidth_gbps, 
                                          double access_frequency_hz) {
    // Calculate distance based on memory latency and access pattern
    double bytes_per_access = memory_bandwidth_gbps * 1e9 / access_frequency_hz;
    
    // Clamp to reasonable range
    return std::clamp(static_cast<size_t>(bytes_per_access), size_t(64), size_t(4096));
}

bool is_hardware_prefetcher_enabled() {
    // Platform-specific implementation would check MSRs or system settings
    return true; // Default assumption
}

bool set_hardware_prefetcher_enabled(bool enabled) {
    // Would require privileged access to control registers
    (void)enabled;
    return false; // Not implemented for security reasons
}

double measure_memory_bandwidth(size_t buffer_size, size_t iterations) {
    std::vector<uint8_t> buffer(buffer_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        // Sequential read test
        volatile uint8_t sum = 0;
        for (size_t j = 0; j < buffer_size; j += 64) {
            sum += buffer[j];
        }
        (void)sum; // Prevent optimization
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    double total_bytes = static_cast<double>(buffer_size * iterations);
    double seconds = duration.count() / 1e9;
    
    return (total_bytes / seconds) / 1e9; // GB/s
}

// Pattern analysis functions (simplified implementations)
AccessPattern analyze_access_pattern(const std::vector<const void*>& addresses) {
    AccessPattern pattern;
    pattern.type = AccessPattern::Type::Unknown;
    
    if (addresses.size() < 2) {
        return pattern;
    }
    
    // Calculate strides
    std::vector<intptr_t> strides;
    for (size_t i = 1; i < addresses.size(); ++i) {
        intptr_t stride = static_cast<const char*>(addresses[i]) - 
                         static_cast<const char*>(addresses[i-1]);
        strides.push_back(stride);
    }
    
    // Analyze stride pattern
    if (strides.empty()) {
        return pattern;
    }
    
    // Calculate average stride
    intptr_t total_stride = 0;
    for (intptr_t stride : strides) {
        total_stride += stride;
    }
    pattern.stride_average = static_cast<double>(total_stride) / strides.size();
    
    // Calculate variance
    double variance_sum = 0.0;
    for (intptr_t stride : strides) {
        double diff = stride - pattern.stride_average;
        variance_sum += diff * diff;
    }
    pattern.stride_variance = variance_sum / strides.size();
    
    // Classify pattern type
    if (pattern.stride_variance < 64.0) {
        if (std::abs(pattern.stride_average) < 128.0) {
            pattern.type = AccessPattern::Type::Sequential;
        } else {
            pattern.type = AccessPattern::Type::Strided;
        }
    } else {
        pattern.type = AccessPattern::Type::Random;
    }
    
    return pattern;
}

PrefetchRecommendation get_prefetch_recommendation(const AccessPattern& pattern) {
    PrefetchRecommendation rec;
    
    switch (pattern.type) {
        case AccessPattern::Type::Sequential:
            rec.distance = PrefetchDistance::Medium;
            rec.target_level = CacheLevel::L1;
            rec.stride = 64;
            rec.enable_hardware_prefetcher = true;
            rec.reasoning = "Sequential access detected - use medium distance prefetch";
            break;
            
        case AccessPattern::Type::Strided:
            rec.distance = PrefetchDistance::Far;
            rec.target_level = CacheLevel::L2;
            rec.stride = static_cast<size_t>(std::abs(pattern.stride_average));
            rec.enable_hardware_prefetcher = true;
            rec.reasoning = "Strided access detected - use stride-based prefetch";
            break;
            
        case AccessPattern::Type::Random:
            rec.distance = PrefetchDistance::Near;
            rec.target_level = CacheLevel::L3;
            rec.stride = 64;
            rec.enable_hardware_prefetcher = false;
            rec.reasoning = "Random access detected - minimal prefetch to avoid pollution";
            break;
            
        default:
            rec.distance = PrefetchDistance::Auto;
            rec.target_level = CacheLevel::Auto;
            rec.stride = 64;
            rec.enable_hardware_prefetcher = true;
            rec.reasoning = "Unknown pattern - use adaptive prefetch";
            break;
    }
    
    return rec;
}

// Template instantiations for common types
template class CacheAlignedAllocator<float>;
template class CacheAlignedAllocator<double>;
template class CacheAlignedAllocator<int32_t>;
template class CacheAlignedAllocator<uint8_t>;

} // namespace perf
} // namespace psyne