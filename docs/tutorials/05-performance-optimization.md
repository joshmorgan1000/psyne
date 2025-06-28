# Tutorial 5: Performance Optimization

This tutorial covers techniques for maximizing Psyne's performance in production environments.

## Performance Fundamentals

### Understanding the Performance Stack

```
Application Layer     <- Your code patterns
    |
Message Layer        <- Message types and sizes  
    |
Channel Layer        <- Channel modes and buffers
    |
Transport Layer      <- Memory, IPC, Network
    |
Hardware Layer       <- CPU, Memory, Network
```

Each layer offers optimization opportunities.

## Measuring Performance

### Built-in Metrics

```cpp
#include <psyne/psyne.hpp>
#include <iostream>
#include <chrono>
#include <thread>

using namespace psyne;

void monitor_channel(Channel& channel) {
    auto start = std::chrono::steady_clock::now();
    auto last_metrics = channel.get_metrics();
    
    while (!channel.is_stopped()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        auto now = std::chrono::steady_clock::now();
        auto current_metrics = channel.get_metrics();
        
        // Calculate rates
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start).count() / 1000.0;
        
        uint64_t msg_delta = current_metrics.messages_sent - last_metrics.messages_sent;
        uint64_t bytes_delta = current_metrics.bytes_sent - last_metrics.bytes_sent;
        
        std::cout << "Performance Metrics:" << std::endl;
        std::cout << "  Message rate: " << msg_delta << " msg/s" << std::endl;
        std::cout << "  Throughput: " << (bytes_delta / 1024.0 / 1024.0) << " MB/s" << std::endl;
        std::cout << "  Avg msg size: " << (msg_delta > 0 ? bytes_delta / msg_delta : 0) << " bytes" << std::endl;
        std::cout << "  Send blocks: " << current_metrics.send_blocks << std::endl;
        std::cout << "  Receive blocks: " << current_metrics.receive_blocks << std::endl;
        
        last_metrics = current_metrics;
    }
}
```

### Custom Performance Benchmarks

```cpp
template<typename MessageType>
class ChannelBenchmark {
    struct Result {
        double messages_per_second;
        double megabytes_per_second;
        double avg_latency_ns;
        double p99_latency_ns;
        size_t total_messages;
    };
    
public:
    static Result run(Channel& channel, size_t num_messages, size_t message_size) {
        std::vector<uint64_t> latencies;
        latencies.reserve(num_messages);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Producer thread
        std::thread producer([&]() {
            for (size_t i = 0; i < num_messages; ++i) {
                auto msg_start = std::chrono::high_resolution_clock::now();
                
                MessageType msg(channel);
                msg.resize(message_size);
                // Fill with test pattern
                for (size_t j = 0; j < message_size; ++j) {
                    msg[j] = static_cast<typename MessageType::value_type>(j % 256);
                }
                
                channel.send(msg);
                
                auto msg_end = std::chrono::high_resolution_clock::now();
                latencies.push_back(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        msg_end - msg_start).count());
            }
        });
        
        // Consumer
        size_t received = 0;
        while (received < num_messages) {
            auto msg = channel.receive<MessageType>();
            if (msg) {
                received++;
            }
        }
        
        producer.join();
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate statistics
        auto total_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
        
        std::sort(latencies.begin(), latencies.end());
        
        Result result;
        result.total_messages = num_messages;
        result.messages_per_second = num_messages * 1e9 / total_duration;
        result.megabytes_per_second = (num_messages * message_size) / 
                                      (total_duration / 1e9) / (1024 * 1024);
        result.avg_latency_ns = std::accumulate(latencies.begin(), latencies.end(), 0.0) / 
                                latencies.size();
        result.p99_latency_ns = latencies[latencies.size() * 99 / 100];
        
        return result;
    }
};

// Usage
auto channel = create_channel("memory://bench", 16 * 1024 * 1024);
auto result = ChannelBenchmark<FloatVector>::run(*channel, 100000, 1024);

std::cout << "Benchmark Results:" << std::endl;
std::cout << "  Messages/sec: " << result.messages_per_second << std::endl;
std::cout << "  Throughput: " << result.megabytes_per_second << " MB/s" << std::endl;
std::cout << "  Avg latency: " << result.avg_latency_ns << " ns" << std::endl;
std::cout << "  P99 latency: " << result.p99_latency_ns << " ns" << std::endl;
```

## Optimization Techniques

### 1. Channel Mode Selection

```cpp
// Benchmark different channel modes
void compare_channel_modes() {
    const size_t buffer_size = 16 * 1024 * 1024;
    const size_t num_messages = 1000000;
    
    struct ModeTest {
        std::string name;
        ChannelMode mode;
    };
    
    std::vector<ModeTest> modes = {
        {"SPSC", ChannelMode::SPSC},
        {"MPSC", ChannelMode::MPSC},
        {"SPMC", ChannelMode::SPMC},
        {"MPMC", ChannelMode::MPMC}
    };
    
    for (const auto& test : modes) {
        auto channel = create_channel("memory://test", buffer_size, test.mode);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Single producer thread
        std::thread producer([&]() {
            for (size_t i = 0; i < num_messages; ++i) {
                ByteVector msg(*channel);
                msg.resize(64);  // Small message
                channel->send(msg);
            }
        });
        
        // Single consumer
        size_t received = 0;
        while (received < num_messages) {
            auto msg = channel->receive<ByteVector>();
            if (msg) received++;
        }
        
        producer.join();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start).count();
        
        std::cout << test.name << ": " 
                  << (num_messages * 1000.0 / duration_ms) << " msg/s" << std::endl;
    }
}
```

### 2. Message Batching

```cpp
// Individual messages (slow)
void send_individual(Channel& channel, const std::vector<float>& data) {
    for (float value : data) {
        FloatVector msg(channel);
        msg.resize(1);
        msg[0] = value;
        channel.send(msg);  // Overhead for each send
    }
}

// Batched messages (fast)
void send_batched(Channel& channel, const std::vector<float>& data, 
                  size_t batch_size = 1000) {
    for (size_t i = 0; i < data.size(); i += batch_size) {
        FloatVector batch(channel);
        size_t this_batch = std::min(batch_size, data.size() - i);
        batch.resize(this_batch);
        
        std::memcpy(batch.data(), &data[i], this_batch * sizeof(float));
        channel.send(batch);
    }
}

// Adaptive batching based on channel state
class AdaptiveBatcher {
    Channel& channel_;
    std::vector<float> buffer_;
    size_t max_batch_size_;
    std::chrono::milliseconds max_latency_;
    std::chrono::steady_clock::time_point last_send_;
    
public:
    AdaptiveBatcher(Channel& channel, size_t max_batch = 1000,
                    std::chrono::milliseconds max_latency = std::chrono::milliseconds(10))
        : channel_(channel), max_batch_size_(max_batch), 
          max_latency_(max_latency), last_send_(std::chrono::steady_clock::now()) {
        buffer_.reserve(max_batch_size_);
    }
    
    void add(float value) {
        buffer_.push_back(value);
        
        // Send if batch is full or timeout exceeded
        auto now = std::chrono::steady_clock::now();
        if (buffer_.size() >= max_batch_size_ || 
            (now - last_send_) >= max_latency_) {
            flush();
        }
    }
    
    void flush() {
        if (buffer_.empty()) return;
        
        FloatVector msg(channel_);
        msg.resize(buffer_.size());
        std::memcpy(msg.data(), buffer_.data(), buffer_.size() * sizeof(float));
        channel_.send(msg);
        
        buffer_.clear();
        last_send_ = std::chrono::steady_clock::now();
    }
};
```

### 3. CPU Optimization

```cpp
// Pin threads to specific cores
void optimize_thread_placement() {
    // Get CPU topology
    unsigned int num_cpus = std::thread::hardware_concurrency();
    std::cout << "Available CPUs: " << num_cpus << std::endl;
    
    auto channel = create_channel("memory://perf", 16 * 1024 * 1024);
    
    // Producer on CPU 0
    std::thread producer([&channel]() {
        // Pin to CPU 0
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        
        // Warm up CPU cache
        for (int i = 0; i < 1000; ++i) {
            FloatVector warmup(*channel);
            warmup.resize(1024);
        }
        
        // Actual work
        for (int i = 0; i < 1000000; ++i) {
            FloatVector msg(*channel);
            msg.resize(1024);
            // Process...
            channel->send(msg);
        }
    });
    
    // Consumer on CPU 1 (same physical core, different hyperthread)
    std::thread consumer([&channel]() {
        // Pin to CPU 1
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(1, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        
        while (true) {
            auto msg = channel->receive<FloatVector>();
            if (msg) {
                // Process...
            }
        }
    });
}

// NUMA-aware allocation
void numa_optimize() {
    // Check NUMA topology
    // $ numactl --hardware
    
    // Allocate on specific NUMA node
    // $ numactl --membind=0 --cpunodebind=0 ./your_app
}
```

### 4. Memory Optimization

```cpp
// Avoid false sharing
struct alignas(64) CacheLinePadded {
    std::atomic<uint64_t> value{0};
    char padding[64 - sizeof(std::atomic<uint64_t>)];
};

// Prefetching for sequential access
template<typename T>
void process_with_prefetch(const T* data, size_t count) {
    const size_t prefetch_distance = 8;  // Tune based on your workload
    
    for (size_t i = 0; i < count; ++i) {
        // Prefetch future data
        if (i + prefetch_distance < count) {
            __builtin_prefetch(&data[i + prefetch_distance], 0, 3);
        }
        
        // Process current data
        process_item(data[i]);
    }
}

// Memory pool for fixed-size allocations
template<typename T>
class MemoryPool {
    std::vector<std::unique_ptr<T[]>> blocks_;
    std::vector<T*> free_list_;
    size_t block_size_;
    std::mutex mutex_;
    
public:
    explicit MemoryPool(size_t block_size = 1024) 
        : block_size_(block_size) {
        grow();
    }
    
    T* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_list_.empty()) {
            grow();
        }
        T* ptr = free_list_.back();
        free_list_.pop_back();
        return ptr;
    }
    
    void deallocate(T* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        free_list_.push_back(ptr);
    }
    
private:
    void grow() {
        auto block = std::make_unique<T[]>(block_size_);
        T* base = block.get();
        
        for (size_t i = 0; i < block_size_; ++i) {
            free_list_.push_back(&base[i]);
        }
        
        blocks_.push_back(std::move(block));
    }
};
```

### 5. Network Optimization

```cpp
// TCP tuning
class OptimizedTCPChannel {
    ChannelPtr channel_;
    
public:
    OptimizedTCPChannel(const std::string& address) {
        channel_ = create_channel(address, 16 * 1024 * 1024);
        
        // Would need access to underlying socket, but conceptually:
        // - Disable Nagle's algorithm (TCP_NODELAY)
        // - Increase socket buffers
        // - Enable TCP_QUICKACK
        // - Consider SO_ZEROCOPY for large messages
    }
    
    // Coalesce small messages
    class MessageCoalescer {
        OptimizedTCPChannel& channel_;
        ByteVector buffer_;
        size_t current_size_ = 0;
        static constexpr size_t MAX_COALESCE_SIZE = 64 * 1024;
        
    public:
        explicit MessageCoalescer(OptimizedTCPChannel& channel) 
            : channel_(channel), buffer_(*channel.channel_) {
            buffer_.resize(MAX_COALESCE_SIZE);
        }
        
        template<typename T>
        void add_message(const T& msg) {
            size_t msg_size = sizeof(uint32_t) + msg.size();  // Size prefix
            
            if (current_size_ + msg_size > MAX_COALESCE_SIZE) {
                flush();
            }
            
            // Add size prefix
            *reinterpret_cast<uint32_t*>(&buffer_[current_size_]) = msg.size();
            current_size_ += sizeof(uint32_t);
            
            // Add message data
            std::memcpy(&buffer_[current_size_], msg.data(), msg.size());
            current_size_ += msg.size();
        }
        
        void flush() {
            if (current_size_ > 0) {
                buffer_.resize(current_size_);
                channel_.channel_->send(buffer_);
                
                // Reset for next batch
                buffer_ = ByteVector(*channel_.channel_);
                buffer_.resize(MAX_COALESCE_SIZE);
                current_size_ = 0;
            }
        }
    };
};
```

### 6. Compression Trade-offs

```cpp
// Benchmark compression impact
void benchmark_compression() {
    struct CompressionTest {
        std::string name;
        compression::CompressionType type;
        int level;
    };
    
    std::vector<CompressionTest> tests = {
        {"None", compression::CompressionType::None, 0},
        {"LZ4-Fast", compression::CompressionType::LZ4, 1},
        {"LZ4-Default", compression::CompressionType::LZ4, 0},
        {"Zstd-Fast", compression::CompressionType::Zstd, 1},
        {"Zstd-Default", compression::CompressionType::Zstd, 3},
        {"Snappy", compression::CompressionType::Snappy, 0}
    };
    
    // Generate test data (compressible)
    std::vector<float> test_data(1000000);
    for (size_t i = 0; i < test_data.size(); ++i) {
        test_data[i] = std::sin(i * 0.01f);  // Repetitive pattern
    }
    
    for (const auto& test : tests) {
        compression::CompressionConfig config;
        config.type = test.type;
        config.level = test.level;
        
        auto channel = create_channel("memory://compress", 
                                     32 * 1024 * 1024,
                                     ChannelMode::SPSC,
                                     ChannelType::MultiType,
                                     false,
                                     config);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Send compressed
        FloatVector msg(*channel);
        msg.resize(test_data.size());
        std::memcpy(msg.data(), test_data.data(), 
                    test_data.size() * sizeof(float));
        channel->send(msg);
        
        // Receive and decompress
        auto received = channel->receive<FloatVector>();
        
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();
        
        std::cout << test.name << ": " << duration_us << " μs" << std::endl;
    }
}
```

## Performance Patterns

### Pattern 1: Zero-Copy Pipeline

```cpp
class ZeroCopyPipeline {
    struct Stage {
        std::function<void(FloatVector&)> process;
        ChannelPtr input;
        ChannelPtr output;
        std::thread worker;
    };
    
    std::vector<Stage> stages_;
    
public:
    void add_stage(std::function<void(FloatVector&)> process_func) {
        size_t stage_id = stages_.size();
        
        auto input = (stage_id == 0) ? 
            create_channel("memory://input", 8 * 1024 * 1024) :
            stages_.back().output;
            
        auto output = create_channel("memory://stage" + std::to_string(stage_id),
                                    8 * 1024 * 1024);
        
        Stage stage;
        stage.process = process_func;
        stage.input = input;
        stage.output = output;
        
        stage.worker = std::thread([stage_id, &stage]() {
            while (true) {
                auto msg = stage.input->receive<FloatVector>();
                if (!msg) break;
                
                // In-place processing
                stage.process(*msg);
                
                // Zero-copy forward
                stage.output->send(*msg);
            }
        });
        
        stages_.push_back(std::move(stage));
    }
    
    Channel& input() { return *stages_.front().input; }
    Channel& output() { return *stages_.back().output; }
};
```

### Pattern 2: Lock-Free Aggregation

```cpp
template<typename T>
class LockFreeAggregator {
    struct alignas(64) Slot {
        std::atomic<T> value{};
        std::atomic<bool> valid{false};
    };
    
    std::vector<Slot> slots_;
    std::atomic<size_t> write_index_{0};
    std::atomic<size_t> read_index_{0};
    
public:
    explicit LockFreeAggregator(size_t capacity) 
        : slots_(capacity) {}
    
    bool try_add(T value) {
        size_t current = write_index_.load(std::memory_order_relaxed);
        size_t next = (current + 1) % slots_.size();
        
        // Check if full
        if (next == read_index_.load(std::memory_order_acquire)) {
            return false;
        }
        
        // Write value
        slots_[current].value.store(value, std::memory_order_relaxed);
        slots_[current].valid.store(true, std::memory_order_release);
        
        // Update write index
        write_index_.store(next, std::memory_order_release);
        return true;
    }
    
    std::optional<T> try_aggregate() {
        size_t current = read_index_.load(std::memory_order_relaxed);
        size_t write = write_index_.load(std::memory_order_acquire);
        
        if (current == write) {
            return std::nullopt;  // Empty
        }
        
        T sum{};
        size_t count = 0;
        
        while (current != write) {
            if (slots_[current].valid.load(std::memory_order_acquire)) {
                sum += slots_[current].value.load(std::memory_order_relaxed);
                slots_[current].valid.store(false, std::memory_order_relaxed);
                count++;
            }
            current = (current + 1) % slots_.size();
        }
        
        read_index_.store(current, std::memory_order_release);
        return count > 0 ? std::optional<T>(sum) : std::nullopt;
    }
};
```

## Performance Checklist

### Before Optimization
- [ ] Define performance requirements
- [ ] Establish baseline measurements
- [ ] Profile to identify bottlenecks
- [ ] Consider algorithmic improvements first

### Channel Configuration
- [ ] Use SPSC mode when possible
- [ ] Size buffers appropriately
- [ ] Enable metrics for monitoring
- [ ] Consider channel placement (NUMA)

### Message Design
- [ ] Minimize message size
- [ ] Batch small messages
- [ ] Use fixed-size messages when possible
- [ ] Avoid unnecessary copies

### System Level
- [ ] Configure CPU affinity
- [ ] Tune network stack (if applicable)
- [ ] Disable CPU frequency scaling
- [ ] Use huge pages for large buffers

### Monitoring
- [ ] Track message rates
- [ ] Monitor latency percentiles
- [ ] Watch for blocking/contention
- [ ] Set up alerting for degradation

## Common Performance Pitfalls

### 1. Oversized Buffers
```cpp
// Bad: Wastes memory, poor cache locality
auto channel = create_channel("memory://huge", 1024 * 1024 * 1024);  // 1GB!

// Good: Right-sized for workload
auto channel = create_channel("memory://rightsized", 16 * 1024 * 1024);  // 16MB
```

### 2. Unnecessary Synchronization
```cpp
// Bad: Mutex in hot path
std::mutex mutex;
void process(Channel& channel) {
    std::lock_guard<std::mutex> lock(mutex);  // Contention!
    FloatVector msg(channel);
    // ...
}

// Good: Lock-free or channel-based synchronization
void process(Channel& channel) {
    FloatVector msg(channel);  // Channel handles synchronization
    // ...
}
```

### 3. Poor Message Reuse
```cpp
// Bad: Allocation per iteration
for (int i = 0; i < 1000000; ++i) {
    std::vector<float> data(1000);  // Heap allocation!
    // Fill and send...
}

// Good: Reuse message buffer
FloatVector msg(channel);
msg.resize(1000);
for (int i = 0; i < 1000000; ++i) {
    // Just fill existing buffer
    // Send...
}
```

## Next Steps

- Tutorial 6: Building Distributed Systems
- Tutorial 7: Advanced Message Patterns
- Tutorial 8: Production Deployment

## Summary

Key performance principles:
1. **Measure first** - Don't optimize blindly
2. **Understand your workload** - Different patterns need different optimizations
3. **Minimize overhead** - Every instruction counts in hot paths
4. **Leverage hardware** - Use CPU features, NUMA awareness
5. **Monitor continuously** - Performance can degrade over time

Remember: The fastest code is code that doesn't run. Always consider if you can eliminate work entirely before optimizing how it's done.