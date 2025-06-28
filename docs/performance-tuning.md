# Psyne Performance Tuning Guide

This guide provides recommendations for optimizing Psyne's performance in production environments.

## Table of Contents

1. [Channel Mode Selection](#channel-mode-selection)
2. [Buffer Size Optimization](#buffer-size-optimization)
3. [CPU Affinity and NUMA](#cpu-affinity-and-numa)
4. [Memory Alignment](#memory-alignment)
5. [Compression Trade-offs](#compression-trade-offs)
6. [Network Optimization](#network-optimization)
7. [Profiling and Monitoring](#profiling-and-monitoring)
8. [Best Practices](#best-practices)

## Channel Mode Selection

### Choosing the Right Mode

The channel mode significantly impacts performance:

- **SPSC (Single Producer, Single Consumer)**: Highest performance, lock-free
  - Use when you have exactly one producer and one consumer
  - No synchronization overhead
  - Best latency characteristics

- **MPSC (Multiple Producer, Single Consumer)**: Good for aggregation
  - Use for collecting data from multiple sources
  - Atomic operations only on producer side
  - Ideal for logging or metrics collection

- **SPMC (Single Producer, Multiple Consumer)**: Good for broadcasting
  - Use for distributing data to multiple workers
  - Atomic operations only on consumer side
  - Ideal for task distribution

- **MPMC (Multiple Producer, Multiple Consumer)**: Most flexible but slowest
  - Use only when truly needed
  - Full synchronization overhead
  - Consider using multiple SPSC channels instead

```cpp
// Optimal: Use SPSC when possible
auto channel = create_channel("memory://fast", 1024*1024, ChannelMode::SPSC);

// Alternative to MPMC: Multiple SPSC channels
std::vector<ChannelPtr> channels;
for (int i = 0; i < num_workers; ++i) {
    channels.push_back(create_channel("memory://worker" + std::to_string(i), 
                                      1024*1024, ChannelMode::SPSC));
}
```

## Buffer Size Optimization

### Finding the Right Buffer Size

Buffer size affects both memory usage and performance:

```cpp
// Small buffers (< 64KB): Good for low-latency, small messages
auto low_latency = create_channel("memory://fast", 64 * 1024);

// Medium buffers (64KB - 1MB): Balanced performance
auto balanced = create_channel("memory://balanced", 256 * 1024);

// Large buffers (> 1MB): High-throughput, batch processing
auto high_throughput = create_channel("memory://batch", 16 * 1024 * 1024);
```

### Buffer Size Guidelines

1. **Calculate based on message rate**:
   ```
   Buffer Size = Message Size × Messages per Second × Latency Tolerance
   ```

2. **Consider cache effects**:
   - L1 cache: 32-64 KB (keep hot data here)
   - L2 cache: 256-512 KB (good for medium buffers)
   - L3 cache: 8-32 MB (shared across cores)

3. **Power of 2 sizes**: Always use power-of-2 buffer sizes for efficient masking

## CPU Affinity and NUMA

### Setting CPU Affinity

Pin threads to specific CPU cores for consistent performance:

```cpp
#include <thread>
#include <pthread.h>

void set_thread_affinity(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    pthread_t thread = pthread_self();
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

// Producer on CPU 0, Consumer on CPU 1
std::thread producer([&channel]() {
    set_thread_affinity(0);
    // Producer code
});

std::thread consumer([&channel]() {
    set_thread_affinity(1);
    // Consumer code
});
```

### NUMA Considerations

For NUMA systems, keep producer and consumer on the same NUMA node:

```cpp
// Check NUMA topology
// $ numactl --hardware

// Bind to NUMA node 0
// $ numactl --cpunodebind=0 --membind=0 ./your_app
```

## Memory Alignment

### Alignment Best Practices

1. **Cache line alignment** (64 bytes on x86_64):
   ```cpp
   alignas(64) struct AlignedData {
       std::atomic<uint64_t> counter;
       char padding[56]; // Prevent false sharing
   };
   ```

2. **SIMD alignment** (32 bytes for AVX):
   ```cpp
   alignas(32) float data[8]; // AVX-aligned
   ```

3. **Page alignment** for huge pages:
   ```cpp
   // Enable huge pages for large buffers
   // $ echo 1024 > /proc/sys/vm/nr_hugepages
   ```

## Compression Trade-offs

### When to Use Compression

Compression can help or hurt performance depending on the scenario:

```cpp
compression::CompressionConfig config;

// High-speed network, CPU-bound: No compression
config.type = compression::CompressionType::None;

// Slow network, spare CPU: LZ4
config.type = compression::CompressionType::LZ4;
config.level = 1; // Fast compression

// Very slow network, lots of spare CPU: Zstd
config.type = compression::CompressionType::Zstd;
config.level = 3; // Balanced

// Only compress large messages
config.min_size_threshold = 1024; // Don't compress < 1KB

auto channel = create_channel("tcp://remote:8080", 1024*1024, 
                            ChannelMode::SPSC, ChannelType::MultiType,
                            false, config);
```

### Compression Guidelines

1. **LZ4**: Best for real-time (5-10 GB/s compression)
2. **Zstd**: Best compression ratio (500 MB/s - 2 GB/s)
3. **Snappy**: Good balance (2-4 GB/s)

## Network Optimization

### TCP Tuning

```cpp
// For TCP channels, tune socket buffers
// System-wide (Linux):
// $ echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
// $ echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
// $ echo 'net.ipv4.tcp_rmem = 4096 87380 134217728' >> /etc/sysctl.conf
// $ echo 'net.ipv4.tcp_wmem = 4096 65536 134217728' >> /etc/sysctl.conf
// $ sysctl -p
```

### UDP Multicast Tuning

```cpp
// Increase receive buffer for multicast
// $ echo 'net.core.netdev_max_backlog = 5000' >> /etc/sysctl.conf
// $ echo 'net.ipv4.udp_mem = 10240 87380 16777216' >> /etc/sysctl.conf
```

### RDMA Optimization

For RDMA channels:
- Use huge pages to reduce TLB misses
- Pin memory to avoid page faults
- Use completion queues efficiently

## Profiling and Monitoring

### Built-in Metrics

```cpp
// Enable metrics during channel creation
auto channel = create_channel("memory://monitored", 1024*1024, 
                            ChannelMode::SPSC, ChannelType::MultiType, 
                            true); // Enable metrics

// Monitor performance
std::thread monitor([&channel]() {
    while (!channel->is_stopped()) {
        auto metrics = channel->get_metrics();
        
        double msg_rate = metrics.messages_sent / elapsed_seconds;
        double throughput_mb = metrics.bytes_sent / (1024.0 * 1024.0 * elapsed_seconds);
        
        std::cout << "Message rate: " << msg_rate << " msg/s, "
                  << "Throughput: " << throughput_mb << " MB/s" << std::endl;
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
});
```

### External Profiling Tools

1. **perf** for CPU profiling:
   ```bash
   perf record -g ./your_app
   perf report
   ```

2. **VTune** for detailed analysis
3. **tcpdump/Wireshark** for network channels
4. **htop/iostat** for system monitoring

## Best Practices

### 1. Message Design

- **Keep messages small**: Fit in L1/L2 cache when possible
- **Avoid dynamic allocation**: Use fixed-size messages
- **Batch when appropriate**: Amortize overhead

```cpp
// Good: Fixed-size, cache-friendly
struct SensorData {
    float values[16];
    uint64_t timestamp;
    uint32_t sensor_id;
    uint32_t flags;
}; // 88 bytes, fits in cache line

// Better: Batched for throughput
struct SensorBatch {
    static constexpr size_t BATCH_SIZE = 64;
    SensorData data[BATCH_SIZE];
    uint32_t count;
};
```

### 2. Access Patterns

- **Sequential access**: Better cache utilization
- **Avoid random access**: Causes cache misses
- **Prefetch when possible**: Use `__builtin_prefetch`

```cpp
// Process messages in batches
std::vector<FloatVector> batch;
batch.reserve(BATCH_SIZE);

// Collect batch
while (batch.size() < BATCH_SIZE) {
    auto msg = channel->receive<FloatVector>(std::chrono::milliseconds(1));
    if (msg) {
        batch.push_back(std::move(*msg));
    } else {
        break;
    }
}

// Process batch efficiently
for (auto& msg : batch) {
    process(msg);
}
```

### 3. Error Handling

- **Avoid exceptions in hot path**: Use error codes
- **Pre-allocate resources**: Avoid allocation failures
- **Handle backpressure**: Don't overwhelm consumers

```cpp
// Good: Handle backpressure
while (true) {
    FloatVector msg(*channel);
    if (!msg.is_valid()) {
        // Channel full, backoff
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        continue;
    }
    
    // Fill and send message
    fill_message(msg);
    channel->send(msg);
}
```

### 4. Testing Performance

Always test with realistic workloads:

```cpp
// Benchmark template
template<typename MessageType>
void benchmark_channel(Channel& channel, size_t num_messages) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Producer
    std::thread producer([&]() {
        for (size_t i = 0; i < num_messages; ++i) {
            MessageType msg(channel);
            // Fill message
            channel.send(msg);
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
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double messages_per_sec = num_messages * 1000000.0 / duration.count();
    double latency_us = duration.count() / (double)num_messages;
    
    std::cout << "Throughput: " << messages_per_sec << " msg/s" << std::endl;
    std::cout << "Average latency: " << latency_us << " μs" << std::endl;
}
```

## Performance Targets

Typical performance targets for Psyne channels:

| Channel Type | Throughput | Latency | Use Case |
|-------------|------------|---------|----------|
| Memory SPSC | 10M+ msg/s | < 100 ns | In-process, ultra-low latency |
| IPC SPSC | 5M+ msg/s | < 500 ns | Inter-process, same machine |
| TCP Local | 1M+ msg/s | < 10 μs | Same machine networking |
| TCP Remote | 100K+ msg/s | < 100 μs | LAN communication |
| RDMA | 5M+ msg/s | < 1 μs | High-performance cluster |

## Conclusion

Performance tuning is an iterative process:

1. **Measure first**: Use built-in metrics and profiling tools
2. **Identify bottlenecks**: CPU, memory, network, or synchronization
3. **Apply optimizations**: Start with algorithmic improvements
4. **Verify improvements**: Always measure after changes
5. **Document findings**: Keep track of what works

Remember: premature optimization is the root of all evil. Focus on correctness first, then optimize based on real measurements.