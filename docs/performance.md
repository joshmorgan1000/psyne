# Psyne Performance Guidelines

## Zero-Copy Philosophy

Psyne is designed from the ground up as a zero-copy messaging system. This means:
- **No memcpy**: Data is written directly into its final destination
- **No intermediate buffers**: Messages are constructed in-place in ring buffers
- **No serialization overhead**: Fixed-size messages with compile-time known layouts
- **Direct memory access**: Consumers read directly from producer's memory

## Performance Best Practices

### 1. Message Creation

**DO:**
```cpp
// Create message directly in channel
FloatVector msg(channel);
msg.resize(100);
// Fill data directly...
msg.send();
```

**DON'T:**
```cpp
// Avoid creating messages outside channels
FloatVector msg;
// ... fill data ...
channel.send_copy(msg); // DEPRECATED - violates zero-copy
```

### 2. Ring Buffer Selection

Choose the right producer-consumer pattern for your use case:
- **SPSC** (Single Producer, Single Consumer): Fastest, no atomics, no contention
- **SPMC** (Single Producer, Multiple Consumer): One writer, multiple readers
- **MPSC** (Multiple Producer, Single Consumer): Multiple writers, one reader  
- **MPMC** (Multiple Producer, Multiple Consumer): Full flexibility, highest overhead

### 3. Memory Alignment

- All allocations are 8-byte aligned by default
- Ring buffers use 64-byte cache line padding to prevent false sharing
- Consider aligning large data structures to cache line boundaries

### 4. Batch Operations

**DO:**
```cpp
// Process multiple messages in one go
while (auto msg = channel.receive_single<MyMessage>()) {
    process(*msg);
}
```

**DON'T:**
```cpp
// Avoid processing one message at a time with delays
if (auto msg = channel.receive_single<MyMessage>()) {
    process(*msg);
    sleep(1); // Bad - increases latency
}
```

### 5. TCP Channel Optimization

The TCP channel uses a dedicated sender thread to avoid double-handling:
- Messages are sent directly from ring buffer memory
- No intermediate copying for network transmission
- Scatter-gather I/O for header + payload

### 6. Eigen Integration

**DO:**
```cpp
// Use Eigen views for zero-copy linear algebra
FloatVector vec(channel);
vec.resize(100);
auto eigen_view = vec.as_eigen();
// Perform operations directly on channel memory
eigen_view = eigen_view.array() * 2.0f;
```

**DON'T:**
```cpp
// Avoid copying to/from Eigen matrices
Eigen::VectorXf eigen_vec(100);
// ... compute ...
vec.from_eigen(eigen_vec); // DEPRECATED - copies data
```

## Performance Measurement

### Key Metrics
- **Latency**: Time from send() to receive
- **Throughput**: Messages per second
- **Memory bandwidth**: GB/s of data transfer
- **CPU cache misses**: Use `perf stat -e cache-misses`

### Profiling Tools
```bash
# CPU profiling
perf record -g ./your_app
perf report

# Cache analysis
perf stat -e cache-references,cache-misses ./your_app

# Memory bandwidth
perf stat -e uncore_imc/data_reads/,uncore_imc/data_writes/ ./your_app
```

## Hardware Optimization

### CPU Affinity
```cpp
// Pin threads to specific cores
std::thread producer([&]() {
    // Set CPU affinity
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    
    // Producer work...
});
```

### NUMA Awareness
- Allocate memory on the same NUMA node as the processing thread
- Use `numactl` to control memory placement

### Huge Pages
Enable transparent huge pages for large buffers:
```bash
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

## Common Pitfalls

1. **Using deprecated copy methods**: All `copy_to()`, `send_copy()`, and `from_eigen()` methods violate zero-copy principles

2. **Creating messages outside channels**: Always create messages with a channel reference

3. **Not batching operations**: Process all available messages before yielding

4. **Ignoring cache effects**: Use the provided cache line padding

5. **Wrong producer-consumer pattern**: Don't use MPMC when SPSC would suffice

## Benchmarking

Always measure performance in release builds:
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

Use the included benchmarks:
```bash
./build/benchmarks/psyne_bench --iterations 1000000
```

## Future Optimizations

Planned performance improvements:
- SIMD operations for bulk data processing
- io_uring support for true kernel-bypass networking
- GPU direct memory access for unified memory architectures
- AVX-512 optimized memory operations