# Psyne Transport Layer Design

## Overview

This document describes the design of Psyne's transport layer, optimized for neural network layer-to-layer communication during training and inference. The design prioritizes sub-microsecond latency and zero-copy operations.

## Key Performance Requirements

- **Latency**: < 100 nanoseconds for small messages (< 1KB)
- **Throughput**: > 100 GB/s for large tensors on same NUMA node
- **Zero-copy**: No memory copies in the data path
- **GPU-aware**: Direct GPU memory access without host staging

## Architecture Decisions

### 1. Memory Management

**Decision**: Use huge pages (2MB) with pre-allocated memory slabs.

**Rationale**:
- Reduces TLB misses by 512x compared to 4KB pages
- Ensures physical memory contiguity for DMA operations
- Eliminates page fault overhead in critical path

**Implementation**:
```cpp
class MemorySlab {
    void* base;      // Aligned to 2MB boundary
    size_t size;     // Multiple of 2MB
    bool gpu_pinned; // Whether memory is pinned for GPU access
};
```

### 2. Channel Architecture

**Decision**: Lock-free SPSC ring buffer with cache-line separation.

**Rationale**:
- SPSC (Single Producer Single Consumer) is the most common pattern for layer-to-layer communication
- Lock-free design eliminates mutex overhead
- Cache-line padding prevents false sharing

**Layout**:
```
[Producer Cache Line (64B)]
  - write_pos (atomic)
  - cached_read_pos
  - padding

[Consumer Cache Line (64B)]  
  - read_pos (atomic)
  - cached_write_pos
  - padding

[Data Region (N * 32MB)]
  - Aligned to cache line
  - Direct message storage
```

### 3. Message Layout

**Decision**: Messages are allocated directly in channel memory with inline headers.

**Structure**:
```cpp
struct MessageHeader {
    uint32_t size;        // Total size including header
    uint16_t type;        // Message type ID
    uint16_t flags;       // GPU memory, compressed, etc.
    uint64_t timestamp;   // For latency measurement
};

// Example: Tensor message
struct TensorMessage {
    MessageHeader header;
    uint32_t ndims;
    uint64_t shape[MAX_DIMS];
    float data[];  // Directly follows in memory
};
```

### 4. GPU Memory Strategy

**Decision**: Use unified memory with explicit prefetching.

**Rationale**:
- Unified memory allows CPU/GPU access without explicit copies
- Prefetching hides transfer latency
- Works across CUDA, Metal, and Vulkan

**Implementation**:
- Allocate channels in GPU-visible memory when GPU flag is set
- Use `cudaMemPrefetchAsync` for CUDA
- Use shared buffers for Metal
- Use device-local host-visible memory for Vulkan

### 5. Synchronization Primitives

**Decision**: Use C++ memory_order_acquire/release with optional x86 PAUSE.

**Key Operations**:
```cpp
// Producer
size_t claim_space(size_t size) {
    size_t write_pos = write_pos_.load(memory_order_relaxed);
    // Check space using cached_read_pos first
    // Only load read_pos if needed (cache miss)
    write_pos_.store(new_pos, memory_order_release);
}

// Consumer  
bool poll_message() {
    size_t read_pos = read_pos_.load(memory_order_relaxed);
    // Use memory_order_acquire when reading data
    // x86: PAUSE instruction in spin loop
}
```

### 6. Batching Support

**Decision**: Support message batching with single synchronization.

**Rationale**:
- Neural networks often process batches
- Amortizes synchronization overhead
- Improves cache efficiency

### 7. Transport Protocols

#### In-Process (Fastest)
- Direct memory access
- No serialization needed
- < 50ns latency

#### IPC (Cross-Process)
- Memory-mapped shared memory
- Same layout as in-process
- < 200ns latency

#### Network (Cross-Node)
- RDMA support for InfiniBand
- TCP with kernel bypass for Ethernet
- GPU Direct for remote GPU access

## Benchmarking Strategy

1. **Latency**: Measure round-trip time for various message sizes
2. **Throughput**: Measure sustained bandwidth for large transfers
3. **GPU Performance**: Measure CPU→GPU→CPU round trip
4. **Scalability**: Test with multiple channels/threads

## Implementation Priority

1. Core memory slab manager with huge page support
2. SPSC channel implementation
3. Basic message types (scalars, vectors, tensors)
4. GPU memory support (CUDA first)
5. IPC shared memory transport
6. Network transport (TCP, then RDMA)
7. Advanced features (compression, encryption)

## Example Usage

```cpp
// Create channel for layer communication
auto channel = Channel<TensorMessage>::create(
    "layer1_to_layer2",
    ChannelConfig{
        .size_mb = 128,        // 128MB for large tensors
        .mode = SPSC,
        .gpu_enabled = true,
        .huge_pages = true
    }
);

// Producer (Layer 1 output)
auto msg = channel->allocate_message<TensorMessage>(tensor_size);
msg->header.type = MessageType::TENSOR_F32;
msg->ndims = 4;
msg->shape[0] = batch_size;
// ... fill tensor data directly ...
channel->send(msg);

// Consumer (Layer 2 input)
if (auto msg = channel->receive()) {
    // Direct access to tensor data, no copy
    process_tensor(msg->data, msg->shape);
    channel->release(msg);
}
```

## Performance Targets

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Small message (64B) | < 50ns | CPU-only |
| Medium message (64KB) | < 500ns | With prefetch |
| Large tensor (10MB) | < 10μs | Memory bandwidth limited |
| GPU round-trip | < 1μs | Pinned memory |
| IPC message | < 200ns | Shared memory |
| Network (RDMA) | < 2μs | InfiniBand |