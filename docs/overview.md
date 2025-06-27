# Psyne Overview

Psyne is a zero-copy, event-driven message pipeline designed for high-performance computing on unified memory architectures. It enables the same binary payload to flow through CPU logic and GPU compute kernels without memory copies.

## Core Design Principles

### 1. Zero-Copy Architecture
- Messages are allocated directly in pre-allocated slabs (ring buffers)
- Producers write directly into the buffer memory
- Consumers receive views into the same memory
- No serialization/deserialization overhead

### 2. Event-Driven Processing
- Asynchronous message passing with callbacks
- Thread-safe producer/consumer patterns (SPSC, SPMC, MPSC, MPMC)
- Efficient signaling mechanisms for IPC

### 3. Unified Memory Model
- Designed for Apple Silicon and other unified memory architectures
- Same memory can be accessed as:
  - CPU arrays (`std::span<float>`, `float*`)
  - Linear algebra structures (Eigen3 matrices)
  - GPU buffers (Metal, Vulkan, CUDA)

## Architecture Components

### Memory Management
- **Slab Allocator**: Pre-allocated contiguous memory blocks
- **Ring Buffers**: Lock-free circular buffers with atomic operations
- **Memory Views**: Zero-copy access to typed data
- **Dynamic Allocation**: Automatic growth/shrink based on usage patterns
  - DynamicSlabAllocator: Manages multiple slabs with automatic cleanup
  - DynamicRingBuffer: Resizes buffer capacity based on load

### Message System
- **VariantHdr**: 8-byte header for runtime type information
- **Message Types**: Strongly-typed message classes that are views into buffer memory
- **Wire Format**: Efficient binary layout with proper alignment

### Channel Types
1. **Single-Type Channels**: Optimized for one message type (no metadata overhead)
2. **Multi-Type Channels**: Support multiple message types (8-byte envelope overhead)

### Transport Layers
- **IPC Channels**: Shared memory with semaphore signaling
- **TCP Channels**: Network transport with Boost.Asio

## Use Cases

### High-Frequency Sensor Data
Single-type channels provide zero-overhead transport for sensor readings at millions of messages per second.

### GPU Pipeline Integration
Messages can flow directly from CPU preprocessing to GPU compute kernels without copying.

### Multi-Process Systems
IPC channels enable efficient communication between processes on the same machine.

### Distributed Computing
TCP channels extend the zero-copy philosophy across network boundaries (with necessary copies only at network interfaces).

## Performance Characteristics

- **Latency**: Sub-microsecond for IPC channels
- **Throughput**: Limited only by memory bandwidth
- **CPU Usage**: Minimal overhead with lock-free algorithms
- **Memory Usage**: Pre-allocated, no dynamic allocation in hot paths