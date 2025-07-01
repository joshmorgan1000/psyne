# Architecture Overview

Psyne is built around a revolutionary concept-based architecture that eliminates traditional inheritance patterns in favor of pure C++20 concepts and composition.

## Core Philosophy

### Zero-Copy First
Every design decision prioritizes eliminating memory copies:

```cpp
// Traditional approach (WRONG):
auto tensor = std::make_unique<TensorMessage>(shape);  // Allocates somewhere
tensor->fill_data();                                   // Works on temp memory
channel->send(std::move(tensor));                      // COPY to ring buffer!

// Psyne approach (RIGHT):
auto channel = create_channel<TensorMessage, InProcessSubstrate, SPSCPattern>();
auto tensor = channel.create_message(shape);           // Allocates DIRECTLY in ring buffer
tensor->fill_data();                                   // Writes directly to final location
channel.send(tensor);                                  // Just updates pointers, NO COPY
```

### Concept-Based Design
No inheritance, no virtual functions, pure duck typing:

```cpp
// Any type that satisfies the Substrate concept works:
template<typename S>
concept Substrate = requires(S substrate, void* ptr, size_t size) {
    { substrate.allocate_memory_slab(size) } -> std::convertible_to<void*>;
    { substrate.transport_send(ptr, size) } -> std::same_as<void>;
    // ... more behaviors
};

// No base classes needed!
struct MyCustomSubstrate {
    void* allocate_memory_slab(size_t size) { /* implementation */ }
    void transport_send(void* data, size_t size) { /* implementation */ }
    // Just satisfy the concept
};
```

## Architecture Layers

The Psyne architecture consists of four clean layers:

```
Message (typed data access)
    ↓
Pattern (producer/consumer coordination)
    ↓  
Protocol (semantic data transformation)
    ↓
Substrate (physical transport)
```

### Layer 1: Substrates (Physical Transport)
**Responsibility**: Move bytes from point A to point B

Examples:
- `InProcessSubstrate` - Shared memory within process
- `TCPSubstrate` - Network transport over TCP
- `GPUSubstrate` - Direct GPU memory access
- `IPCSubstrate` - Inter-process shared memory

### Layer 2: Protocols (Data Transformation)
**Responsibility**: Intelligent semantic data transformation

Examples:
- `TDTCompressionProtocol` - Tensor data compression
- `EncryptionProtocol` - Data encryption/decryption
- `SerializationProtocol` - Type serialization
- `ChecksumProtocol` - Data integrity verification

### Layer 3: Patterns (Coordination)
**Responsibility**: Coordinate producer/consumer access

Examples:
- `SPSC` - Single Producer, Single Consumer (lock-free)
- `MPSC` - Multiple Producer, Single Consumer
- `SPMC` - Single Producer, Multiple Consumer
- `MPMC` - Multiple Producer, Multiple Consumer

### Layer 4: Messages (Typed Access)
**Responsibility**: Provide typed access to raw substrate memory

Examples:
- `TensorMessage<float>` - Neural network tensors
- `GradientMessage` - Gradient data for backpropagation
- `ActivationMessage` - Layer activation values

## Performance Characteristics

### Latency Goals
- **< 100 nanoseconds** for small messages (< 1KB)
- **< 1 microsecond** for medium messages (< 64KB)
- **< 10 microseconds** for large messages (< 1MB)

### Throughput Goals
- **> 100 GB/s** for large tensors on same NUMA node
- **> 10 GB/s** for network transport (10GbE)
- **> 1 TB/s** for GPU-direct transfers

### Memory Efficiency
- **Zero allocations** in critical path
- **Huge page support** (2MB pages)
- **NUMA-aware** memory placement
- **Cache-line aligned** data structures

## Channel Composition

Channels are composed from the four layers:

```cpp
// High-performance in-process tensor channel
Channel<TensorMessage<float>, InProcessSubstrate, SPSCPattern> tensor_channel;

// Compressed network channel for distributed training
Channel<GradientMessage, TCPSubstrate, MPSCPattern, TDTCompressionProtocol> network_channel;

// GPU-direct channel for inference
Channel<ActivationMessage, CUDASubstrate, SPMCPattern> gpu_channel;
```

## Key Design Principles

### 1. Composition over Inheritance
No base classes, just concepts that define behavior contracts.

### 2. Compile-Time Polymorphism
All polymorphism resolved at compile time for zero runtime overhead.

### 3. Hardware-Aware Design
Different substrates optimize for different hardware characteristics.

### 4. Plugin Architecture
Easy to extend with custom substrates, protocols, and patterns.

### 5. Type Safety
Strong typing prevents common messaging errors at compile time.

## Memory Model

### Substrate Memory Ownership
Each substrate owns and manages its memory:

```cpp
void* slab = substrate.allocate_memory_slab(size);  // Substrate allocates
// Use memory for zero-copy messaging
substrate.deallocate_memory_slab(slab);             // Substrate deallocates
```

### Ring Buffer Layout
Memory is organized as circular buffers for efficient producer/consumer access:

```
[Header][Message1][Message2][Message3]...[MessageN][Header]
    ↑                                                    ↑
  tail                                                 head
```

### Cache-Line Awareness
All data structures are aligned to cache line boundaries (64 bytes) to minimize false sharing.

## Thread Safety

### Lock-Free Patterns
SPSC and some SPMC patterns use lock-free algorithms for maximum performance.

### Synchronization Strategies
- **SPSC**: Lock-free ring buffer with memory ordering
- **MPSC**: Producer locks, consumer lock-free
- **SPMC**: Producer lock-free, consumer locks
- **MPMC**: Locks for both producers and consumers

## Error Handling

### Concept Validation
Compile-time validation ensures types satisfy required concepts:

```cpp
static_assert(Substrate<MySubstrate>, "Type must satisfy Substrate concept");
```

### Runtime Error Strategy
- **Graceful degradation** for recoverable errors
- **Fast failure** for programming errors
- **Detailed logging** for debugging

## Platform Support

### Supported Platforms
- **Linux** (primary development platform)
- **macOS** (Apple Silicon and Intel)
- **Windows** (MSVC and Clang)

### Hardware Acceleration
- **SIMD** optimizations (AVX2, AVX-512, NEON)
- **GPU** support (CUDA, Metal, Vulkan)
- **RDMA** for low-latency networking
- **Custom hardware** via plugin architecture