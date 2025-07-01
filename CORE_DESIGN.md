# Psyne Design Document & Implementation Tasks

## Project Overview

Psyne is a high-performance, zero-copy messaging library specifically designed for AI/ML workloads in the Psynetics neural network framework. It provides multiple transport mechanisms (in-process, IPC, network) with a unified API, optimized for tensor transport, gradient synchronization, and layer-to-layer communication patterns.

## Current Development Systems

Nothing unsupported by this hardware will be part of this project:
- Apple MacBook Pro M4, 128GB RAM, 16-core, 40-core GPU (Darwin)
- AMD Ryzen 7 3700X 8-Core, 16-Thread with a GeForce Dual RTX 3060, 12GB (Ubuntu)

## Core Concepts

**A std::move counts as a copy.**

```cpp
// WRONG (what most people do):
auto tensor = std::make_unique<Float32TensorMessage>(shape);  // Allocates somewhere
tensor->fill_data();                                           // Works on temp memory
channel->send(std::move(tensor));                              // COPY to ring buffer!

// RIGHT (our record-breaking approach):
auto channel = create_channel("memory://ml_ring");
auto tensor = Float32TensorMessage(channel);  // Allocates DIRECTLY in ring buffer
tensor->fill_data();                          // Writes directly to final location
tensor.send();                                // Just updates pointers, NO COPY
```

**A Channel is Actually a Memory Slab.**

```cpp
template<Message P>
class Channel<T> {
public:
    Channel(uint16_t size_factor = 0) : size_factor_(size_factor) {
        // Allocate memory slab based on size_factor
        slab_ = allocate_memory_slab((size_factor + 1) * 1024 * 1024 * 32);  // Only works in 32MB chunks.
    }
    // ...

private:
    uint16_t size_factor_;  // Factor to determine slab size
    T* slab_;               // Pointer to the memory slab

    std::vector<std::function<void(T*)>> listeners_; // Callbacks for message handling, passed direct pointers to slab

    T* allocate_memory_slab(size_t size) {
        // ....
    }
};

```

**Messages Only Exist on Channel Slabs.**

Which prefers to be unified or host-visible memory if concepts are met.

```cpp
template<typename T>
class Message<T> {
public:
    Message(Channel<T>& channel) : channel_(channel) {}
    void send(T* data) {
        // Directly write to the slab memory
        channel_.write(data);
        // Notify via asio
        channel_.notify_listeners(data);
    }
};
```

Is this okay? Sure. But you will pay the performance tax.
```cpp
Message<std::string> msg(channel);
msg.send("Hello, Psyne!");  // Needs a header for string size.
```

Or instead
```cpp
// 64 dimensional fleat vector - feature embedding.
// Fixed size, dedicated channel, no header needed. Ultimate performance.
using Float32by64VectorMessage = Float32TensorMessage<64>;
// Can be cast directly to Eigen::VectorXf
// Or operated on directly by GPU shaders.
```

### PsynePool Integration

Psyne leverages the existing PsynePool thread pool implementation for asynchronous operations: `src/global/threadpool.hpp`.

Usage in Psyne:
- Async message handling
- Parallel compression/decompression
- Multi-channel monitoring
- Background reliability tasks
- Memory allocation (buffer stealing)


### Channel Variants

**SPSC vs MPSC vs SPMC vs MPMC Trade-offs:**

| Mode | Use Case | Performance | Complexity | ML Pattern |
|------|----------|-------------|------------|------------|
| SPSC | Layer→Layer forward pass | Highest (lock-free) | Lowest | Sequential layers |
| MPSC | Gradient accumulation | High (lock-free) | Medium | Multiple backward paths |
| SPMC | Data parallel broadcast | Medium | Medium | Batch splitting |
| MPMC | Parameter server | Lower (locks) | Highest | Distributed training |


### Transport Protocols

- **In-Process (Memory Slab)**: Fastest, zero-copy, ideal for single-node training.
- **IPC (POSIX Shared Memory)**: Zero-copy across processes, suitable for multi-process applications.
- **Network (TCP/UDP)**: For distributed training, with optional compression.


### Asio Integration

Psyne uses Asio for asynchronous I/O operations, allowing non-blocking message handling and efficient event-driven programming.

Users can also specify std::function callbacks with a dedicated PsynePool for message handling.