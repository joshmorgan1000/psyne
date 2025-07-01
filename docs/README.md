# Psyne Documentation

## What is Psyne?

Psyne is a messaging library for AI and machine learning systems that moves data between components without copying it. When training neural networks or running inference, you need to move massive amounts of tensor data between layers, GPUs, and machines. Traditional approaches copy this data multiple times, wasting memory bandwidth and adding latency. Psyne eliminates these copies entirely.

## Core Philosophy

**Zero-copy messaging.** When you send a message through Psyne, the data never moves. Instead of allocating memory, filling it with data, then copying it to a network buffer or shared memory region, Psyne allocates the message directly in its final destination. The sender writes directly into the receiver's memory space. This approach can be 10-100x faster than traditional messaging systems.

**Pure concepts, no inheritance.** Most messaging frameworks force you to inherit from base classes and override virtual functions. Psyne uses C++20 concepts instead - if your type has the right methods, it works. This means zero runtime overhead and complete flexibility to integrate with existing code.

**Intelligent data transformation.** Between the sender and receiver, Psyne can intelligently transform data based on conditions. On a fast local network, tensor data flows uncompressed. On a slower connection, the same data automatically gets compressed. High CPU usage? Compression turns off. This adaptation happens transparently without changing your code.

## Architecture

Psyne separates concerns into four independent layers that compose together:

**Substrates** handle physical transport - whether that's shared memory within a process, TCP sockets over a network, or direct GPU memory transfers. Each substrate optimizes for its specific hardware.

**Protocols** transform data between substrates. The TDT compression protocol understands floating-point tensor data and compresses it intelligently. Encryption protocols secure data. Checksum protocols ensure integrity.

**Patterns** coordinate producer and consumer access. Single-producer-single-consumer achieves the highest performance with lock-free algorithms. Multiple-producer patterns handle gradient aggregation. Multiple-consumer patterns enable model parallel training.

**Messages** provide typed access to raw memory. Your tensor structures map directly onto substrate memory with zero overhead.

## Performance Characteristics

Psyne targets sub-microsecond latency for small messages and over 100 GB/s throughput for large tensors on modern hardware. The lock-free SPSC pattern can move millions of messages per second between threads. Network channels saturate 10GbE links while using minimal CPU. GPU channels bypass the CPU entirely for device-to-device transfers.

Memory usage stays constant after initialization - no allocations occur during message passing. Cache-line-aware data structures minimize false sharing between cores. NUMA-aware allocation keeps data close to the cores that process it.

## Installation

Build Psyne with any C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+). The build system uses CMake 3.20 or later. Network channels require Boost.Asio. GPU channels need CUDA, Metal, or Vulkan development headers.

Clone the repository, create a build directory, run CMake, and build with your preferred generator. The build produces a static library and headers that integrate into your project.

## Research Attribution

The TDT (Tensor Data Transform) compression protocol implements research from "TDT: Tensor Data Transform for Efficient Compression of Multidimensional Data" (arXiv:2506.18062v1). This groundbreaking work introduced byte-level separation and clustering for floating-point tensors, achieving high compression ratios on neural network data. We thank the original researchers for advancing the field and making their ideas available to the community.

## Learn More

The examples directory contains working code for common patterns. The benchmarks directory measures performance on your specific hardware. Both are designed to be readable and educational.

Psyne focuses on one thing: moving data between AI/ML components with zero copies and maximum performance. If you're building neural network training systems, inference servers, or distributed AI applications, Psyne provides the messaging foundation you need.