# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2025-01-29

### Added
- **ZeroMQ-style Socket Patterns** - Complete messaging patterns implementation
  - Request-Reply (REQ/REP)
  - Publish-Subscribe (PUB/SUB)
  - Push-Pull (PUSH/PULL)
  - Dealer-Router (DEALER/ROUTER)
  - Pair (PAIR)
  - Multi-part message support
  - Pattern-specific optimizations

- **RUDP (Reliable UDP) Transport** - TCP-like reliability over UDP
  - Automatic packet retransmission
  - Flow control and congestion control
  - Configurable reliability/performance trade-offs
  - Connection management and state tracking
  - Comprehensive benchmarking examples

- **QUIC Transport Protocol** - Modern transport for HTTP/3 and beyond
  - Built-in TLS 1.3 encryption
  - Stream multiplexing without head-of-line blocking
  - 0-RTT connection resumption
  - Connection migration support
  - Advanced congestion control (CUBIC, BBR)
  - Comprehensive configuration options

- **Enhanced GPU Support**
  - Apple Metal compute kernels
  - Unified memory optimization
  - GPU buffer abstraction layer
  - Metal shader compilation
  - Vector operations and ML primitives

- **Collective Operations**
  - Ring-based algorithms for optimal bandwidth utilization
  - Broadcast primitive
  - All-reduce operations (sum, max, min, product)
  - Scatter/gather operations
  - Barrier synchronization
  - Support for large-scale distributed computing

- **InfiniBand/RDMA Enhancement**
  - Real Verbs API implementation
  - Queue pair management
  - Memory registration for zero-copy
  - Completion queue handling
  - GPUDirect RDMA support structure
  - Comprehensive RDMA benchmarking

- **Libfabric Integration**
  - Unified high-performance fabric interface
  - Provider auto-selection
  - Support for InfiniBand, RoCE, Omni-Path, Ethernet
  - RMA (Remote Memory Access) operations
  - Atomic operations support
  - Hardware-independent programming model

- **Dynamic Memory Management**
  - Adaptive slab allocator with automatic growth
  - Configurable high/low water marks (75%/25% default)
  - Thread-local allocators for reduced contention
  - Starts at 64MB, grows to 1GB based on usage
  - Automatic shrinking to reclaim unused memory
  - Perfect for varying message size workloads

### Improved
- Enhanced channel factory with new transport support
- Better error handling across all transports
- Comprehensive documentation and examples
- Performance optimizations for all patterns

### Fixed
- WebRTC browser input handling in demo
- Various compilation warnings
- Memory alignment issues in some transports

## [1.2.2] - Upcoming

### Planned
- **NVIDIA GPUDirect Integration** - Direct GPU-to-GPU communication over RDMA
- **AMD ROCm Support** - GPU acceleration for AMD hardware
- **UCX Integration** - Unified Communication X framework
- **Nanomsg/NNG Compatible Patterns** - Additional messaging patterns

## [1.2.0] - 2024-12-15

### Added
- WebRTC channel implementation with P2P support
- Browser-based WebRTC game demo
- Initial GPU support infrastructure
- Enhanced type system for messages
- Compression and encryption support

### Improved
- Performance optimizations across all channels
- Better memory management
- Enhanced debugging and metrics

## [1.1.0] - 2024-11-01

### Added
- Multiple language bindings (Python, JavaScript/TypeScript, Rust, Go, Java, C#, Swift, Julia)
- TCP and Unix socket channels
- WebSocket support
- UDP multicast channels
- Channel multiplexing

### Improved
- Zero-copy performance enhancements
- Better cross-platform support
- Comprehensive test coverage

## [1.0.0] - 2024-09-15

### Initial Release
- Core zero-copy messaging framework
- In-memory and IPC channels
- Basic message types
- Ring buffer implementation
- C++ header-only library
- Basic examples and documentation