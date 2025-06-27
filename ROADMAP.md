# Psyne Development Roadmap

## Current Status (v0.1.0)

### ✅ Completed Features

#### Core Architecture
- **Single Public Header**: Clean API with just `psyne.hpp`
- **Zero-Copy Messaging**: Messages are views into pre-allocated ring buffers
- **Ring Buffer Implementations**: 
  - SPSC (Single Producer, Single Consumer) - lock-free
  - SPMC (Single Producer, Multiple Consumer) 
  - MPSC (Multiple Producer, Single Consumer)
  - MPMC (Multiple Producer, Multiple Consumer)
- **Message Types**:
  - `FloatVector`: Dynamic-size float arrays
  - `DoubleMatrix`: 2D double matrices
  - Template for custom message types
- **Channel Factory**: URI-based channel creation (`memory://`, `ipc://`)
- **Dynamic Memory Management**:
  - `DynamicSlabAllocator`: Automatically grows/shrinks memory slabs
  - `DynamicRingBuffer`: Auto-resizing ring buffers based on usage

#### Examples
- Simple messaging demonstration
- Producer/consumer patterns
- Multi-type channels
- Fixed-size messages
- Dynamic allocation demo
- Channel factory usage

#### Testing & Benchmarks
- Basic unit tests
- Throughput benchmark
- Latency benchmark  
- Dynamic allocation benchmark

### 🚧 In Progress / Partially Implemented

#### IPC Channels
- Basic shared memory support exists but needs:
  - Cross-platform semaphore implementation
  - Proper cleanup on process termination
  - Named shared memory management

#### TCP Channels
- Skeleton code exists (`tcp_echo_server.cpp`, `tcp_echo_client.cpp`)
- `tcp_protocol.cpp` has hash function but no actual implementation
- Needs:
  - Boost.Asio integration
  - Message framing protocol
  - Connection management
  - Async send/receive

## TODO List

### High Priority

#### 1. Complete IPC Implementation
- [ ] Implement cross-platform IPC using Boost.Interprocess
- [ ] Add proper semaphore/condition variable signaling
- [ ] Handle process crashes and cleanup
- [ ] Add IPC performance benchmarks
- [ ] Create robust IPC examples

#### 2. TCP Channel Implementation  
- [ ] Remove standalone `tcp_protocol.cpp` or integrate properly
- [ ] Implement TCP channel using Boost.Asio
- [ ] Design message framing:
  ```
  [4 bytes: length][4 bytes: checksum][8 bytes: type header][payload]
  ```
- [ ] Add connection management (connect, disconnect, reconnect)
- [ ] Implement async operations with coroutines
- [ ] Add TCP benchmarks and examples

#### 3. GPU Integration
- [ ] Design GPU buffer abstraction interface
- [ ] Metal support for macOS/iOS
- [ ] Vulkan support for cross-platform
- [ ] CUDA support for NVIDIA GPUs
- [ ] Zero-copy GPU buffer mapping
- [ ] GPU compute pipeline examples

### Medium Priority

#### 4. Enhanced Message Types
- [ ] Fixed-size matrix types (e.g., `Matrix4x4f`)
- [ ] Quantized types (`Int8Vector`, `UInt8Vector`)
- [ ] Complex number support
- [ ] Tensor type for ML workloads
- [ ] Sparse matrix support

#### 5. Performance Optimizations
- [ ] SIMD optimizations for message operations
- [ ] Huge page support for large buffers
- [ ] NUMA-aware allocation
- [ ] CPU affinity helpers
- [ ] Memory prefetching hints

#### 6. Reliability Features
- [ ] Message acknowledgment system
- [ ] Retry mechanisms for TCP
- [ ] Heartbeat/keepalive for connections
- [ ] Circuit breaker pattern
- [ ] Message replay buffers

### Low Priority

#### 7. Additional Transports
- [ ] Unix domain sockets
- [ ] RDMA/InfiniBand support
- [ ] UDP multicast channels
- [ ] WebSocket channels
- [ ] gRPC compatibility layer
- [ ] Apache Arrow integration for data interchange

#### 8. Developer Experience
- [ ] Comprehensive API documentation
- [ ] Tutorial series
- [ ] Performance tuning guide
- [ ] Debugging utilities
- [ ] Channel introspection tools
- [ ] Visual buffer usage monitor

#### 9. Language Bindings
- [ ] Python bindings (pybind11)
- [ ] Rust bindings
- [ ] C API for FFI
- [ ] Julia bindings

#### 10. Advanced Features
- [ ] Message routing/filtering
- [ ] Channel multiplexing
- [ ] Compression support
- [ ] Encryption support
- [ ] Distributed tracing integration

## Breaking Changes Planned

### v0.2.0
- Remove legacy code in `src/tcp_protocol.cpp`
- Standardize error handling (exceptions vs error codes)
- Potential API changes for GPU buffer support

### v1.0.0
- Stable API commitment
- ABI compatibility guarantees
- Semantic versioning

## Contributing

Areas where contributions are especially welcome:
1. TCP channel implementation
2. Cross-platform IPC testing
3. GPU integration (especially Vulkan/CUDA)
4. Performance optimizations
5. Documentation and examples

## Timeline Estimates

- **Q1 2024**: Complete IPC and TCP implementations
- **Q2 2024**: GPU integration (Metal first, then Vulkan)
- **Q3 2024**: Performance optimizations and reliability features
- **Q4 2024**: Version 1.0 release with stable API

## Notes

- Focus remains on maintaining zero-copy principles throughout all implementations
- Apple Silicon (unified memory) remains the primary target, with x86_64 Linux as secondary