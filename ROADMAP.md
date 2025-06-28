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
- **Channel Factory**: URI-based channel creation (`memory://`, `ipc://`, `tcp://`)
- **Dynamic Memory Management**:
  - `DynamicSlabAllocator`: Automatically grows/shrinks memory slabs
  - `DynamicRingBuffer`: Auto-resizing ring buffers based on usage

#### IPC Channels ✅ 
- Cross-platform IPC using Boost.Interprocess
- Named shared memory management with semaphore signaling
- Process cleanup and proper resource management
- Complete integration with channel factory

#### TCP Channels ✅
- Full TCP implementation using Boost.Asio
- Message framing protocol with length prefix and xxHash32 checksums
- Connection management (connect, disconnect, reconnect)
- Async send/receive operations
- Client and server support via URI scheme

#### GPU Integration ✅
- GPU buffer abstraction interface for Metal/Vulkan/CUDA
- Metal backend for Apple Silicon unified memory
- Zero-copy GPU buffer mapping and synchronization
- GPU-aware message types (GPUFloatVector, GPUMatrix, GPUTensor)
- Compute pipeline integration

#### Enhanced Message Types ✅
- **Fixed-size matrix types**: `Matrix4x4f`, `Matrix3x3f`, `Matrix2x2f`
- **Fixed-size vectors**: `Vector4f`, `Vector3f` with named accessors
- **Quantized types**: `Int8Vector`, `UInt8Vector` for ML inference
- **Complex number support**: `ComplexVector<T>` for signal processing
- **ML tensor type**: `MLTensor<T>` with NCHW/NHWC layouts and activation functions
- **Sparse matrices**: `SparseMatrix<T>` with CSR format for scientific computing
- **Full Eigen integration** for mathematical operations
- **Zero-copy compatible** in-place operations only

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

#### Performance Optimizations ✅
- [x] SIMD optimizations for message operations
- [x] Huge page support for large buffers
- [x] NUMA-aware allocation
- [x] CPU affinity helpers
- [x] Memory prefetching hints

## TODO List

### High Priority

#### 1. Complete Enhanced Message Types ✅
- [x] Fixed-size matrix types (e.g., `Matrix4x4f`)
- [x] Quantized types (`Int8Vector`, `UInt8Vector`)  
- [x] Complex number support (`ComplexVector<T>`)
- [x] Enhanced tensor type for ML workloads (`MLTensor<T>`)
- [x] Sparse matrix support (`SparseMatrix<T>` with CSR format)

### Medium Priority

#### 2. Reliability Features ✅
- [x] Message acknowledgment system
- [x] Retry mechanisms for TCP
- [x] Heartbeat/keepalive for connections  
- [x] Circuit breaker pattern
- [x] Message replay buffers

### Low Priority

#### 7. Additional Transports
- [x] Unix domain sockets ✅
- [x] RDMA/InfiniBand support ✅
- [x] UDP multicast channels ✅
- [ ] WebSocket channels
- [ ] gRPC compatibility layer
- [ ] Apache Arrow integration for data interchange

#### 8. Developer Experience
- [x] Comprehensive API documentation ✅
- [ ] Tutorial series
- [ ] Performance tuning guide
- [x] Debugging utilities ✅
- [x] Channel introspection tools ✅
- [x] Visual buffer usage monitor ✅

#### 9. Advanced Features
- [ ] Message routing/filtering
- [ ] Channel multiplexing
- [x] Compression support ✅
- [ ] Encryption support
- [ ] Distributed tracing integration

#### 10. Language Bindings
- [ ] C API for FFI
- [ ] Rust bindings
- [ ] Go bindings
- [ ] Java bindings
- [ ] C# bindings
- [ ] Swift bindings
- [ ] JavaScript/TypeScript bindings
- [ ] Julia bindings
- [x] Python bindings (pybind11) ✅

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