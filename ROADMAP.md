# Psyne Development Roadmap 🚀

## Current Status (v1.0.1)

### ✅ Actually Completed Features
- **Single Public Header**: Clean API with `psyne.hpp`
- **Basic Message Framework**: Template-based message system with CRTP
- **Core Message Types**: FloatVector, ByteVector, DoubleMatrix (basic implementations)
- **Channel Factory**: URI-based channel creation framework
- **Memory Transport**: Basic stub implementation for testing
- **CI/CD Pipeline**: GitHub Actions for Linux/macOS builds
- **Release Automation**: Automatic releases on version tags

### 🚧 Stub Implementations (Need Real Implementation)
- Ring buffers (currently just allocate new memory each time)
- Message queues (unbounded growth, no cleanup)
- Channel implementations (minimal functionality)
- Test suite (disabled due to crashes)

## Phase 1: Fix Critical Issues (Q1 2025) 🔥

### High Priority - From QA Report
- [ ] **Fix memory leaks in SPSCRingBuffer**
  - Currently allocates new memory on each reserve()
  - Implement proper circular buffer with reusable memory
- [ ] **Fix unbounded SimpleMessageQueue growth**
  - Add maximum queue size limits
  - Implement backpressure/flow control
- [ ] **Fix global message queue cleanup**
  - Add destructor to remove queues from global map
  - Prevent memory leaks from destroyed channels
- [ ] **Remove AddressSanitizer from release builds**
  - Move to debug-only or CMake option
  - Currently impacts performance

### Test Suite Revival
- [ ] Fix segfaults in basic_test
- [ ] Fix message passing in simple_test
- [ ] Implement proper test fixtures
- [ ] Add memory leak detection tests
- [ ] Enable tests in CI pipeline

## Phase 2: Core Implementation (Q2 2025)

### Real Ring Buffer Implementation
- [ ] Lock-free SPSC ring buffer
- [ ] Proper memory reuse
- [ ] Configurable buffer sizes
- [ ] Buffer overflow handling

### Transport Implementations
- [ ] **Memory Channel**: Proper shared memory with ring buffers
- [ ] **IPC Channel**: Platform-specific implementations
  - Linux: POSIX shared memory
  - macOS: Mach ports
  - Windows: Named pipes
- [ ] **TCP Channel**: Socket-based networking
  - Message framing
  - Connection management
  - Error recovery

### Performance Benchmarks
- [ ] Latency measurements
- [ ] Throughput tests
- [ ] Memory usage profiling
- [ ] Comparison with other IPC libraries

## Phase 3: GPU Acceleration (Q3 2025) 🎮

### NVIDIA GPUDirect
- [ ] CUDA IPC integration
- [ ] GPUDirect RDMA support
- [ ] Unified memory optimizations
- [ ] Multi-GPU support

### AMD ROCm
- [ ] DirectGMA integration
- [ ] HIP compatibility layer
- [ ] Cross-vendor abstractions

### Apple Metal
- [ ] Unified memory zero-copy
- [ ] Metal Performance Shaders integration
- [ ] Shared event synchronization

## Phase 4: Advanced Networking (Q4 2025)

### High-Performance Fabrics
- [ ] **InfiniBand/RoCE**
  - Verbs API integration
  - RDMA operations
  - Queue pair management
- [ ] **Intel/AWS OFI (libfabric)**
  - Provider abstraction
  - Cloud-optimized transports
- [ ] **UCX Integration**
  - Unified communication framework
  - Automatic transport selection

### Collective Operations
- [ ] Broadcast primitives
- [ ] All-reduce algorithms
- [ ] Scatter/gather operations
- [ ] Ring algorithms

## Phase 5: AI/ML Integration (2026)

### Framework Backends
- [ ] **PyTorch Distributed**
  - Custom ProcessGroup
  - Tensor serialization
  - Gradient compression
- [ ] **JAX Collective Ops**
  - XLA custom calls
  - SPMD primitives
- [ ] **TensorFlow DTensor**
  - Mesh communication
  - Layout optimizations

### Optimizations
- [ ] Gradient quantization
- [ ] Sparse tensor support
- [ ] Mixed precision communication
- [ ] Tensor fusion

## Future Ideas & Research 🔮

### Cutting-Edge Transports
- [ ] **CXL.mem**: Compute Express Link memory pooling
- [ ] **NVLink/NVSwitch**: NVIDIA proprietary interconnects
- [ ] **Silicon Photonics**: Optical interconnects
- [ ] **Quantum Networks**: Post-quantum secure channels

### Advanced Features
- [ ] **Smart Routing**: ML-based transport selection
- [ ] **Predictive Prefetching**: Anticipate message patterns
- [ ] **Compression**: LZ4, Zstd, custom ML compressors
- [ ] **Encryption**: Hardware-accelerated crypto
- [ ] **Time-series Optimization**: For streaming data

### Research Projects
- [ ] **Distributed shared memory** abstractions
- [ ] **Transactional messaging** with ACID guarantees
- [ ] **Byzantine fault tolerance** for untrusted networks
- [ ] **Homomorphic encryption** for secure computation

## Contributing

Want to help? Priority areas:
1. Fix memory management issues (see QA_REPORT.md)
2. Implement real ring buffers
3. Create comprehensive test suite
4. Add transport implementations
5. Performance benchmarking

## Success Metrics 🎯

### v2.0 Goals
- Zero memory leaks (Valgrind clean)
- < 1μs latency (same NUMA node)
- > 10 GB/s throughput (memory transport)
- 100% test coverage
- Production deployments

### Long-term Vision
- Industry standard for AI/ML communication
- Integration in major ML frameworks
- Support for exotic accelerators
- Microsecond-scale distributed training

---

*"Move fast and fix things"* - The Psyne Way™