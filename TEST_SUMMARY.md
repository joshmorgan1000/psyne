# PSYNE Test Results Summary

## Overall Results
- **Passed**: 23 examples
- **Failed**: 27 examples
- **Total**: 50 examples

## Failure Categories

### 1. Buffer Size Issues (FloatVector 64MB)
These examples fail with "Channel buffer full" because FloatVector requires 64MB per message:
- `test_floatvector`
- `simple_messaging` (tries to create 2 FloatVectors in 128MB buffer)
- `enhanced_types_demo`
- `metrics_demo`
- `debug_demo`
- `zero_copy_showcase`
- `dynamic_allocation_demo`
- `producer_consumer`
- `channel_patterns_showcase`
- `debug_multicast`
- `channel_factory_demo`
- `websocket_demo`
- `request_reply_demo`
- `routing_demo_final`

### 2. Network Transport Not Available
These fail because TCP/network channels aren't available in minimal build:
- `tcp_server`/`tcp_client` pair
- `c_api_demo`

### 3. Unsupported Protocols
- `publish_subscribe_demo` - uses UDP protocol which isn't supported

### 4. Missing Dependencies
- `compression_demo` - experimental feature
- `coroutine_example` - coroutine support issues
- `simple_enhanced_types_test` - type issues

### 5. Network Setup Required
These need actual network services or peers:
- `quic_demo`
- `webrtc_demo`
- `webrtc_p2p_demo`
- `webrtc_simple_example`
- `grpc_demo`

### 6. IPC Issues
- `ipc_producer_consumer` - timing/synchronization issues

## Successfully Passing Examples
These examples work correctly:
- `arrow_demo`
- `async_messaging_demo`
- `collective_simple_test`
- `custom_allocator_demo`
- `fixed_size_demo`
- `performance_demo`
- `simd_demo`
- `tensor_optimization_demo`
- `test_bytevector`
- `windows_test`
- `simple_messaging_zero_copy`
- `filtered_fanout_demo`
- `pair_pattern_demo`
- `modern_cpp20_demo`
- `message_types_demo`
- `high_performance_messaging`
- `multi_type_channel`
- `rudp_demo`
- `tcp_demo`
- `unix_socket_demo`
- `udp_multicast_demo`
- `ipc_demo`
- `ipc_test`

## Key Findings

1. **FloatVector Size**: The 64MB default size for FloatVector is causing many failures. Examples need larger buffers or smaller test message types.

2. **Network Support**: Many network-based examples fail because they require full psyne library features not available in minimal build.

3. **IPC Implementation**: The IPC implementation has synchronization issues between producer and consumer processes.

4. **Good Coverage**: Despite failures, 23 examples pass successfully, showing that core functionality works for:
   - ByteVector and smaller message types
   - Async messaging
   - Custom allocators
   - SIMD operations
   - Performance benchmarks
   - Modern C++20 features
   - Multi-type channels
   - Basic IPC when run correctly

## Recommendations

1. **For Testing**: Create smaller test-specific message types (e.g., TestFloatVector with 1MB size)
2. **For Examples**: Increase buffer sizes to accommodate multiple 64MB messages
3. **For CI**: Skip network-dependent tests or mock network functionality
4. **For IPC**: Fix synchronization between producer/consumer processes