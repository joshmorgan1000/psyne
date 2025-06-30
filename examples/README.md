# Psyne Examples

This directory contains examples demonstrating Psyne's zero-copy messaging capabilities.

## Core Examples (v1.3.0)

### Zero-Copy Architecture
- **zero_copy_showcase.cpp** - Demonstrates the fundamental zero-copy principles
- **simple_messaging.cpp** - Basic message sending and receiving
- **producer_consumer.cpp** - Classic producer/consumer pattern

### Messaging Patterns
- **request_reply_demo.cpp** - Request/reply pattern (RPC-style)
- **publish_subscribe_demo.cpp** - Pub/sub with UDP multicast
- **channel_patterns_showcase.cpp** - Various channel configurations

### Transport Examples
- **tcp_demo.cpp** - TCP channels with zero-copy
- **udp_multicast_demo.cpp** - UDP multicast broadcasting
- **ipc_demo.cpp** - Inter-process communication via shared memory
- **unix_socket_demo.cpp** - Unix domain sockets

### GPU/Tensor Support
- **gpu_vector_demo.cpp** - GPU memory coordination
- **cuda_vector_demo.cpp** - CUDA integration (requires CUDA)
- **metal_simple_demo.cpp** - Metal integration (macOS)
- **tensor_optimization_demo.cpp** - Optimized tensor transport

### Performance & Testing
- **performance_demo.cpp** - Performance benchmarks
- **high_performance_messaging.cpp** - Optimization techniques
- **test_floatvector.cpp** - Float vector operations
- **test_bytevector.cpp** - Byte vector operations

## Experimental (v2.0+)

These examples demonstrate features planned for future releases:

- **quic_demo.cpp** - QUIC transport (requires special dependencies)
- **compression_demo.cpp** - Message compression (not zero-copy since it's only used for network serialization)
- **webrtc_demo.cpp** - WebRTC data channels
- **grpc_demo.cpp** - gRPC integration

## Building Examples

```bash
mkdir build
cd build
cmake ..
make examples
```

## Key Concepts

1. **Messages are Views**: Messages don't own memory, they're views into ring buffers
2. **Zero Allocations**: No dynamic memory allocation during message transport
3. **Direct Write**: Data is written directly to its final destination
4. **Notification-Only Send**: `send()` just notifies, no data copying
5. **SPSC Optimization**: Single producer/consumer uses zero atomics

## Example Pattern

```cpp
// Create channel with pre-allocated ring buffer
auto channel = Channel::create("memory://demo", 
                              64 * 1024 * 1024,  // 64MB
                              ChannelMode::SPSC,
                              ChannelType::SingleType);

// Write directly to ring buffer
MyMessage msg(*channel);
msg.set_data(...);  // Writes to ring buffer
msg.send();         // Just a notification

// Read with zero-copy view
auto span = channel->buffer_span();
// Process data...
channel->advance_read_pointer(size);
```