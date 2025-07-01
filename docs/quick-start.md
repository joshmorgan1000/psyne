# Quick Start Guide

Get up and running with Psyne in just a few minutes!

## Installation

### Prerequisites
- C++20 compatible compiler (GCC 11+, Clang 14+, MSVC 2022+)
- CMake 3.20 or higher
- Boost.Asio (for network substrates)

### Build from Source

```bash
git clone https://github.com/your-org/psyne.git
cd psyne
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Quick Test

```bash
# Run basic functionality test
./tests/test_basic_functionality

# Run SPSC benchmark
./benchmarks/spsc_inprocess_benchmark
```

## Your First Psyne Channel

### 1. Simple In-Process Messaging

```cpp
#include "psyne/psyne.hpp"

using namespace psyne;

// Create a simple message type
struct SimpleMessage {
    int id;
    float value;
    char data[64];
};

int main() {
    // Create an in-process SPSC channel
    Channel<SimpleMessage, InProcessSubstrate, SPSCPattern> channel;
    
    // Send a message
    auto message = channel.create_message();
    message->id = 42;
    message->value = 3.14f;
    strcpy(message->data, "Hello Psyne!");
    
    channel.send(message);
    
    // Receive the message
    auto received = channel.receive();
    if (received) {
        std::cout << "Received: " << received->id << std::endl;
    }
    
    return 0;
}
```

### 2. Network Messaging with Compression

```cpp
#include "psyne/psyne.hpp"
#include "psyne/protocol/tdt_compression.hpp"

using namespace psyne;
using namespace psyne::protocol;

struct TensorMessage {
    uint32_t width, height, channels;
    float data[];  // Variable-sized tensor data
    
    size_t total_size() const {
        return sizeof(TensorMessage) + (width * height * channels * sizeof(float));
    }
};

int main() {
    // Create a compressed network channel
    ProtocolChannel<TensorMessage, TCPSubstrate, TDTCompressionProtocol> channel;
    
    // Create a tensor message
    auto tensor = channel.create_message(256 * 256 * 3 * sizeof(float));
    tensor->width = 256;
    tensor->height = 256;
    tensor->channels = 3;
    
    // Fill with data (omitted for brevity)
    
    // Send with automatic compression
    channel.send_message(tensor);
    
    return 0;
}
```

## Key Concepts in 2 Minutes

### Substrates = Transport
How bytes move from A to B:
- `InProcessSubstrate` - Shared memory
- `TCPSubstrate` - Network
- `IPCSubstrate` - Inter-process

### Protocols = Intelligence
How data gets transformed:
- `TDTCompressionProtocol` - Compress tensors
- `EncryptionProtocol` - Secure data
- Raw (no protocol) - Pass through

### Patterns = Coordination
How producers/consumers coordinate:
- `SPSC` - One sender, one receiver (fastest)
- `MPSC` - Many senders, one receiver
- `SPMC` - One sender, many receivers
- `MPMC` - Many senders, many receivers

### Messages = Types
Your data structures with zero-copy access

## Common Patterns

### High-Performance ML Training

```cpp
// Gradient aggregation channel
Channel<GradientMessage, InProcessSubstrate, MPSCPattern> gradient_channel;

// Compressed network distribution
ProtocolChannel<WeightMessage, TCPSubstrate, TDTCompressionProtocol> weight_channel;

// GPU tensor channel
Channel<TensorMessage, CUDASubstrate, SPSCPattern> gpu_channel;
```

### Real-Time Inference

```cpp
// Ultra-low latency in-process
Channel<ActivationMessage, InProcessSubstrate, SPSCPattern> inference_channel;

// Memory-mapped IPC for multi-process
Channel<ResultMessage, IPCSubstrate, SPMCPattern> result_channel;
```

## Performance Tips

### 1. Choose the Right Pattern
- **SPSC**: Fastest for single producer/consumer
- **MPSC**: Good for many producers, single consumer
- **SPMC**: Good for single producer, many consumers
- **MPMC**: Most flexible but slowest

### 2. Size Your Messages Appropriately
- Small messages (< 1KB): Optimize for latency
- Large messages (> 1MB): Optimize for throughput
- Use protocols (compression) for network transport

### 3. Use Zero-Copy Design
```cpp
// WRONG: Creates temporary, then copies
auto msg = std::make_unique<MyMessage>();
channel.send(std::move(msg));

// RIGHT: Allocates directly in channel memory
auto msg = channel.create_message();
channel.send(msg);
```

### 4. Profile Your Application
```cpp
// Use the built-in metrics
auto metrics = channel.get_metrics();
std::cout << "Throughput: " << metrics.messages_per_second << std::endl;
```

## Next Steps

- Read the [Architecture Overview](architecture.md) for deeper understanding
- Explore [Examples](../examples/) for more complex use cases
- Check [Benchmarks](../benchmarks/) for performance characteristics
- Learn about [Custom Extensions](extensions.md) for specialized hardware

## Troubleshooting

### Build Issues
- Ensure C++20 support: `g++ --version` should show 11+ or `clang++ --version` should show 14+
- Install Boost: `sudo apt install libboost-all-dev` (Ubuntu) or `brew install boost` (macOS)

### Runtime Issues
- Check memory alignment for custom messages
- Verify channel is not full when sending
- Use debug builds for detailed error messages

### Performance Issues
- Profile with `perf` or similar tools
- Check NUMA topology: `numactl --hardware`
- Monitor cache misses: `perf stat -e cache-misses`

Ready to build high-performance messaging systems? Check out the [examples](../examples/) directory for more comprehensive code samples!