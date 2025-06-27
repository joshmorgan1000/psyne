# IPC and TCP Channels

This guide covers the network and inter-process communication capabilities of Psyne.

## Channel URIs

Psyne uses URI-style addressing for channels:

- **IPC**: `ipc://channel_name` - Shared memory on local machine
- **TCP**: `tcp://host:port` - Network communication

## IPC Channels

IPC (Inter-Process Communication) channels use shared memory for zero-copy communication between processes on the same machine.

### How IPC Works

1. **Shared Memory**: A memory-mapped file is created in `/dev/shm` (Linux) or similar
2. **Ring Buffer**: The shared memory contains the ring buffer structure
3. **Synchronization**: POSIX semaphores provide wake-up signaling
4. **Zero-Copy**: Both processes access the same physical memory

### Creating an IPC Channel

```cpp
// Process 1: Create the channel (server/producer)
SPSCChannel channel("ipc://sensor_data", 10 * 1024 * 1024);

// Process 2: Connect to existing channel (client/consumer)
SPSCChannel channel("ipc://sensor_data", 10 * 1024 * 1024);
```

### IPC Performance Characteristics

- **Latency**: < 1 microsecond typical
- **Throughput**: Limited only by memory bandwidth (10+ GB/s)
- **CPU Usage**: Minimal with semaphore-based signaling

### IPC Example: Multi-Process Pipeline

**Producer Process:**
```cpp
#include <psyne/psyne.hpp>
#include <iostream>

int main() {
    // Create IPC channel
    SPSCChannel channel("ipc://pipeline", 50 * 1024 * 1024);
    
    std::cout << "Producer started on ipc://pipeline\n";
    
    // Generate data
    for (int i = 0; i < 1000; ++i) {
        FloatVector data(channel);
        if (!data.is_valid()) {
            std::cerr << "Buffer full!\n";
            continue;
        }
        
        data.resize(1024);
        for (int j = 0; j < 1024; ++j) {
            data[j] = i * 1024 + j;
        }
        
        channel.send(data);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    return 0;
}
```

**Consumer Process:**
```cpp
#include <psyne/psyne.hpp>
#include <iostream>

int main() {
    // Connect to IPC channel
    SPSCChannel channel("ipc://pipeline", 50 * 1024 * 1024);
    
    std::cout << "Consumer connected to ipc://pipeline\n";
    
    size_t total_processed = 0;
    
    auto listener = channel.listen<FloatVector>([&](FloatVector&& data) {
        // Process data (zero-copy!)
        float sum = 0;
        for (float val : data) {
            sum += val;
        }
        
        total_processed += data.size();
        
        if (total_processed % (100 * 1024) == 0) {
            std::cout << "Processed " << total_processed / 1024 
                      << "K floats, last sum: " << sum << "\n";
        }
    });
    
    // Run until interrupted
    std::cout << "Press Ctrl+C to stop\n";
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    return 0;
}
```

## TCP Channels

TCP channels extend Psyne's zero-copy philosophy across network boundaries. While network transmission requires copying, the local buffer management remains zero-copy.

### TCP Architecture

1. **Local Buffers**: Zero-copy ring buffers on each end
2. **Network Layer**: Boost.Asio for async networking
3. **Wire Protocol**: Efficient binary format with optional compression
4. **Flow Control**: TCP backpressure prevents buffer overflow

### Creating TCP Channels

```cpp
// Server: Listen on a port
MPMCChannel server("tcp://0.0.0.0:9999", 100 * 1024 * 1024);

// Client: Connect to server
MPMCChannel client("tcp://192.168.1.100:9999", 10 * 1024 * 1024);
```

### TCP Performance Characteristics

- **Latency**: Network-dependent (typically 0.1-1ms LAN, 10-100ms WAN)
- **Throughput**: Limited by network bandwidth
- **CPU Usage**: Efficient async I/O with Boost.Asio

### TCP Example: Distributed System

**Server Node:**
```cpp
#include <psyne/psyne.hpp>
#include <iostream>

class RequestHandler {
public:
    RequestHandler(MPMCChannel& channel) : channel_(channel) {}
    
    void start() {
        listener_ = channel_.listen({
            Channel<MPMCRingBuffer>::make_handler<FloatVector>(
                [this](FloatVector&& request) {
                    // Process request
                    float result = process_data(request);
                    
                    // Send response
                    FloatVector response(channel_);
                    response = {result};
                    channel_.send(response);
                }
            )
        });
    }
    
private:
    float process_data(const FloatVector& data) {
        float sum = 0;
        for (float val : data) {
            sum += val * val;  // Sum of squares
        }
        return std::sqrt(sum / data.size());  // RMS
    }
    
    MPMCChannel& channel_;
    std::unique_ptr<std::thread> listener_;
};

int main() {
    MPMCChannel channel("tcp://0.0.0.0:9999", 100 * 1024 * 1024);
    
    std::cout << "Server listening on tcp://0.0.0.0:9999\n";
    
    RequestHandler handler(channel);
    handler.start();
    
    // Run forever
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    return 0;
}
```

**Client Node:**
```cpp
#include <psyne/psyne.hpp>
#include <iostream>
#include <future>

int main() {
    MPMCChannel channel("tcp://localhost:9999", 10 * 1024 * 1024);
    
    std::cout << "Client connected to tcp://localhost:9999\n";
    
    // Send requests
    for (int i = 0; i < 10; ++i) {
        FloatVector request(channel);
        request.resize(100);
        
        // Fill with test data
        for (int j = 0; j < 100; ++j) {
            request[j] = i + j * 0.01f;
        }
        
        std::cout << "Sending request " << i << "\n";
        channel.send(request);
        
        // Wait for response
        auto response = channel.receive_as<FloatVector>(
            std::chrono::seconds(5)
        );
        
        if (response) {
            std::cout << "Received response: " << (*response)[0] << "\n";
        } else {
            std::cerr << "Timeout waiting for response\n";
        }
    }
    
    return 0;
}
```

## Channel Selection Guidelines

### Use IPC Channels When:
- Processes are on the same machine
- Minimum latency is critical
- High throughput is required
- Zero-copy is essential end-to-end

### Use TCP Channels When:
- Communication across machines
- Network transparency is needed
- Firewall/NAT traversal required
- Standard networking infrastructure

### Use Single-Type Channels When:
- Only one message type is used
- Maximum performance is critical
- 8-byte overhead per message is unacceptable

### Use Multi-Type Channels When:
- Multiple message types are needed
- Control and data planes are mixed
- Protocol flexibility is important

## Advanced Topics

### Custom Network Protocols

While TCP is the default, Psyne's architecture supports custom protocols:

```cpp
// Future: UDP channels for low-latency, lossy communication
// UDPChannel channel("udp://239.1.1.1:5000", 1024 * 1024);

// Future: RDMA for HPC clusters
// RDMAChannel channel("rdma://10.0.0.1:5000", 100 * 1024 * 1024);
```

### Security Considerations

Current implementation focuses on performance. For production use, consider:

- **Authentication**: Add authentication to channel establishment
- **Encryption**: Use TLS for TCP channels
- **Access Control**: Restrict IPC channel permissions
- **Validation**: Validate message contents

### Performance Tuning

#### IPC Optimization:
- Use huge pages for large buffers
- Pin threads to cores
- Disable CPU frequency scaling
- Use NUMA-aware allocation

#### TCP Optimization:
- Tune TCP buffer sizes
- Enable TCP_NODELAY for low latency
- Use jumbo frames on supported networks
- Consider kernel bypass networking

## Troubleshooting

### Common IPC Issues:
- **Permission Denied**: Check `/dev/shm` permissions
- **Already Exists**: Previous process didn't clean up
- **Out of Memory**: Increase shared memory limits

### Common TCP Issues:
- **Connection Refused**: Check firewall/port availability
- **Slow Performance**: Check network MTU and latency
- **Message Loss**: Ensure proper error handling