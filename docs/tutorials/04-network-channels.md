# Tutorial 4: Network Channels

This tutorial covers using Psyne's network channels for distributed communication across machines.

## Network Channel Types

Psyne supports several network protocols:

1. **TCP**: Reliable, ordered delivery
2. **Unix Domain Sockets**: High-performance local networking
3. **UDP Multicast**: One-to-many broadcasting
4. **RDMA/InfiniBand**: Ultra-low latency (HPC environments)

## TCP Channels

### Basic TCP Server

```cpp
#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>

using namespace psyne;

int main() {
    // Create TCP server channel listening on port 8080
    auto channel = create_channel("tcp://0.0.0.0:8080", 
                                  4 * 1024 * 1024,
                                  ChannelMode::SPSC);
    
    std::cout << "TCP server listening on port 8080..." << std::endl;
    
    while (true) {
        auto msg = channel->receive<FloatVector>();
        if (msg) {
            std::cout << "Received " << msg->size() << " floats" << std::endl;
            
            // Echo back with transformation
            FloatVector response(*channel);
            response.resize(msg->size());
            for (size_t i = 0; i < msg->size(); ++i) {
                response[i] = (*msg)[i] * 2.0f;  // Double the values
            }
            channel->send(response);
        }
    }
    
    return 0;
}
```

### Basic TCP Client

```cpp
#include <psyne/psyne.hpp>
#include <iostream>
#include <chrono>

using namespace psyne;

int main() {
    // Connect to TCP server
    auto channel = create_channel("tcp://localhost:8080",
                                  4 * 1024 * 1024,
                                  ChannelMode::SPSC);
    
    std::cout << "Connected to TCP server" << std::endl;
    
    // Send data
    for (int i = 0; i < 10; ++i) {
        FloatVector data(*channel);
        data.resize(100);
        
        // Fill with test data
        for (size_t j = 0; j < 100; ++j) {
            data[j] = i + j * 0.01f;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        channel->send(data);
        
        // Wait for response
        auto response = channel->receive<FloatVector>();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (response) {
            auto rtt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "Round-trip time: " << rtt.count() << " μs" << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return 0;
}
```

## Unix Domain Sockets

For local machine communication with lower overhead than TCP:

```cpp
// Server
auto server = create_channel("unix:///tmp/psyne.sock",
                            4 * 1024 * 1024,
                            ChannelMode::SPSC);

// Client (same machine)
auto client = create_channel("unix:///tmp/psyne.sock",
                            4 * 1024 * 1024,
                            ChannelMode::SPSC);
```

### Performance Comparison

```cpp
void benchmark_channel(const std::string& uri, const std::string& name) {
    auto channel = create_channel(uri, 4 * 1024 * 1024);
    
    const size_t num_messages = 10000;
    const size_t message_size = 1024;  // 1KB messages
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Producer thread
    std::thread producer([&]() {
        for (size_t i = 0; i < num_messages; ++i) {
            ByteVector msg(*channel);
            msg.resize(message_size);
            channel->send(msg);
        }
    });
    
    // Consumer
    size_t received = 0;
    while (received < num_messages) {
        auto msg = channel->receive<ByteVector>();
        if (msg) received++;
    }
    
    producer.join();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double throughput_mbps = (num_messages * message_size * 8.0) / duration.count();
    double latency_us = duration.count() / (double)num_messages;
    
    std::cout << name << ":" << std::endl;
    std::cout << "  Throughput: " << throughput_mbps << " Mbps" << std::endl;
    std::cout << "  Avg latency: " << latency_us << " μs" << std::endl;
}

// Compare different channel types
benchmark_channel("memory://bench", "In-Memory");
benchmark_channel("unix:///tmp/bench.sock", "Unix Socket");
benchmark_channel("tcp://localhost:9999", "TCP Loopback");
```

## UDP Multicast

For one-to-many communication patterns:

### Publisher

```cpp
#include <psyne/psyne.hpp>
#include <iostream>
#include <chrono>

using namespace psyne;

int main() {
    // Create multicast publisher
    // Args: multicast_group:port/interface_address
    auto channel = create_channel("udp://239.255.0.1:5000/0.0.0.0",
                                  64 * 1024,  // 64KB buffer
                                  ChannelMode::SPSC);
    
    std::cout << "Multicast publisher started" << std::endl;
    
    // Publish market data
    for (int i = 0; i < 1000; ++i) {
        // Create market data message
        struct MarketData {
            uint64_t timestamp;
            uint32_t symbol_id;
            float bid_price;
            float ask_price;
            uint32_t bid_volume;
            uint32_t ask_volume;
        };
        
        ByteVector msg(*channel);
        msg.resize(sizeof(MarketData));
        
        auto* data = msg.as<MarketData>();
        data->timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        data->symbol_id = i % 100;
        data->bid_price = 100.0f + (i % 10) * 0.1f;
        data->ask_price = data->bid_price + 0.01f;
        data->bid_volume = 1000 + (i % 1000);
        data->ask_volume = 1000 + ((i + 500) % 1000);
        
        channel->send(msg);
        
        if (i % 100 == 0) {
            std::cout << "Published " << i << " messages" << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    return 0;
}
```

### Subscriber

```cpp
int main() {
    // Join multicast group
    auto channel = create_channel("udp://239.255.0.1:5000/0.0.0.0",
                                  64 * 1024,
                                  ChannelMode::SPSC);
    
    std::cout << "Multicast subscriber started" << std::endl;
    
    int received = 0;
    while (received < 1000) {
        auto msg = channel->receive<ByteVector>();
        if (msg && msg->size() >= sizeof(MarketData)) {
            auto* data = msg->as<MarketData>();
            
            if (received % 100 == 0) {
                std::cout << "Symbol " << data->symbol_id 
                          << ": " << data->bid_price 
                          << "/" << data->ask_price << std::endl;
            }
            
            received++;
        }
    }
    
    std::cout << "Received " << received << " messages" << std::endl;
    return 0;
}
```

## RDMA/InfiniBand

For HPC environments with sub-microsecond latency:

```cpp
// RDMA server
auto rdma_server = create_channel("rdma://0.0.0.0:18515",
                                  16 * 1024 * 1024,  // 16MB
                                  ChannelMode::SPSC);

// RDMA client (different node)
auto rdma_client = create_channel("rdma://compute01:18515",
                                  16 * 1024 * 1024,
                                  ChannelMode::SPSC);

// Ultra-low latency operations
MLTensorF tensor(*rdma_client);
tensor.reshape({64, 3, 224, 224});  // Batch of images
// Fill tensor...
rdma_client->send(tensor);  // Sub-microsecond latency
```

## Advanced Networking Patterns

### Connection Pooling

```cpp
class ConnectionPool {
    std::vector<ChannelPtr> channels_;
    std::atomic<size_t> next_{0};
    std::mutex mutex_;
    
public:
    ConnectionPool(const std::string& host, int base_port, size_t pool_size) {
        for (size_t i = 0; i < pool_size; ++i) {
            auto uri = "tcp://" + host + ":" + std::to_string(base_port + i);
            channels_.push_back(create_channel(uri, 4 * 1024 * 1024));
        }
    }
    
    ChannelPtr get_connection() {
        // Round-robin selection
        size_t idx = next_++ % channels_.size();
        return channels_[idx];
    }
    
    void send_balanced(const Message<>& msg) {
        auto channel = get_connection();
        channel->send(msg);
    }
};
```

### Fault-Tolerant Client

```cpp
class ResilientClient {
    std::vector<std::string> servers_;
    ChannelPtr active_channel_;
    size_t current_server_ = 0;
    
    bool connect_to_next() {
        for (size_t i = 0; i < servers_.size(); ++i) {
            current_server_ = (current_server_ + 1) % servers_.size();
            try {
                active_channel_ = create_channel(servers_[current_server_],
                                               4 * 1024 * 1024);
                std::cout << "Connected to " << servers_[current_server_] << std::endl;
                return true;
            } catch (const std::exception& e) {
                std::cout << "Failed to connect to " << servers_[current_server_]
                          << ": " << e.what() << std::endl;
            }
        }
        return false;
    }
    
public:
    ResilientClient(std::vector<std::string> servers) 
        : servers_(std::move(servers)) {
        connect_to_next();
    }
    
    template<typename T>
    bool send_with_failover(const T& msg) {
        for (int retry = 0; retry < 3; ++retry) {
            try {
                active_channel_->send(msg);
                return true;
            } catch (const std::exception& e) {
                std::cout << "Send failed: " << e.what() << std::endl;
                if (!connect_to_next()) {
                    return false;
                }
            }
        }
        return false;
    }
};

// Usage
ResilientClient client({
    "tcp://primary.example.com:8080",
    "tcp://secondary.example.com:8080",
    "tcp://tertiary.example.com:8080"
});
```

### Secure Communication

```cpp
// Wrapper for encrypted channels
class SecureChannel {
    ChannelPtr channel_;
    std::array<uint8_t, 32> key_;  // 256-bit key
    
    void encrypt(uint8_t* data, size_t size) {
        // Simple XOR encryption (use proper crypto in production!)
        for (size_t i = 0; i < size; ++i) {
            data[i] ^= key_[i % key_.size()];
        }
    }
    
    void decrypt(uint8_t* data, size_t size) {
        encrypt(data, size);  // XOR is symmetric
    }
    
public:
    SecureChannel(const std::string& uri, const std::array<uint8_t, 32>& key)
        : channel_(create_channel(uri, 4 * 1024 * 1024)), key_(key) {}
    
    void send_encrypted(const ByteVector& plain) {
        ByteVector encrypted(*channel_);
        encrypted.resize(plain.size());
        std::memcpy(encrypted.data(), plain.data(), plain.size());
        encrypt(encrypted.data(), encrypted.size());
        channel_->send(encrypted);
    }
    
    std::optional<ByteVector> receive_encrypted() {
        auto encrypted = channel_->receive<ByteVector>();
        if (encrypted) {
            decrypt(encrypted->data(), encrypted->size());
        }
        return encrypted;
    }
};
```

## Network Performance Tuning

### TCP Socket Options

```cpp
// In your channel implementation or wrapper
void optimize_tcp_socket(int socket_fd) {
    // Disable Nagle's algorithm for low latency
    int flag = 1;
    setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    
    // Increase socket buffers
    int buffer_size = 4 * 1024 * 1024;  // 4MB
    setsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size));
    setsockopt(socket_fd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size));
    
    // Enable SO_KEEPALIVE for long-lived connections
    int keepalive = 1;
    setsockopt(socket_fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive));
}
```

### Bandwidth vs Latency Trade-offs

```cpp
// Low latency configuration
auto low_latency_channel = create_channel("tcp://server:8080",
                                         256 * 1024);  // Small buffer
// Send immediately
channel->send(msg);

// High bandwidth configuration  
auto high_bandwidth_channel = create_channel("tcp://server:8080",
                                            16 * 1024 * 1024);  // Large buffer
// Batch messages
std::vector<FloatVector> batch;
for (int i = 0; i < 100; ++i) {
    FloatVector msg(*channel);
    // Fill msg...
    batch.push_back(std::move(msg));
}
// Send batch
for (auto& msg : batch) {
    channel->send(msg);
}
```

## Monitoring Network Channels

```cpp
class NetworkMonitor {
    struct Stats {
        std::atomic<uint64_t> bytes_sent{0};
        std::atomic<uint64_t> bytes_received{0};
        std::atomic<uint64_t> messages_sent{0};
        std::atomic<uint64_t> messages_received{0};
        std::atomic<uint64_t> errors{0};
        std::chrono::steady_clock::time_point start_time;
    };
    
    Stats stats_;
    
public:
    NetworkMonitor() : stats_{} {
        stats_.start_time = std::chrono::steady_clock::now();
    }
    
    void record_send(size_t bytes) {
        stats_.messages_sent++;
        stats_.bytes_sent += bytes;
    }
    
    void record_receive(size_t bytes) {
        stats_.messages_received++;
        stats_.bytes_received += bytes;
    }
    
    void record_error() {
        stats_.errors++;
    }
    
    void print_stats() {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            now - stats_.start_time).count();
        
        if (duration == 0) duration = 1;  // Avoid division by zero
        
        std::cout << "Network Statistics:" << std::endl;
        std::cout << "  Messages sent: " << stats_.messages_sent << std::endl;
        std::cout << "  Messages received: " << stats_.messages_received << std::endl;
        std::cout << "  Bytes sent: " << stats_.bytes_sent << std::endl;
        std::cout << "  Bytes received: " << stats_.bytes_received << std::endl;
        std::cout << "  Errors: " << stats_.errors << std::endl;
        std::cout << "  Send rate: " << stats_.messages_sent / duration << " msg/s" << std::endl;
        std::cout << "  Receive rate: " << stats_.messages_received / duration << " msg/s" << std::endl;
        std::cout << "  Send bandwidth: " 
                  << (stats_.bytes_sent * 8) / (duration * 1024 * 1024) 
                  << " Mbps" << std::endl;
        std::cout << "  Receive bandwidth: " 
                  << (stats_.bytes_received * 8) / (duration * 1024 * 1024) 
                  << " Mbps" << std::endl;
    }
};
```

## Best Practices

### 1. Choose the Right Protocol

- **TCP**: General purpose, reliable delivery needed
- **Unix Sockets**: Same-machine, lower overhead than TCP
- **UDP Multicast**: One-to-many, can tolerate loss
- **RDMA**: Ultra-low latency in HPC environments

### 2. Handle Network Failures

```cpp
template<typename T>
bool send_with_timeout(Channel& channel, const T& msg, 
                      std::chrono::milliseconds timeout) {
    auto start = std::chrono::steady_clock::now();
    
    while (std::chrono::steady_clock::now() - start < timeout) {
        try {
            channel.send(msg);
            return true;
        } catch (const std::exception& e) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    return false;
}
```

### 3. Implement Flow Control

```cpp
class FlowController {
    std::atomic<size_t> in_flight_{0};
    const size_t max_in_flight_;
    std::condition_variable cv_;
    std::mutex mutex_;
    
public:
    explicit FlowController(size_t max_in_flight) 
        : max_in_flight_(max_in_flight) {}
    
    void wait_for_capacity() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return in_flight_ < max_in_flight_; });
        in_flight_++;
    }
    
    void release() {
        in_flight_--;
        cv_.notify_one();
    }
};
```

## Next Steps

- Tutorial 5: Performance Optimization
- Tutorial 6: Building Distributed Systems
- Tutorial 7: Advanced Message Patterns

## Exercises

1. Build a simple chat application using TCP channels
2. Create a distributed sensor network using UDP multicast
3. Implement a load balancer that distributes requests across multiple servers
4. Build a resilient client that automatically fails over between servers

## Summary

Network channels enable distributed Psyne applications with:
- Multiple protocol options for different use cases
- Zero-copy messaging across network boundaries
- High performance with proper configuration
- Building blocks for distributed systems