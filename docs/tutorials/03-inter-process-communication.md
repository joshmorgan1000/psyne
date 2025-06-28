# Tutorial 3: Inter-Process Communication

This tutorial covers using Psyne for communication between separate processes on the same machine.

## Why Inter-Process Communication?

IPC allows you to:
- Isolate components for fault tolerance
- Scale beyond single-process limitations
- Mix programming languages (via future language bindings)
- Build microservice-style architectures

## IPC Channel Basics

IPC channels use shared memory for zero-copy messaging between processes:

```cpp
// Process 1: Producer
#include <psyne/psyne.hpp>
#include <iostream>

using namespace psyne;

int main() {
    // Create IPC channel
    auto channel = create_channel("ipc://sensor_data", 
                                  4 * 1024 * 1024,  // 4MB buffer
                                  ChannelMode::SPSC);
    
    std::cout << "Producer started, sending sensor data..." << std::endl;
    
    for (int i = 0; i < 1000; ++i) {
        FloatVector data(*channel);
        data.resize(100);
        
        // Simulate sensor readings
        for (size_t j = 0; j < 100; ++j) {
            data[j] = std::sin(i * 0.1f) * std::cos(j * 0.05f);
        }
        
        channel->send(data);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::cout << "Producer finished" << std::endl;
    return 0;
}
```

```cpp
// Process 2: Consumer
#include <psyne/psyne.hpp>
#include <iostream>
#include <numeric>

using namespace psyne;

int main() {
    // Connect to existing IPC channel
    auto channel = create_channel("ipc://sensor_data", 
                                  4 * 1024 * 1024,
                                  ChannelMode::SPSC);
    
    std::cout << "Consumer started, processing sensor data..." << std::endl;
    
    int count = 0;
    while (count < 1000) {
        auto data = channel->receive<FloatVector>();
        if (data) {
            // Process the data
            float sum = std::accumulate(data->begin(), data->end(), 0.0f);
            float avg = sum / data->size();
            
            if (count % 100 == 0) {
                std::cout << "Processed " << count << " messages, "
                          << "latest avg: " << avg << std::endl;
            }
            count++;
        }
    }
    
    std::cout << "Consumer finished" << std::endl;
    return 0;
}
```

## Compilation and Running

Save as `ipc_producer.cpp` and `ipc_consumer.cpp`, then:

```bash
# Compile
g++ -std=c++20 ipc_producer.cpp -lpsyne -lpthread -o ipc_producer
g++ -std=c++20 ipc_consumer.cpp -lpsyne -lpthread -o ipc_consumer

# Run in separate terminals
# Terminal 1:
./ipc_producer

# Terminal 2:
./ipc_consumer
```

## Multiple Producers/Consumers

For multiple producers or consumers, use appropriate channel modes:

### MPSC Example: Log Aggregation

```cpp
// Log aggregator (single consumer)
auto log_channel = create_channel("ipc://logs", 
                                  8 * 1024 * 1024,
                                  ChannelMode::MPSC);

// Multiple producer processes can write
ByteVector log_entry(*log_channel);
std::string message = format_log_message();
log_entry.resize(message.size());
std::memcpy(log_entry.data(), message.data(), message.size());
log_channel->send(log_entry);
```

### SPMC Example: Task Distribution

```cpp
// Task distributor (single producer)
auto task_channel = create_channel("ipc://tasks",
                                   4 * 1024 * 1024,
                                   ChannelMode::SPMC);

// Multiple worker processes can receive
auto task = task_channel->receive<TaskMessage>();
if (task) {
    process_task(*task);
}
```

## Handling Process Lifecycle

### Graceful Startup

The first process to open an IPC channel creates it. Subsequent processes attach to it:

```cpp
class IPCManager {
    ChannelPtr channel_;
    std::atomic<bool> connected_{false};
    
public:
    bool connect(const std::string& name, int max_retries = 10) {
        for (int i = 0; i < max_retries; ++i) {
            try {
                channel_ = create_channel(name, 4 * 1024 * 1024);
                connected_ = true;
                return true;
            } catch (const std::exception& e) {
                std::cout << "Retry " << i << ": " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        return false;
    }
    
    void send_with_retry(const Message<>& msg) {
        while (connected_) {
            try {
                channel_->send(msg);
                return;
            } catch (const std::exception& e) {
                // Handle channel full or disconnection
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
};
```

### Cleanup

IPC channels are automatically cleaned up when the last process detaches. However, you can explicitly clean up:

```cpp
// In destructor or cleanup code
channel_.reset();  // Release channel
```

## Advanced IPC Patterns

### Request-Response Over IPC

Create bidirectional communication using two channels:

```cpp
// Service process
class IPCService {
    ChannelPtr request_channel_;
    ChannelPtr response_channel_;
    
public:
    IPCService() {
        request_channel_ = create_channel("ipc://service_requests", 
                                         1024 * 1024, ChannelMode::MPSC);
        response_channel_ = create_channel("ipc://service_responses", 
                                          1024 * 1024, ChannelMode::SPMC);
    }
    
    void run() {
        while (true) {
            auto request = request_channel_->receive<RequestMessage>();
            if (request) {
                // Process request
                ResponseMessage response(*response_channel_);
                response.request_id = request->request_id;
                response.result = process_request(*request);
                response_channel_->send(response);
            }
        }
    }
};

// Client process
class IPCClient {
    ChannelPtr request_channel_;
    ChannelPtr response_channel_;
    std::atomic<uint64_t> next_request_id_{0};
    
public:
    IPCClient() {
        request_channel_ = create_channel("ipc://service_requests", 
                                         1024 * 1024, ChannelMode::MPSC);
        response_channel_ = create_channel("ipc://service_responses", 
                                          1024 * 1024, ChannelMode::SPMC);
    }
    
    std::optional<ResponseMessage> call(const RequestData& data, 
                                       std::chrono::milliseconds timeout) {
        // Send request
        RequestMessage request(*request_channel_);
        request.request_id = next_request_id_++;
        request.data = data;
        request_channel_->send(request);
        
        // Wait for response with matching ID
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            auto response = response_channel_->receive<ResponseMessage>();
            if (response && response->request_id == request.request_id) {
                return response;
            }
        }
        return std::nullopt;  // Timeout
    }
};
```

### Process Pool Pattern

Distribute work across a pool of worker processes:

```cpp
// Master process
class ProcessPool {
    std::vector<ChannelPtr> work_channels_;
    ChannelPtr result_channel_;
    std::atomic<size_t> next_worker_{0};
    
public:
    ProcessPool(size_t num_workers) {
        // Create work distribution channels
        for (size_t i = 0; i < num_workers; ++i) {
            work_channels_.push_back(
                create_channel("ipc://worker_" + std::to_string(i),
                              1024 * 1024, ChannelMode::SPSC));
        }
        
        // Create result collection channel
        result_channel_ = create_channel("ipc://results",
                                       4 * 1024 * 1024, ChannelMode::MPSC);
    }
    
    void submit_work(const WorkItem& item) {
        // Round-robin distribution
        size_t worker = next_worker_++ % work_channels_.size();
        
        WorkMessage msg(*work_channels_[worker]);
        msg.data = item;
        work_channels_[worker]->send(msg);
    }
    
    std::optional<ResultMessage> get_result() {
        return result_channel_->receive<ResultMessage>();
    }
};

// Worker process (run N instances)
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: worker <worker_id>" << std::endl;
        return 1;
    }
    
    int worker_id = std::stoi(argv[1]);
    
    // Connect to assigned work channel
    auto work_channel = create_channel("ipc://worker_" + std::to_string(worker_id),
                                      1024 * 1024, ChannelMode::SPSC);
    
    // Connect to shared result channel
    auto result_channel = create_channel("ipc://results",
                                        4 * 1024 * 1024, ChannelMode::MPSC);
    
    while (true) {
        auto work = work_channel->receive<WorkMessage>();
        if (work) {
            // Process work
            ResultMessage result(*result_channel);
            result.worker_id = worker_id;
            result.data = process_work(work->data);
            result_channel->send(result);
        }
    }
}
```

## Performance Considerations

### IPC vs In-Memory Channels

| Aspect | In-Memory | IPC |
|--------|-----------|-----|
| Latency | ~100 ns | ~500 ns |
| Throughput | 10M+ msg/s | 5M+ msg/s |
| Isolation | None | Process boundary |
| Fault tolerance | None | Process isolation |

### Optimizing IPC Performance

1. **Use appropriate buffer sizes**:
   ```cpp
   // Small, frequent messages
   auto channel = create_channel("ipc://control", 256 * 1024);
   
   // Large, batched data
   auto channel = create_channel("ipc://data", 16 * 1024 * 1024);
   ```

2. **Minimize context switches**:
   ```cpp
   // Batch multiple messages
   std::vector<DataPoint> batch;
   batch.reserve(100);
   
   // Collect batch
   while (batch.size() < 100 && more_data_available()) {
       batch.push_back(get_next_data());
   }
   
   // Send as one message
   ByteVector msg(*channel);
   msg.resize(batch.size() * sizeof(DataPoint));
   std::memcpy(msg.data(), batch.data(), msg.size());
   channel->send(msg);
   ```

3. **CPU affinity for producer/consumer**:
   ```cpp
   // Keep producer and consumer on same NUMA node
   // but different cores to avoid contention
   ```

## Debugging IPC Issues

### Common Problems

1. **Channel not found**: Ensure producer starts before consumer, or implement retry logic
2. **Permission issues**: Check file permissions in `/dev/shm/` (Linux) or temporary directory
3. **Buffer full**: Increase buffer size or add flow control
4. **Stale channels**: Clean up orphaned shared memory segments

### Debugging Tools

```cpp
// Add debug output
class DebugChannel {
    ChannelPtr channel_;
    std::string name_;
    
public:
    DebugChannel(const std::string& uri, size_t size) 
        : name_(uri) {
        std::cout << "[" << getpid() << "] Creating channel: " << uri << std::endl;
        channel_ = create_channel(uri, size);
    }
    
    template<typename T>
    void send(const T& msg) {
        std::cout << "[" << getpid() << "] Sending to " << name_ 
                  << ", size: " << msg.size() << std::endl;
        channel_->send(msg);
    }
    
    template<typename T>
    std::optional<T> receive() {
        auto msg = channel_->receive<T>();
        if (msg) {
            std::cout << "[" << getpid() << "] Received from " << name_
                      << ", size: " << msg->size() << std::endl;
        }
        return msg;
    }
};
```

## Security Considerations

IPC channels have process-level access control:

1. **File permissions**: Shared memory files inherit process permissions
2. **Channel naming**: Use unique, unguessable names for sensitive channels
3. **Data validation**: Always validate data from IPC channels
4. **Encryption**: For sensitive data, encrypt before sending

```cpp
// Example: Secure channel naming
std::string generate_secure_channel_name(const std::string& base) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 999999);
    
    return "ipc://" + base + "_" + std::to_string(getpid()) + 
           "_" + std::to_string(dis(gen));
}
```

## Next Steps

- Tutorial 4: Network Channels (TCP, Unix sockets, UDP)
- Tutorial 5: Performance Optimization
- Tutorial 6: Building Distributed Systems

## Exercises

1. Create a multi-process pipeline where each stage runs in a separate process
2. Implement a simple pub-sub system using SPMC channels
3. Build a process pool that dynamically scales based on workload
4. Create a fault-tolerant service that can restart without losing messages

## Summary

IPC channels provide:
- Process isolation with near-memory performance
- Fault tolerance through process boundaries
- Scalability through multi-process architectures
- Foundation for distributed systems

Use IPC when you need isolation, fault tolerance, or multi-language support without sacrificing too much performance.