# Getting Started with Psyne

This guide will walk you through the basic concepts and usage patterns of Psyne.

## Installation

```bash
# Clone the repository
git clone https://github.com/joshmorgan1000/psyne.git
cd psyne

# Build with CMake
mkdir build && cd build
cmake ..
make
```

## Basic Concepts

### Messages
Messages in Psyne are views into pre-allocated memory. When you create a message, you're not allocating new memory - you're getting a pointer to existing buffer space.

### Channels
Channels are the communication primitives. They contain ring buffers and handle synchronization.

### Zero-Copy Workflow
1. Producer reserves space in the channel
2. Producer writes directly into that space
3. Producer commits the message
4. Consumer receives a view of the same memory

## Hello World Example

```cpp
#include <psyne/psyne.hpp>
#include <iostream>

using namespace psyne;

int main() {
    // Create a single-type channel for float vectors
    SPSCChannel channel("ipc://hello", 1024 * 1024, ChannelType::SingleType);
    
    // Producer: Create and send a message
    FloatVector message(channel);
    message = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    channel.send(message);
    
    // Consumer: Receive the message
    auto received = channel.receive_single<FloatVector>();
    if (received) {
        std::cout << "Received " << received->size() << " floats: ";
        for (float val : *received) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
```

## Producer/Consumer Pattern

```cpp
#include <psyne/psyne.hpp>
#include <thread>
#include <iostream>

using namespace psyne;

void producer(SPSCChannel& channel) {
    for (int i = 0; i < 10; ++i) {
        FloatVector data(channel);
        data = {
            static_cast<float>(i),
            static_cast<float>(i * 2),
            static_cast<float>(i * 3)
        };
        channel.send(data);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void consumer(SPSCChannel& channel) {
    int count = 0;
    while (count < 10) {
        auto msg = channel.receive_single<FloatVector>(
            std::chrono::milliseconds(1000)
        );
        if (msg) {
            std::cout << "Message " << count << ": ";
            for (float val : *msg) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
            count++;
        }
    }
}

int main() {
    SPSCChannel channel("ipc://producer-consumer", 1024 * 1024);
    
    std::thread prod(producer, std::ref(channel));
    std::thread cons(consumer, std::ref(channel));
    
    prod.join();
    cons.join();
    
    return 0;
}
```

## Event-Driven Processing

```cpp
#include <psyne/psyne.hpp>
#include <iostream>
#include <atomic>

using namespace psyne;

int main() {
    SPSCChannel channel("ipc://events", 10 * 1024 * 1024);
    std::atomic<bool> done{false};
    
    // Start event listener
    auto listener = channel.listen<FloatVector>([&done](FloatVector&& msg) {
        std::cout << "Received vector with " << msg.size() << " elements" << std::endl;
        
        // Process the data (zero-copy!)
        float sum = 0;
        for (float val : msg) {
            sum += val;
        }
        std::cout << "Sum: " << sum << std::endl;
        
        if (msg.size() > 0 && msg[0] < 0) {
            done = true;
        }
    });
    
    // Send some messages
    for (int i = 0; i < 5; ++i) {
        FloatVector data(channel);
        data.resize(10);
        for (int j = 0; j < 10; ++j) {
            data[j] = i * 10 + j;
        }
        channel.send(data);
    }
    
    // Send termination signal
    FloatVector term(channel);
    term = {-1.0f};
    channel.send(term);
    
    // Wait for processing to complete
    while (!done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    channel.stop();
    listener->join();
    
    return 0;
}
```

## Multi-Type Channels

```cpp
#include <psyne/psyne.hpp>
#include <iostream>

using namespace psyne;

int main() {
    // Create a multi-type channel
    MPMCChannel channel("tcp://localhost:9999", 10 * 1024 * 1024, ChannelType::MultiType);
    
    // Start listener with multiple handlers
    auto listener = channel.listen({
        Channel<MPMCRingBuffer>::make_handler<FloatVector>([](FloatVector&& msg) {
            std::cout << "Received FloatVector with " << msg.size() << " elements" << std::endl;
        }),
        
        Channel<MPMCRingBuffer>::make_handler<DoubleMatrix>([](DoubleMatrix&& msg) {
            std::cout << "Received DoubleMatrix " << msg.rows() << "x" << msg.cols() << std::endl;
        })
    });
    
    // Send different message types
    FloatVector vec(channel);
    vec = {1.0f, 2.0f, 3.0f};
    channel.send(vec);
    
    DoubleMatrix mat(channel);
    mat.set_dimensions(2, 3);
    mat.at(0, 0) = 1.0;
    mat.at(0, 1) = 2.0;
    mat.at(0, 2) = 3.0;
    mat.at(1, 0) = 4.0;
    mat.at(1, 1) = 5.0;
    mat.at(1, 2) = 6.0;
    channel.send(mat);
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    channel.stop();
    listener->join();
    
    return 0;
}
```

## Next Steps

- [API Reference](api-reference.md) - Detailed API documentation
- [Examples](../examples/) - More complex examples
- [Performance Guide](performance.md) - Optimization tips
- [IPC/TCP Channels](channels.md) - Network and IPC configuration