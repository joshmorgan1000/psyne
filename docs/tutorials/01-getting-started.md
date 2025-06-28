# Tutorial 1: Getting Started with Psyne

Welcome to Psyne! This tutorial will help you get started with high-performance, zero-copy messaging.

## What is Psyne?

Psyne is a zero-copy messaging library optimized for AI/ML workloads. It provides:

- **Zero-copy messaging**: Messages are views into pre-allocated buffers
- **Multiple transport types**: In-memory, IPC, TCP, Unix sockets, UDP multicast, RDMA
- **Type safety**: Strongly-typed messages with compile-time checks
- **High performance**: Lock-free SPSC channels, optimized for modern CPUs

## Installation

### Prerequisites

- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.15+
- Boost 1.70+ (for networking)
- Optional: Eigen3 (for matrix operations)

### Building from Source

```bash
git clone https://github.com/joshmorgan1000/psyne.git
cd psyne
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

## Your First Psyne Program

Let's create a simple program that sends messages between threads:

```cpp
#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>

using namespace psyne;

int main() {
    // Create a channel
    auto channel = create_channel("memory://demo", 1024 * 1024);
    
    // Producer thread
    std::thread producer([&channel]() {
        // Create and send 10 messages
        for (int i = 0; i < 10; ++i) {
            FloatVector msg(*channel);
            msg.resize(5);
            
            // Fill the message
            for (size_t j = 0; j < 5; ++j) {
                msg[j] = i * 10.0f + j;
            }
            
            // Send the message
            channel->send(msg);
            std::cout << "Sent message " << i << std::endl;
        }
    });
    
    // Consumer thread
    std::thread consumer([&channel]() {
        int received = 0;
        
        while (received < 10) {
            // Receive a message
            auto msg = channel->receive<FloatVector>();
            
            if (msg) {
                std::cout << "Received: ";
                for (size_t i = 0; i < msg->size(); ++i) {
                    std::cout << (*msg)[i] << " ";
                }
                std::cout << std::endl;
                received++;
            }
        }
    });
    
    // Wait for threads to complete
    producer.join();
    consumer.join();
    
    std::cout << "Done!" << std::endl;
    return 0;
}
```

## Compilation

Save the code as `first_program.cpp` and compile:

```bash
g++ -std=c++20 first_program.cpp -lpsyne -lpthread -o first_program
./first_program
```

## Understanding the Code

Let's break down what's happening:

### 1. Creating a Channel

```cpp
auto channel = create_channel("memory://demo", 1024 * 1024);
```

- `"memory://demo"`: URI specifying an in-memory channel named "demo"
- `1024 * 1024`: Buffer size (1 MB)
- Default mode is SPSC (Single Producer, Single Consumer)

### 2. Creating Messages

```cpp
FloatVector msg(*channel);
```

This creates a message directly in the channel's buffer - no allocation!

### 3. Sending Messages

```cpp
channel->send(msg);
```

Sending marks the message as ready for consumption. After sending, the message object is invalidated.

### 4. Receiving Messages

```cpp
auto msg = channel->receive<FloatVector>();
```

Returns an optional containing the message if available, or empty if no message is ready.

## Key Concepts

### Zero-Copy

Traditional messaging:
```cpp
// Allocates memory
std::vector<float> data(1000);
// Copies data
send_queue.push(data);  // Copy!
// Consumer copies again
auto received = receive_queue.pop();  // Copy!
```

Psyne zero-copy:
```cpp
// No allocation - uses pre-allocated buffer
FloatVector msg(*channel);
// Direct write to buffer
msg[0] = 42.0f;
// No copy - just marks ready
channel->send(msg);
// No copy - returns view
auto received = channel->receive<FloatVector>();
```

### Channel URIs

Psyne uses URIs to specify channel types:

- `memory://name` - In-process shared memory
- `ipc://name` - Inter-process communication
- `tcp://host:port` - TCP networking
- `unix:///path/to/socket` - Unix domain sockets
- `udp://multicast_ip:port` - UDP multicast

### Message Lifecycle

1. **Reserve**: Message constructor reserves space in buffer
2. **Fill**: Write data directly to buffer
3. **Send**: Mark message as ready
4. **Receive**: Get view of message
5. **Process**: Read message data
6. **Release**: Automatic when message goes out of scope

## Common Pitfalls

### 1. Using Message After Send

```cpp
FloatVector msg(*channel);
channel->send(msg);
// msg is now invalid!
msg[0] = 1.0f;  // ERROR: undefined behavior
```

### 2. Message Size Limits

```cpp
FloatVector msg(*channel);
msg.resize(1000000);  // ERROR if capacity < 1000000
```

Always check capacity:
```cpp
FloatVector msg(*channel);
std::cout << "Max capacity: " << msg.capacity() << std::endl;
```

### 3. Blocking Behavior

By default, receive is non-blocking:
```cpp
auto msg = channel->receive<FloatVector>();  // Returns immediately
if (!msg) {
    // No message available
}
```

## Next Steps

Now that you understand the basics:

1. Try different channel types (IPC, TCP)
2. Experiment with different message types
3. Measure performance with the built-in metrics
4. Read Tutorial 2: Message Types and Patterns

## Exercises

1. Modify the program to send 100 messages with 1000 floats each
2. Add timing to measure messages per second
3. Try using `ByteVector` instead of `FloatVector`
4. Create two channels and send messages between them

Happy messaging with Psyne!