<div align="center">
  <img src="docs/assets/psyne_logo.png" alt="Psyne Logo" width="200"/>
  
  **Ultra-high-performance zero-copy messaging library for C++20**
  
  [![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/joshmorgan1000/psyne)
  [![C++ Standard](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  
  [![Linux GCC](https://img.shields.io/github/actions/workflow/status/joshmorgan1000/psyne/ci.yml?branch=main&label=Linux%20GCC&logo=linux)](https://github.com/joshmorgan1000/psyne/actions/workflows/ci.yml)
  [![Linux Clang](https://img.shields.io/github/actions/workflow/status/joshmorgan1000/psyne/ci.yml?branch=main&label=Linux%20Clang&logo=llvm)](https://github.com/joshmorgan1000/psyne/actions/workflows/ci.yml)
  [![macOS](https://img.shields.io/github/actions/workflow/status/joshmorgan1000/psyne/ci.yml?branch=main&label=macOS&logo=apple)](https://github.com/joshmorgan1000/psyne/actions/workflows/ci.yml)
  [![Windows MSVC](https://img.shields.io/github/actions/workflow/status/joshmorgan1000/psyne/ci.yml?branch=main&label=Windows%20MSVC&logo=windows)](https://github.com/joshmorgan1000/psyne/actions/workflows/ci.yml)
  [![Release](https://img.shields.io/github/v/release/joshmorgan1000/psyne?label=Release&logo=github)](https://github.com/joshmorgan1000/psyne/releases/latest)
  
  **The fastest messaging library you'll ever use. Period.**

  *Header Only!*

  ## [Philosophy](Philosophy.md)
</div>



## Performance

Real-world benchmarks on AMD Ryzen 7 3700X (8 cores, 16 threads):

| Pattern | Configuration | Throughput | Notes |
|---------|--------------|------------|-------|
| SPSC | 1P × 1C | 3.95M msgs/s | Single producer/consumer |
| MPSC | 32P × 1C | **22.5M msgs/s** | Peak performance |
| SPMC | 1P × 4C | 4.91M msgs/s | Perfect for work distribution |
| MPMC | 16P × 16C | 12.8M msgs/s | 256 threads! |

- **10 nanosecond** cross-thread latency
- **Zero-copy** architecture
- **Lock-free** atomic operations
- **Header-only** - just include and go

MacBook Pro M4 Pro Max (16-core, 40-core GPU):

| Pattern | Configuration | Throughput | Notes |
|---------|--------------|------------|-------|
| SPSC | 1P × 1C | 12.38M msgs/s | Single producer/consumer |
| MPSC | 4P × 1C | 19.03M msgs/s | Gradients |
| SPMC | 1P × 8C | **20.7M msgs/s** | Peak Performance |
| MPMC | 4P × 4C | 8.2M msgs/s| Balanced w/contention |

## Quick Start

```cpp
#include <psyne/psyne.hpp>

// Create a single-producer, single-consumer channel
auto channel = psyne::channel<MyMessage, psyne::spsc>();

// Producer thread
auto msg = channel.create_message();
msg->data = "Hello, World!";
channel.send_message(msg);

// Consumer thread
if (auto msg = channel.try_receive()) {
    std::cout << msg->data << std::endl;
}
```

## Features

### Channel Patterns
- **SPSC** - Single Producer, Single Consumer (lowest latency)
- **MPSC** - Multiple Producer, Single Consumer (highest throughput)
- **SPMC** - Single Producer, Multiple Consumer (work distribution)
- **MPMC** - Multiple Producer, Multiple Consumer (full flexibility)

### Transport Substrates
- **InProcess** - Zero-copy shared memory within process
- **IPC** - Zero-copy shared memory across processes
- **TCP** - Network transport with optional compression

### Key Benefits
- **Header-only** - No build complexity, just `#include`
- **Zero dependencies** - Pure C++20, no external libraries
- **Zero-copy** - Messages allocated directly in channel memory
- **Lock-free** - Wait-free progress guarantees where possible
- **Type-safe** - Full compile-time type checking

## Installation

Copy the `include/psyne` directory to your project:

```bash
cp -r include/psyne /path/to/your/project/include/
```

Or use CMake:

```cmake
add_subdirectory(psyne)
target_link_libraries(your_app psyne::psyne)
```

Requires C++20 (GCC 10+, Clang 12+, MSVC 2019+).

### Windows Support

On Windows, build with Visual Studio 2019 or later:

```batch
# Using provided build script
scripts\build_windows.bat

# Or manually with MSVC
cl /std:c++20 /O2 /EHsc /I"include" your_app.cpp
```

Note: Full coroutine support on Windows requires Boost.Asio. Basic channel functionality works without Boost.

## Advanced Examples

### High-Throughput Pipeline
```cpp
// 22.5M msgs/sec with multiple producers
auto channel = psyne::channel<DataPacket, psyne::mpsc>();

// Launch 32 producers
for (int i = 0; i < 32; ++i) {
    producers.emplace_back([&channel, id = i]() {
        while (running) {
            auto msg = channel.create_message();
            msg->producer_id = id;
            msg->timestamp = now();
            channel.send_message(msg);
        }
    });
}
```

### Cross-Process Communication
```cpp
// Process A - Producer
auto channel = psyne::channel<Frame, psyne::spsc, psyne::ipc>("camera_feed");
channel.send_message(frame);

// Process B - Consumer
auto channel = psyne::channel<Frame, psyne::spsc, psyne::ipc>("camera_feed");
auto frame = channel.receive();
```

### Work Distribution (SPMC)
```cpp
// One producer, multiple consumers for parallel processing
auto channel = psyne::channel<WorkItem, psyne::spmc>();

// Producer
while (auto work = get_next_work()) {
    channel.send_message(work);
}

// Multiple consumer threads
for (int i = 0; i < num_workers; ++i) {
    workers.emplace_back([&]() {
        while (auto work = channel.try_receive()) {
            process(work);
        }
    });
}
```

### Coroutine Support (C++20)
```cpp
// Asynchronous message reception with coroutines
auto channel = psyne::channel<Event, psyne::spsc>();

// Consumer coroutine
boost::asio::awaitable<void> consume_events(boost::asio::io_context& io) {
    while (running) {
        // Await message with timeout
        auto msg = co_await channel.async_receive(io, std::chrono::seconds(5));
        
        if (msg) {
            co_await process_event(*msg);
        } else {
            // Handle timeout
            std::cout << "No events received in 5 seconds\n";
        }
    }
}

// Run with boost::asio
boost::asio::io_context io_context;
boost::asio::co_spawn(io_context, consume_events(io_context), boost::asio::detached);
io_context.run();
```

### Backpressure Handling
```cpp
// Configure channel with backpressure policy
auto channel = psyne::channel<Data, psyne::mpsc>();

// Drop policy - drop messages when full
channel.set_backpressure_policy(
    std::make_unique<psyne::backpressure::DropPolicy>()
);

// Block policy - block producer with timeout
channel.set_backpressure_policy(
    std::make_unique<psyne::backpressure::BlockPolicy>(
        std::chrono::milliseconds(100)
    )
);

// Retry policy - exponential backoff
channel.set_backpressure_policy(
    std::make_unique<psyne::backpressure::RetryPolicy>(
        10, // max retries
        std::chrono::microseconds(10) // initial delay
    )
);

// Adaptive policy - changes strategy based on load
channel.set_backpressure_policy(
    std::make_unique<psyne::backpressure::AdaptivePolicy>()
);

// Custom callback policy
channel.set_backpressure_policy(
    std::make_unique<psyne::backpressure::CallbackPolicy>(
        []() -> bool {
            std::cerr << "Channel full! Shedding load...\n";
            return false; // Don't retry
        }
    )
);
```

## Architecture

Psyne v2.0 introduces a revolutionary architecture based on:

1. **Physical Substrates** - Handle memory allocation and transport
2. **Abstract Messages** - Type-safe views into channel memory
3. **Pattern Behaviors** - Implement coordination logic
4. **Channel Bridge** - Orchestrates the complete pipeline

This separation enables incredible performance while maintaining clean abstractions.

## Use Cases

- **High-Frequency Trading** - Sub-microsecond order routing
- **Video Game Engines** - Lock-free actor messaging
- **ML/AI Pipelines** - Zero-copy tensor transport
- **Robotics** - Real-time sensor fusion
- **Data Streaming** - High-throughput event processing

## Benchmarks

Run the comprehensive benchmark suite:

```bash
./benchmarks/beast_mode_test
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Built with ❤️ for extreme performance by the Psyne team.

