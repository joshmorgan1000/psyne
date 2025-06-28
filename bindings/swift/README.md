# Psyne Swift Bindings

Swift bindings for the Psyne high-performance zero-copy messaging library, providing idiomatic Swift APIs with full support for Swift concurrency (async/await).

## Features

- 🚀 **High Performance**: Zero-copy messaging with minimal overhead
- 🧵 **Thread Safety**: Support for SPSC, SPMC, MPSC, and MPMC channel modes
- 🔄 **Async/Await**: Full Swift concurrency support with async sequences
- 🏗️ **Builder Pattern**: Fluent API for channel configuration
- 📦 **Compression**: Built-in support for LZ4, Zstandard, and Snappy compression
- 🔌 **Multiple Transports**: Memory, IPC, TCP, Unix sockets, and WebSockets
- 📊 **Metrics**: Optional performance monitoring and debugging
- 🛡️ **Type Safety**: Protocol-oriented design with strong typing
- 📝 **Comprehensive**: Extensive documentation and examples

## Requirements

- Swift 5.7+
- macOS 11.0+ / iOS 14.0+ / tvOS 14.0+ / watchOS 7.0+
- Psyne C library

## Installation

### Swift Package Manager

Add this to your `Package.swift`:

```swift
dependencies: [
    .package(path: "path/to/psyne/bindings/swift")
]
```

Or add it through Xcode:
1. File → Add Package Dependencies
2. Enter the repository URL
3. Select the version/branch you want to use

## Quick Start

### Basic Usage

```swift
import Psyne

// Initialize the library
try Psyne.initialize()
defer { Psyne.cleanup() }

// Create a simple memory channel
let channel = try Psyne.createMemoryChannel(name: "my_channel")

// Send a text message
try channel.sendText("Hello, Psyne!")

// Receive the message
if let message = try channel.receiveMessage() {
    if let textMsg = message.asTextMessage() {
        print("Received: \(textMsg.text)")
    }
}
```

### Builder Pattern

```swift
// Create a channel with custom configuration
let channel = try Psyne.createChannel()
    .tcp(host: "localhost", port: 8080)
    .withBufferSize(2 * 1024 * 1024) // 2MB buffer
    .multipleProducersMultipleConsumers()
    .withLZ4Compression()
    .build()
```

### Async/Await Support

```swift
// Send data asynchronously
try await channel.sendDataAsync("Hello async world!".data(using: .utf8)!)

// Receive using async sequence
for await (data, messageType) in channel.dataSequence() {
    let message = String(data: data, encoding: .utf8) ?? "Binary data"
    print("Received: \(message)")
}
```

### Typed Messages

```swift
// Define a custom message type
struct MyMessage: MessageType, Codable {
    static let messageTypeID: UInt32 = 42
    
    let id: Int
    let content: String
    let timestamp: Date
}

// Send typed message
let message = MyMessage(id: 1, content: "Hello", timestamp: Date())
try channel.sendCodable(message)

// Receive typed message
if let receivedMessage = try channel.receiveMessage() {
    if receivedMessage.isType(MyMessage.self) {
        let myMessage = try receivedMessage.decodeCodable(as: MyMessage.self)
        print("Received: \(myMessage.content)")
    }
}
```

## Channel Types and Modes

### Channel Modes

- **SPSC** (Single Producer, Single Consumer): Highest performance, lock-free
- **SPMC** (Single Producer, Multiple Consumer): One writer, many readers
- **MPSC** (Multiple Producer, Single Consumer): Many writers, one reader
- **MPMC** (Multiple Producer, Multiple Consumer): Full multi-threading support

### Transport Types

- **Memory**: `memory://name` - In-process communication
- **IPC**: `ipc://name` - Inter-process shared memory
- **TCP**: `tcp://host:port` - Network communication
- **Unix Sockets**: `unix:///path` - Local domain sockets
- **WebSockets**: `ws://host:port` - WebSocket communication

## Compression

Psyne supports multiple compression algorithms:

```swift
// LZ4 - Fast compression/decompression
let channel1 = try Psyne.createChannel()
    .memory("fast")
    .withLZ4Compression()
    .build()

// Zstandard - Better compression ratio
let channel2 = try Psyne.createChannel()
    .memory("small")
    .withZstdCompression()
    .build()

// Custom compression configuration
let channel3 = try Psyne.createChannel()
    .memory("custom")
    .withCompression(CompressionConfig(
        type: .zstd,
        level: 6,
        minSizeThreshold: 256,
        enableChecksum: true
    ))
    .build()
```

## Built-in Message Types

Psyne provides several built-in message types:

```swift
// Text messages
try channel.sendText("Hello world!")

// Float arrays (for ML/AI)
let data: [Float] = [1.0, 2.0, 3.0, 4.0]
try channel.sendFloatArray(data)

// JSON messages
let json = ["key": "value", "number": 42]
try channel.sendJSON(json)

// Heartbeat messages
try channel.sendHeartbeat(sequenceNumber: 1, nodeID: "node-1")

// Control messages
try channel.sendControl(.start, parameters: ["mode": "fast"])
```

## Error Handling

The Swift bindings provide comprehensive error handling:

```swift
do {
    try channel.sendText("Hello")
} catch PsyneError.channelFull {
    print("Channel is full, try again later")
} catch PsyneError.channelStopped {
    print("Channel has been stopped")
} catch PsyneError.timeout {
    print("Operation timed out")
} catch {
    print("Other error: \(error)")
}
```

## Metrics and Monitoring

```swift
// Enable metrics
try channel.setMetricsEnabled(true)

// Get metrics
let metrics = try channel.getMetrics()
print("Messages sent: \(metrics.messagesSent)")
print("Bytes sent: \(metrics.bytesSent)")
print("Average message size: \(metrics.averageSentMessageSize)")

// Reset metrics
try channel.resetMetrics()
```

## Examples

The `Examples/` directory contains comprehensive examples:

- **BasicExample.swift**: Simple sending and receiving
- **AsyncExample.swift**: Async/await and async sequences
- **BuilderPatternExample.swift**: Channel configuration with builder pattern
- **CompressionExample.swift**: Compression algorithms comparison
- **ProducerConsumerExample.swift**: Multi-threaded producer-consumer pattern

## API Reference

### Core Classes

- **`Psyne`**: Main entry point and library management
- **`Channel`**: High-performance messaging channel
- **`ChannelBuilder`**: Fluent API for channel configuration
- **`Message`**: Outgoing message for writing data
- **`ReceivedMessage`**: Incoming message for reading data

### Enums

- **`ChannelMode`**: Threading and synchronization modes
- **`ChannelType`**: Single vs multi-type message support
- **`CompressionType`**: Available compression algorithms
- **`PsyneError`**: Error types with detailed descriptions

### Protocols

- **`MessageType`**: Protocol for typed messages
- **`AsyncSequence`**: Standard Swift async sequence support

## Performance Tips

1. **Choose the right channel mode**: Use SPSC for highest performance when possible
2. **Size your buffers appropriately**: Larger buffers reduce blocking but use more memory
3. **Use compression wisely**: Enable compression for large, repetitive data
4. **Reuse channels**: Channel creation has overhead, reuse when possible
5. **Enable metrics only when needed**: Metrics add small overhead

## Thread Safety

- All channel operations are thread-safe according to the chosen `ChannelMode`
- `Channel` objects can be shared between threads
- `Message` and `ReceivedMessage` objects are NOT thread-safe
- The Swift bindings add additional safety through move semantics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all examples compile and run
5. Submit a pull request

## License

This project is licensed under the same license as the main Psyne library.

## Support

- [GitHub Issues](https://github.com/joshmorgan1000/psyne/issues)
- [Documentation](https://github.com/joshmorgan1000/psyne/docs)
- [Examples](./Examples/)

---

For more information about Psyne, visit the [main repository](https://github.com/joshmorgan1000/psyne).