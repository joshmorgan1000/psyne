# Psyne Java Bindings

Java bindings for the Psyne zero-copy messaging library, providing high-performance inter-process and network communication for Java applications.

## Features

- **Zero-copy messaging** - Direct access to native memory through ByteBuffers
- **Multiple channel types** - Memory, TCP, IPC, UDP multicast
- **Builder pattern** - Fluent API for channel configuration
- **Compression support** - LZ4, Zstandard, and Snappy compression
- **Thread-safe** - Support for SPSC, SPMC, MPSC, and MPMC modes
- **Automatic resource management** - Try-with-resources support
- **Performance metrics** - Built-in performance monitoring

## Requirements

- Java 11 or higher
- Psyne native library installed
- Gradle 7.0+ (for building)

## Installation

### Building from Source

```bash
# Build the native JNI library and Java classes
./gradlew build

# Run tests
./gradlew test

# Build documentation
./gradlew javadoc
```

### Maven/Gradle Dependency

Once published to Maven Central:

```xml
<!-- Maven -->
<dependency>
    <groupId>com.psyne</groupId>
    <artifactId>psyne-java</artifactId>
    <version>0.1.1</version>
</dependency>
```

```gradle
// Gradle
implementation 'com.psyne:psyne-java:0.1.1'
```

## Quick Start

```java
import com.psyne.*;

// Initialize the library
Psyne.init();

// Create a channel
try (Channel channel = Channel.builder()
        .uri("memory://my-channel")
        .bufferSize(1024 * 1024)  // 1MB
        .mode(ChannelMode.SPSC)
        .build()) {
    
    // Send a message
    channel.send("Hello, Psyne!".getBytes(), 1);
    
    // Receive a message
    Channel.ReceivedMessage received = channel.receive();
    if (received != null) {
        try (Message msg = received.getMessage()) {
            byte[] data = msg.toByteArray();
            System.out.println("Received: " + new String(data));
        }
    }
}
```

## Examples

The `examples` directory contains several demonstration programs:

- **BasicExample** - Simple send/receive operations
- **CompressionExample** - Using compression for large messages
- **ProducerConsumerExample** - Multi-threaded producer/consumer pattern
- **NetworkExample** - TCP client/server communication
- **BuilderPatternExample** - Various channel configurations

To run an example:

```bash
./gradlew runExample -Pexample=BasicExample
```

## API Documentation

### Channel Creation

Channels are created using the builder pattern:

```java
Channel channel = Channel.builder()
    .uri("tcp://localhost:8080")     // Required: channel URI
    .bufferSize(4 * 1024 * 1024)     // Optional: buffer size (default 1MB)
    .mode(ChannelMode.MPMC)          // Optional: sync mode (default SPSC)
    .type(ChannelType.MULTI)         // Optional: message types (default SINGLE)
    .compression(compressionConfig)   // Optional: compression settings
    .build();
```

### Supported URIs

- `memory://name` - In-process shared memory channel
- `tcp://host:port` - TCP network channel
- `ipc:///path/to/socket` - Unix domain socket (Linux/macOS)
- `udp://host:port` - UDP multicast channel

### Compression

Configure compression using the CompressionConfig builder:

```java
CompressionConfig compression = CompressionConfig.builder()
    .type(CompressionType.LZ4)
    .level(9)                    // Compression level (algorithm-specific)
    .minSizeThreshold(1024)      // Only compress messages > 1KB
    .enableChecksum(true)        // Enable integrity checking
    .build();
```

### Zero-Copy Operations

For maximum performance, use direct ByteBuffer access:

```java
// Sending with zero-copy
try (Message message = channel.reserve(1024)) {
    ByteBuffer buffer = message.getData();
    buffer.putInt(42);
    buffer.putDouble(3.14);
    message.send(1);
}

// Receiving with zero-copy
Channel.ReceivedMessage received = channel.receive();
if (received != null) {
    try (Message msg = received.getMessage()) {
        ByteBuffer buffer = msg.getData();
        int value = buffer.getInt();
        double pi = buffer.getDouble();
    }
}
```

### Performance Metrics

Monitor channel performance:

```java
channel.enableMetrics(true);

// ... use the channel ...

Metrics metrics = channel.getMetrics();
System.out.println("Messages sent: " + metrics.getMessagesSent());
System.out.println("Throughput: " + metrics.getBytesSent() + " bytes");
System.out.println("Send blocks: " + metrics.getSendBlocks());
```

## Thread Safety

Psyne channels support different synchronization modes:

- **SPSC** - Single producer, single consumer (fastest)
- **SPMC** - Single producer, multiple consumers
- **MPSC** - Multiple producers, single consumer
- **MPMC** - Multiple producers, multiple consumers

Choose the appropriate mode based on your threading model.

## Error Handling

All Psyne operations that can fail throw `PsyneException`:

```java
try {
    channel.send(data, type);
} catch (PsyneException e) {
    System.err.println("Error: " + e.getMessage());
    
    // Get the specific error code
    PsyneException.ErrorCode code = e.getErrorCode();
    switch (code) {
        case CHANNEL_FULL:
            // Handle full channel
            break;
        case TIMEOUT:
            // Handle timeout
            break;
        default:
            // Handle other errors
    }
}
```

## Resource Management

All Psyne resources implement `AutoCloseable` for use with try-with-resources:

```java
try (Channel channel = Channel.builder().uri("memory://test").build();
     Message msg = channel.reserve(1024)) {
    // Use channel and message
    // Automatic cleanup on scope exit
}
```

## Performance Tips

1. **Use appropriate buffer sizes** - Larger buffers reduce contention but use more memory
2. **Choose the right sync mode** - SPSC is fastest when applicable
3. **Enable compression for large messages** - Reduces bandwidth at the cost of CPU
4. **Use zero-copy operations** - Direct ByteBuffer access avoids copying
5. **Batch operations** - Send multiple messages before receiving
6. **Monitor metrics** - Use metrics to identify bottlenecks

## Building the Native Library

The JNI native library must be built before using the Java bindings:

```bash
# From the bindings/java directory
gradle buildNative

# The native library will be in bindings/native/
# - Linux: libpsyne_jni.so
# - macOS: libpsyne_jni.dylib
# - Windows: psyne_jni.dll
```

## License

See the main Psyne project for license information.