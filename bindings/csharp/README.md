# Psyne C# Bindings

High-performance zero-copy messaging library for AI/ML applications, with C# bindings using P/Invoke.

## Features

- Zero-copy messaging for maximum performance
- Support for multiple transport types (memory, TCP, Unix sockets, UDP multicast)
- Configurable compression (LZ4, Zstd, Snappy)
- Thread-safe operation with multiple synchronization modes
- Comprehensive metrics and monitoring
- Async/await support
- Fluent builder API for easy channel configuration
- Full IDisposable pattern implementation

## Installation

### NuGet Package

```bash
dotnet add package Psyne
```

### From Source

1. Clone the repository
2. Build the native library
3. Build the C# project:

```bash
cd bindings/csharp/src/Psyne
dotnet build
```

## Quick Start

### Basic Usage

```csharp
using Psyne;

// Initialize the library
Psyne.Initialize();

try
{
    // Create a simple memory channel
    using var channel = Psyne.CreateMemoryChannel("test-channel");
    
    // Send a message
    channel.Send("Hello, World!");
    
    // Receive a message
    using var message = channel.Receive();
    if (message != null)
    {
        Console.WriteLine($"Received: {message.GetString()}");
    }
}
finally
{
    // Clean up the library
    Psyne.Cleanup();
}
```

### Builder Pattern

```csharp
using var channel = Psyne.CreateChannel()
    .Memory("high-perf-channel")
    .WithBufferSize(10, SizeUnit.MB)
    .MultipleProducerSingleConsumer()
    .WithLz4Compression()
    .WithMetrics()
    .Build();
```

### Async Operations

```csharp
// Async receive with cancellation
var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
var message = await channel.ReceiveAsync(cts.Token);

if (message != null)
{
    Console.WriteLine($"Received: {message.GetString()}");
    message.Dispose();
}
```

### Zero-Copy Operations

```csharp
// Reserve a message buffer
using var message = channel.ReserveMessage(1024);

// Get direct access to the buffer
var span = message.GetSpan();
// Write data directly to the span...

// Send the message
message.Send(messageType: 42);
```

## API Reference

### Channel Creation

#### Simple Factory Methods

```csharp
// Memory channel
var channel = Psyne.CreateMemoryChannel("channel-name", bufferSize: 1024 * 1024);

// TCP channel
var channel = Psyne.CreateTcpChannel("localhost", 8080);

// Unix socket channel
var channel = Psyne.CreateUnixSocketChannel("/tmp/psyne.sock");
```

#### Builder Pattern

```csharp
var channel = Psyne.CreateChannel()
    .WithUri("tcp://localhost:8080")
    .WithBufferSize(10, SizeUnit.MB)
    .WithMode(ChannelMode.MultipleProducerMultipleConsumer)
    .WithType(ChannelType.Multi)
    .WithZstdCompression()
    .WithMetrics()
    .Build();
```

### Message Operations

#### Sending Messages

```csharp
// Send string
channel.Send("Hello, World!");

// Send bytes
byte[] data = Encoding.UTF8.GetBytes("Hello");
channel.Send(data, messageType: 1);

// Zero-copy send
using var message = channel.ReserveMessage(1024);
message.SetString("Hello");
message.Send(messageType: 1);

// Raw data send
ReadOnlySpan<byte> data = stackalloc byte[] { 1, 2, 3, 4 };
channel.SendRaw(data, messageType: 2);
```

#### Receiving Messages

```csharp
// Blocking receive
using var message = channel.Receive();

// Non-blocking receive
if (channel.TryReceive(out var message, out var messageType))
{
    // Process message
    message?.Dispose();
}

// Async receive
var message = await channel.ReceiveAsync(cancellationToken);

// Raw data receive
Span<byte> buffer = stackalloc byte[1024];
var result = channel.ReceiveRaw(buffer, timeoutMs: 1000);
if (result.HasValue)
{
    Console.WriteLine($"Received {result.Value.BytesReceived} bytes of type {result.Value.MessageType}");
}
```

### Compression

```csharp
// Built-in compression configs
var lz4Config = CompressionConfig.Lz4();
var zstdConfig = CompressionConfig.Zstd();
var snappyConfig = CompressionConfig.Snappy();

// Custom compression config
var customConfig = new CompressionConfig
{
    Type = CompressionType.Zstd,
    Level = 6,
    MinSizeThreshold = 512,
    EnableChecksum = true
};

var channel = Psyne.CreateChannel()
    .Memory("compressed-channel")
    .WithCompression(customConfig)
    .Build();
```

### Metrics

```csharp
// Enable metrics
channel.EnableMetrics();

// Get metrics
var metrics = channel.GetMetrics();
Console.WriteLine($"Messages sent: {metrics.MessagesSent}");
Console.WriteLine($"Bytes sent: {metrics.BytesSent}");
Console.WriteLine($"Average message size: {metrics.AverageSentMessageSize:F2} bytes");

// Reset metrics
channel.ResetMetrics();
```

### Event-Driven Operations

```csharp
// Set up receive callback
channel.SetReceiveCallback((message, messageType) =>
{
    Console.WriteLine($"Received message of type {messageType}: {message.GetString()}");
    message.Dispose();
});
```

## Channel Modes

- **SingleProducerSingleConsumer**: Most efficient for 1:1 communication
- **SingleProducerMultipleConsumer**: One producer, multiple consumers
- **MultipleProducerSingleConsumer**: Multiple producers, one consumer  
- **MultipleProducerMultipleConsumer**: Most flexible, potentially less efficient

## Channel Types

- **Single**: Optimized for a single message type
- **Multi**: Supports multiple message types with type identifiers

## Transport Types

### Memory Channels
```csharp
var channel = Psyne.CreateChannel().Memory("buffer-name").Build();
```

### TCP Channels
```csharp
var channel = Psyne.CreateChannel().Tcp("localhost", 8080).Build();
```

### Unix Socket Channels
```csharp
var channel = Psyne.CreateChannel().UnixSocket("/tmp/psyne.sock").Build();
```

### UDP Multicast Channels
```csharp
var channel = Psyne.CreateChannel().UdpMulticast("239.1.1.1", 8080).Build();
```

## Error Handling

```csharp
try
{
    var channel = Psyne.CreateMemoryChannel("test");
    channel.Send("Hello");
}
catch (PsyneException ex)
{
    Console.WriteLine($"Psyne error: {ex.ErrorCode} - {ex.Message}");
}
```

## Best Practices

1. **Initialize once**: Call `Psyne.Initialize()` at application startup
2. **Clean up**: Call `Psyne.Cleanup()` at application shutdown
3. **Dispose resources**: Always dispose channels and messages
4. **Use using statements**: Leverage C#'s using statements for automatic disposal
5. **Zero-copy when possible**: Use `ReserveMessage()` and spans for maximum performance
6. **Choose appropriate channel mode**: Use SPSC when possible for best performance
7. **Enable metrics in production**: Monitor performance with built-in metrics

## Thread Safety

- Channels are thread-safe and can be used from multiple threads
- Messages should not be shared between threads
- Each thread should dispose its own messages

## Performance Tips

1. Use SPSC mode when you have single producer/consumer
2. Enable compression for large messages
3. Use zero-copy operations with `ReserveMessage()` and spans
4. Choose appropriate buffer sizes based on your message patterns
5. Monitor metrics to identify performance bottlenecks

## Examples

See the `examples/` directory for complete working examples demonstrating various features and usage patterns.