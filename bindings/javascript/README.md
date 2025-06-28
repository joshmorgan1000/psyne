# Psyne JavaScript/TypeScript Bindings

High-performance zero-copy messaging library for Node.js applications, optimized for AI/ML workloads.

[![npm version](https://badge.fury.io/js/psyne.svg)](https://badge.fury.io/js/psyne)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js CI](https://github.com/joshmorgan1000/psyne/workflows/Node.js%20CI/badge.svg)](https://github.com/joshmorgan1000/psyne/actions)

## Overview

Psyne is a cutting-edge messaging library designed for high-performance, zero-copy communication between processes and threads. The JavaScript/TypeScript bindings provide a modern, Promise-based API with EventEmitter integration for real-time applications.

### Key Features

- **Zero-Copy Messaging**: Minimal overhead data transfer using shared memory and advanced buffer management
- **Multiple Transport Types**: In-memory, IPC, TCP, Unix sockets, WebSockets, UDP multicast, and RDMA
- **Thread-Safe Modes**: SPSC, SPMC, MPSC, and MPMC for different concurrency patterns
- **Compression Support**: LZ4, Zstandard, and Snappy compression algorithms
- **TypeScript Support**: Full type definitions with comprehensive JSDoc documentation
- **Reliability Features**: Acknowledgments, retries, heartbeats, and replay buffers
- **Performance Monitoring**: Built-in metrics and benchmarking tools
- **EventEmitter Integration**: Real-time message streaming with Node.js event patterns

## Installation

### Prerequisites

- Node.js 16.0.0 or higher
- C++20 compatible compiler (GCC 10+, Clang 12+, or MSVC 2019+)
- CMake 3.16 or higher
- Python 3.7+ (for node-gyp)

### Install from npm

```bash
npm install psyne
```

### Build from Source

```bash
git clone https://github.com/joshmorgan1000/psyne.git
cd psyne/bindings/javascript
npm install
npm run build
```

## Quick Start

### Basic Messaging

```javascript
const { Psyne, ChannelMode } = require('psyne');

async function basicExample() {
  // Create a Psyne instance
  const psyne = new Psyne();
  
  // Create an in-memory channel
  const channel = psyne.createChannel('memory://example', {
    mode: ChannelMode.SPSC,
    enableMetrics: true
  });
  
  // Send data
  await channel.send([1.0, 2.0, 3.0, 4.0]);
  await channel.send('Hello, Psyne!');
  await channel.send(new Float32Array([10.5, 20.3, 30.7]));
  
  // Receive data
  const message1 = await channel.receive();
  const message2 = await channel.receive();
  const message3 = await channel.receive();
  
  console.log('Received:', message1.data, message2.data, message3.data);
  
  // Show performance metrics
  const metrics = channel.getMetrics();
  console.log(`Processed ${metrics.messagesSent} messages`);
  
  // Clean up
  channel.close();
}

basicExample().catch(console.error);
```

### TypeScript with Builder Pattern

```typescript
import { 
  Channel, 
  ChannelMode, 
  CompressionType,
  FloatVector,
  DoubleMatrix 
} from 'psyne';

async function advancedExample(): Promise<void> {
  // Create channel with fluent builder
  const channel = Channel.builder()
    .uri('memory://advanced')
    .mode(ChannelMode.MPSC)
    .bufferSize(2 * 1024 * 1024)
    .enableMetrics()
    .compression({
      type: CompressionType.LZ4,
      level: 3,
      enableChecksum: true
    })
    .build();

  // Send typed messages
  const vector = new FloatVector([1.1, 2.2, 3.3]);
  const matrix = new DoubleMatrix(2, 2, [1, 2, 3, 4]);
  
  await channel.send(vector);
  await channel.send(matrix);
  
  // Receive with type information
  const msg1 = await channel.receive();
  const msg2 = await channel.receive();
  
  console.log('Vector:', msg1?.data);
  console.log('Matrix:', msg2?.data);
  
  channel.close();
}
```

### Event-Driven Messaging

```javascript
const { createChannel, ChannelMode } = require('psyne');

async function eventDrivenExample() {
  const channel = createChannel('memory://events')
    .mode(ChannelMode.SPSC)
    .enableMetrics()
    .build();

  // Set up event handlers
  channel.on('message', (message) => {
    console.log('Received:', message.type, message.data);
  });

  channel.on('error', (error) => {
    console.error('Channel error:', error);
  });

  // Start listening for messages
  channel.startListening();

  // Send messages
  await channel.send([1, 2, 3]);
  await channel.send('Hello Events!');
  await channel.send({ type: 'object', value: 42 });

  // Stop listening after a delay
  setTimeout(() => {
    channel.stopListening();
    channel.close();
  }, 1000);
}
```

### Network Communication

```javascript
const { createChannel } = require('psyne');

// TCP Server
async function createServer() {
  const server = createChannel('tcp://:8080')
    .mode(ChannelMode.MPSC)
    .enableMetrics()
    .buildReliable({
      enableAcknowledgments: true,
      maxRetries: 3
    });

  server.on('message', async (message) => {
    console.log('Server received:', message.data);
    
    // Echo back with timestamp
    await server.send({
      echo: message.data,
      serverTime: new Date().toISOString()
    });
  });

  server.startListening();
  return server;
}

// TCP Client
async function createClient() {
  const client = createChannel('tcp://localhost:8080')
    .enableMetrics()
    .buildReliable();

  client.on('message', (message) => {
    console.log('Client received:', message.data);
  });

  client.startListening();
  
  // Send data to server
  await client.send('Hello from client!');
  
  return client;
}
```

## API Reference

### Channel Creation

#### Psyne Class

```typescript
class Psyne {
  constructor(options?: PsyneOptions);
  createChannel(uri: string, options?: ChannelOptions): Channel;
  createReliableChannel(uri: string, options?: ChannelOptions, reliabilityOptions?: ReliabilityOptions): Channel;
  getChannel(uri: string): Channel | undefined;
  closeAll(): void;
}
```

#### Builder Pattern

```typescript
Channel.builder()
  .uri(uri: string)
  .mode(mode: ChannelMode)
  .bufferSize(size: number)
  .enableMetrics()
  .compression(config: CompressionConfig)
  .build(): Channel;
```

### Channel URIs

| Scheme | Format | Description |
|--------|--------|-------------|
| `memory://` | `memory://name` | In-process shared memory |
| `ipc://` | `ipc://name` | Inter-process communication |
| `tcp://` | `tcp://host:port` or `tcp://:port` | TCP client or server |
| `unix://` | `unix:///path/to/socket` | Unix domain sockets |
| `ws://` | `ws://host:port` or `ws://:port` | WebSocket client or server |
| `multicast://` | `multicast://address:port` | UDP multicast |

### Channel Modes

- **SPSC**: Single Producer, Single Consumer (highest performance)
- **SPMC**: Single Producer, Multiple Consumer
- **MPSC**: Multiple Producer, Single Consumer
- **MPMC**: Multiple Producer, Multiple Consumer

### Message Types

#### Built-in Types

```typescript
// Numeric arrays
const floatVector = new FloatVector([1.0, 2.0, 3.0]);
const doubleMatrix = new DoubleMatrix(2, 3, [1, 2, 3, 4, 5, 6]);

// Binary data
const byteVector = new ByteVector(Buffer.from('binary data'));

// 3D graphics
const vector3 = new Vector3f(1.0, 2.0, 3.0);
const matrix4 = new Matrix4x4f(identityMatrix);

// ML/AI data
const tensor = new MLTensor([10, 20, 30], 'NCHW', data);
const sparse = new SparseMatrix(rows, cols, values, indices, pointers);
```

#### Custom Data

```javascript
// Send any serializable data
await channel.send([1, 2, 3]);           // Number array
await channel.send('text message');      // String
await channel.send({ key: 'value' });    // Object
await channel.send(new Float32Array());  // Typed array
await channel.send(Buffer.from('data')); // Buffer
```

### Error Handling

```typescript
import { 
  PsyneError, 
  ChannelError, 
  MessageError, 
  TimeoutError 
} from 'psyne/errors';

try {
  await channel.send(data);
} catch (error) {
  if (error instanceof ChannelError) {
    console.error('Channel error:', error.message);
  } else if (error instanceof TimeoutError) {
    console.error('Operation timed out:', error.timeout);
  }
}
```

### Performance Monitoring

```typescript
import { createPerformanceMonitor } from 'psyne/utils';

const monitor = createPerformanceMonitor(channel, 1000);

monitor.onUpdate((metrics) => {
  console.log(`Rate: ${metrics.messageRate} msg/s`);
  console.log(`Throughput: ${metrics.throughputMbps} MB/s`);
  console.log(`Efficiency: ${metrics.efficiency * 100}%`);
});

monitor.start();
```

## Examples

The `examples/` directory contains comprehensive examples:

### Basic Examples
- `examples/basic/simple-messaging.js` - Basic send/receive operations
- `examples/basic/producer-consumer.js` - EventEmitter pattern usage
- `examples/basic/builder-pattern.ts` - TypeScript builder pattern

### Advanced Examples
- `examples/advanced/compression-demo.js` - Compression algorithms comparison
- `examples/advanced/reliability-features.js` - Error recovery and retries

### Performance Examples
- `examples/performance/benchmark-suite.js` - Comprehensive performance testing
- `examples/performance/memory-stress-test.js` - Memory usage analysis

### Network Examples
- `examples/network/tcp-client-server.js` - TCP communication
- `examples/network/websocket-chat.js` - WebSocket real-time messaging
- `examples/network/multicast-publisher.js` - UDP multicast broadcasting

### Run Examples

```bash
# Basic messaging
node examples/basic/simple-messaging.js

# Producer-consumer pattern
node examples/basic/producer-consumer.js

# TypeScript builder pattern
npx ts-node examples/basic/builder-pattern.ts

# Compression comparison
node examples/advanced/compression-demo.js

# Performance benchmarks
node examples/performance/benchmark-suite.js

# TCP server
node examples/network/tcp-client-server.js server 8080

# TCP client
node examples/network/tcp-client-server.js client 8080 localhost
```

## Performance

Psyne is designed for high-performance scenarios:

### Benchmarks

Typical performance on modern hardware:

| Message Size | Throughput | Latency (P99) | Mode |
|--------------|------------|---------------|------|
| 64 bytes | 2.5M msg/s | 15μs | SPSC |
| 4KB | 800MB/s | 25μs | SPSC |
| 64KB | 1.2GB/s | 150μs | SPSC |
| 1MB | 2.8GB/s | 800μs | SPSC |

### Optimization Tips

1. **Choose the Right Mode**: Use SPSC for highest performance when possible
2. **Buffer Sizing**: Match buffer size to your message patterns
3. **Compression**: Use LZ4 for speed, Zstd for ratio
4. **Message Batching**: Send multiple small messages together
5. **TypedArrays**: Use TypedArrays for zero-copy numeric data

### Memory Usage

- **Zero-copy**: No data copying for TypedArrays and Buffers
- **Shared buffers**: Efficient memory usage with multiple channels
- **Configurable**: Adjust buffer sizes based on requirements

## Building and Testing

### Development Setup

```bash
git clone https://github.com/joshmorgan1000/psyne.git
cd psyne/bindings/javascript
npm install
```

### Build Commands

```bash
# Build native addon and TypeScript
npm run build

# Build only native addon
npm run build:native

# Build only TypeScript
npm run build:js

# Clean build artifacts
npm run clean
```

### Testing

```bash
# Run test suite
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

### Linting and Formatting

```bash
# Lint code
npm run lint

# Fix linting issues
npm run lint:fix

# Format code
npm run format
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### Development Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Code Standards

- Follow TypeScript best practices
- Add JSDoc comments for public APIs
- Include unit tests for new features
- Maintain backward compatibility
- Use conventional commit messages

## License

MIT License. See [LICENSE](../../LICENSE) for details.

## Support

- **Documentation**: [Full API Docs](https://github.com/joshmorgan1000/psyne/docs)
- **Issues**: [GitHub Issues](https://github.com/joshmorgan1000/psyne/issues)
- **Discussions**: [GitHub Discussions](https://github.com/joshmorgan1000/psyne/discussions)

## Related Projects

- [Psyne Core](../../) - C++ library
- [Python Bindings](../python/) - Python support
- [Rust Bindings](../rust/) - Rust support
- [Java Bindings](../java/) - Java support

---

**Psyne**: High-performance messaging for the AI/ML era.