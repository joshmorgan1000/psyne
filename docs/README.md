# Psyne Documentation

Welcome to the Psyne documentation! This folder contains comprehensive guides and references for using the Psyne zero-copy messaging library.

## Documentation Structure

### Getting Started
- [Getting Started Guide](getting-started.md) - Quick introduction and basic examples
- [Overview](overview.md) - High-level architecture and design principles

### API Documentation
- [API Reference](api-reference.md) - Complete API documentation for all classes and functions

### Advanced Topics
- [IPC and TCP Channels](channels.md) - Network and inter-process communication guide

### Examples
The [examples](../examples/) directory contains working code examples:
- `simple_messaging.cpp` - Basic zero-copy messaging demonstration
- `producer_consumer.cpp` - Event-driven producer/consumer pattern
- `multi_type_channel.cpp` - Using channels with multiple message types

## Quick Links

### Core Concepts
- **Zero-Copy Architecture**: Messages are views into pre-allocated buffers
- **Event-Driven Processing**: Asynchronous callbacks for message handling
- **Type Safety**: Compile-time type checking for messages
- **Performance**: Lock-free algorithms for high throughput

### Common Use Cases
1. **High-frequency sensor data**: Use single-type SPSC channels
2. **Multi-process pipelines**: Use IPC channels with appropriate sync modes
3. **Distributed systems**: Use TCP channels for network transparency
4. **GPU integration**: Messages can be directly mapped to GPU buffers

### Performance Guidelines
- Use single-type channels when possible (no overhead)
- Choose the right synchronization mode (SPSC, SPMC, MPSC, MPMC)
- Pre-size buffers appropriately to avoid contention
- Use event-driven callbacks instead of polling

## Building and Installing

See the main [README](../README.md) for build instructions.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the main repository.

## License

Psyne is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.