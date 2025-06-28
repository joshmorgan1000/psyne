# Psyne.jl Examples

This directory contains comprehensive examples demonstrating the capabilities of Psyne.jl, the Julia bindings for the Psyne zero-copy messaging library.

## Quick Start

Make sure Psyne.jl is installed and the native Psyne library is available in your system's library path.

```julia
# Add the Psyne.jl package to your environment
using Pkg
Pkg.develop(path="../")  # Assuming you're in the examples directory

# Run any example
julia basic_example.jl
```

## Examples Overview

### 1. `basic_example.jl`
**Fundamental usage of Psyne.jl**

Demonstrates:
- Creating memory channels
- Sending/receiving different data types (Float32, Float64, UInt8, Complex)
- Automatic type detection
- Psyne message types (FloatVector, ByteVector, etc.)
- Array operations and broadcasting
- Proper resource cleanup

```bash
julia basic_example.jl
```

**Perfect for**: First-time users, understanding basic concepts

---

### 2. `performance_example.jl`
**Performance monitoring and optimization**

Demonstrates:
- High-performance channel configuration
- Performance testing and benchmarking
- Metrics collection and analysis
- Compression impact measurement
- Real-time monitoring
- Health checking
- Algorithm benchmarking

```bash
julia performance_example.jl
```

**Perfect for**: Performance tuning, production monitoring

---

### 3. `network_example.jl`
**Network communication with TCP channels**

Demonstrates:
- TCP server/client channels
- Network communication with compression
- Error handling and timeouts
- Different message sizes and types
- Round-trip performance measurement
- Compression effectiveness testing

```bash
# Run as server (default port 8080)
julia network_example.jl server

# Run as client (in another terminal)
julia network_example.jl client

# Custom port
julia network_example.jl server 9000
julia network_example.jl client localhost 9000
```

**Perfect for**: Distributed systems, microservices, remote computing

---

### 4. `producer_consumer_example.jl`
**Multi-threaded producer-consumer patterns**

Demonstrates:
- Multi-threaded communication
- Different synchronization modes (SPSC, MPSC, SPMC, MPMC)
- Thread-safe statistics collection
- Performance comparison between modes
- Timeout handling and error recovery
- Channel health monitoring

```bash
# Run with multiple threads for best results
julia -t 4 producer_consumer_example.jl
```

**Perfect for**: Multi-threaded applications, parallel computing, load balancing

---

### 5. `array_operations_example.jl`
**Advanced array operations and scientific computing**

Demonstrates:
- Broadcasting and in-place operations
- Linear algebra with matrices
- Large array performance testing
- Scientific computing workflows (signal processing)
- Memory-efficient operations with ByteVector
- Complex number operations
- Type reinterpretation for binary data

```bash
julia array_operations_example.jl
```

**Perfect for**: Scientific computing, machine learning, signal processing

## Running Examples

### Prerequisites

1. **Julia**: Version 1.6 or later
2. **Psyne Library**: Native Psyne library must be available
3. **Multiple Threads**: Some examples benefit from multiple Julia threads

### Installation

```bash
# Clone or download the Psyne repository
git clone https://github.com/your-repo/psyne.git
cd psyne/bindings/julia

# Install dependencies (if any)
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### Running Individual Examples

```bash
# Basic usage
julia examples/basic_example.jl

# Performance testing
julia examples/performance_example.jl

# Network communication (requires two terminals)
julia examples/network_example.jl server
julia examples/network_example.jl client

# Multi-threaded patterns
julia -t 4 examples/producer_consumer_example.jl

# Scientific computing
julia examples/array_operations_example.jl
```

## Example Features Matrix

| Feature | Basic | Performance | Network | Producer-Consumer | Array Ops |
|---------|-------|-------------|---------|-------------------|-----------|
| Channel Creation | ✓ | ✓ | ✓ | ✓ | ✓ |
| Memory Channels | ✓ | ✓ | - | ✓ | ✓ |
| TCP Channels | - | - | ✓ | - | - |
| Compression | - | ✓ | ✓ | - | - |
| Metrics | - | ✓ | ✓ | ✓ | ✓ |
| Multi-threading | - | ✓ | - | ✓ | - |
| Broadcasting | ✓ | - | - | - | ✓ |
| Linear Algebra | - | - | - | - | ✓ |
| Error Handling | ✓ | ✓ | ✓ | ✓ | ✓ |
| Performance Tuning | - | ✓ | ✓ | ✓ | ✓ |

## Common Patterns

### Channel Creation

```julia
# Memory channel (in-process)
ch = channel("memory://buffer_name")

# TCP server
ch = channel("tcp://:8080", mode=MPMC)

# TCP client with compression
config = lz4_config(level=3)
ch = channel("tcp://server:8080", compression=config)

# High-performance SPSC
ch = channel("memory://fast", buffer_size=16*1024*1024, mode=SPSC)
```

### Data Transmission

```julia
# Basic types
send(ch, Float32[1.0, 2.0, 3.0])
send(ch, rand(Float64, 10, 20))
send(ch, UInt8[0x48, 0x65, 0x6c, 0x6c, 0x6f])

# Receive with type specification
data = receive(ch, Vector{Float32})
matrix = receive(ch, Matrix{Float64})

# Automatic type detection
auto_data = receive(ch)
```

### Performance Monitoring

```julia
# Enable metrics
enable_metrics!(ch, true)

# Get current metrics
metrics = get_metrics(ch)
println("Messages sent: $(metrics.messages_sent)")

# Monitor over time
snapshots = monitor(ch, 10.0, interval_seconds=1.0)

# Health check
health = health_check(ch)
if health.status != :healthy
    println("Issues: $(health.issues)")
end
```

### Error Handling

```julia
try
    data = receive(ch, Vector{Float32}, timeout_ms=1000)
catch e
    if isa(e, PsyneError) && e.code == PSYNE_ERROR_TIMEOUT
        println("Receive timed out")
    else
        println("Unexpected error: $e")
    end
end
```

## Troubleshooting

### Common Issues

1. **"Cannot find libpsyne"**
   - Ensure the Psyne native library is built and in your library path
   - Set `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) if needed

2. **"Channel creation failed"**
   - Check URI format (e.g., "memory://name", "tcp://host:port")
   - Ensure port is available for TCP channels
   - Verify buffer size is reasonable

3. **"No message available"**
   - Use timeouts for receive operations
   - Check if sender and receiver are using compatible message types
   - Verify channel mode supports your usage pattern

4. **Poor performance**
   - Use SPSC mode for highest performance when possible
   - Increase buffer size for large messages
   - Enable compression for network channels
   - Use multiple threads with appropriate channel modes

### Performance Tips

1. **Choose the right channel mode**:
   - SPSC: Fastest for single producer/consumer
   - MPSC: Multiple producers, single consumer
   - SPMC: Single producer, multiple consumers
   - MPMC: Full multi-threading support

2. **Buffer sizing**:
   - Larger buffers reduce blocking but use more memory
   - Size based on message size and frequency
   - Start with 1-4MB for most applications

3. **Compression**:
   - Use LZ4 for speed, Zstd for compression ratio
   - Set appropriate minimum size thresholds
   - Test with your actual data patterns

4. **Memory management**:
   - Close channels when done
   - Monitor metrics to detect issues
   - Use health checks in production

## Contributing

To add new examples:

1. Create a new `.jl` file in this directory
2. Follow the existing pattern with clear documentation
3. Include comprehensive error handling
4. Add performance measurements where relevant
5. Update this README with the new example

## Support

For questions about these examples or Psyne.jl in general:

- Check the main Psyne documentation
- Review the source code in `../src/`
- Run examples with `-v` flag for verbose output
- Enable metrics and health checks for debugging