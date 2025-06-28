# Psyne.jl - Julia Bindings for Psyne

High-performance zero-copy messaging library for Julia, optimized for AI/ML applications.

Psyne.jl provides Julia bindings to the native Psyne C++ library, enabling efficient inter-process and inter-thread communication with minimal overhead through zero-copy message passing.

## Features

- **Zero-copy messaging** for maximum performance
- **Multiple transport types**: memory, IPC, TCP, Unix sockets, multicast
- **Configurable synchronization modes**: SPSC, SPMC, MPSC, MPMC
- **Built-in compression support**: LZ4, Zstd, Snappy
- **Type-safe operations** with Julia's type system
- **Performance metrics and monitoring**
- **Array types and broadcasting support**
- **Parametric types and multiple dispatch**
- **Comprehensive error handling**

## Installation

### Prerequisites

1. **Julia**: Version 1.6 or later
2. **Psyne Library**: The native Psyne C++ library must be built and available

### Building Psyne Library

```bash
# From the Psyne repository root
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### Installing Psyne.jl

```julia
# In Julia REPL
using Pkg

# Install from local directory (if you have the source)
Pkg.develop(path="path/to/psyne/bindings/julia")

# Or add to your project
Pkg.add(url="https://github.com/your-repo/psyne.git", subdir="bindings/julia")
```

## Quick Start

```julia
using Psyne

# Create a channel
ch = channel("memory://quickstart")

# Send data
data = Float32[1.0, 2.0, 3.14159, 4.0, 5.0]
send(ch, data)

# Receive data
received = receive(ch, Vector{Float32})
println("Received: $received")

# Clean up
close(ch)
```

## Basic Usage

### Creating Channels

```julia
using Psyne

# Memory channel (in-process)
ch1 = channel("memory://buffer1")

# IPC channel (inter-process)
ch2 = channel("ipc://shared_data", buffer_size=2^20)

# TCP server
server = channel("tcp://:8080", mode=MPMC)

# TCP client with compression
config = lz4_config(level=3)
client = channel("tcp://localhost:8080", compression=config)

# High-performance SPSC channel
fast_ch = channel("memory://fast", 
                 buffer_size=16*1024*1024,
                 mode=SPSC)
```

### Sending and Receiving Data

```julia
ch = channel("memory://data")

# Built-in Julia types
send(ch, Float32[1.0, 2.0, 3.0])
send(ch, rand(Float64, 10, 20))
send(ch, UInt8[0x48, 0x65, 0x6c, 0x6c, 0x6f])  # "Hello"
send(ch, Complex{Float32}[1+2im, 3+4im])

# Receive with type specification
float_data = receive(ch, Vector{Float32})
matrix_data = receive(ch, Matrix{Float64})
byte_data = receive(ch, Vector{UInt8})
complex_data = receive(ch, Vector{Complex{Float32}})

# Automatic type detection
auto_data = receive(ch)  # Type inferred from message
```

### Using Psyne Message Types

```julia
# FloatVector for dynamic arrays
float_msg = FloatVector(1000)
for i in 1:length(float_msg)
    float_msg[i] = sin(i * 0.1)
end
send(ch, float_msg)

# Broadcasting operations
float_msg.data .*= 2.0f0
float_msg.data .+= 1.0f0

# DoubleMatrix for 2D data
matrix_msg = DoubleMatrix(10, 20)
matrix_msg.data .= rand(10, 20)
send(ch, matrix_msg)

# ByteVector for binary data
byte_msg = ByteVector(1024)
byte_msg.data .= rand(UInt8, 1024)
send(ch, byte_msg)
```

## Advanced Features

### Performance Monitoring

```julia
# Enable metrics
ch = channel("memory://perf")
enable_metrics!(ch, true)

# Send some data
for i in 1:1000
    send(ch, rand(Float32, 100))
    receive(ch, Vector{Float32})
end

# Get metrics
metrics = get_metrics(ch)
println("Messages sent: $(metrics.messages_sent)")
println("Bytes sent: $(metrics.bytes_sent)")

# Calculate throughput
tput = throughput(metrics, 10.0)  # 10 seconds
println("Throughput: $(tput.mbps) Mbps")

# Health check
health = health_check(ch)
println("Channel status: $(health.status)")
```

### Compression

```julia
# Different compression algorithms
lz4_config = lz4_config(level=1)      # Fast compression
zstd_config = zstd_config(level=6)    # High compression
snappy_config = snappy_config()       # Balanced

# Create compressed channel
ch = channel("tcp://localhost:8080", compression=lz4_config)

# Estimate compression effectiveness
data = rand(Float32, 10000)
ratio = estimate_compression_ratio(data, lz4_config)
println("Estimated compression ratio: $ratio")

# Get recommendations
recommended = recommend_compression(data, priority=:speed)
```

### Multi-threading

```julia
# Run with: julia -t 4 script.jl

using Base.Threads

# Create MPMC channel for multi-threading
ch = channel("memory://mt", mode=MPMC)

# Producer task
producer_task = @spawn begin
    for i in 1:1000
        data = rand(Float32, 100)
        send(ch, data)
        sleep(0.001)  # 1ms
    end
end

# Consumer tasks
consumer_tasks = [@spawn begin
    for i in 1:250  # 1000/4 threads
        data = receive(ch, Vector{Float32}, timeout_ms=5000)
        # Process data...
    end
end for _ in 1:4]

# Wait for completion
wait(producer_task)
for task in consumer_tasks
    wait(task)
end
```

### Network Communication

```julia
# Server (run in one process)
server = channel("tcp://:8080", mode=MPMC)
enable_metrics!(server, true)

# Handle clients in a loop
while !is_stopped(server)
    try
        data = receive(server, timeout_ms=1000)
        # Process and send response
        send(server, process_data(data))
    catch e
        if isa(e, PsyneError) && e.code == PSYNE_ERROR_TIMEOUT
            continue  # No message, continue waiting
        else
            println("Error: $e")
            break
        end
    end
end

# Client (run in another process)
client = channel("tcp://localhost:8080")
send(client, my_data)
response = receive(client, expected_type, timeout_ms=5000)
```

## API Reference

### Core Types

#### `PsyneChannel`
Main channel abstraction for communication.

#### Enums
- `ChannelMode`: SPSC, SPMC, MPSC, MPMC
- `ChannelType`: SingleType, MultiType
- `CompressionType`: None, LZ4, Zstd, Snappy

#### Message Types
- `FloatVector`: Dynamic Float32 arrays
- `DoubleMatrix`: 2D Float64 matrices
- `ByteVector`: Raw byte arrays
- `ComplexVector`: Complex{Float32} arrays

### Functions

#### Channel Management
- `channel(uri; kwargs...)`: Create a channel
- `close(channel)`: Close and cleanup channel
- `stop!(channel)`: Stop channel operations
- `is_stopped(channel)`: Check if channel is stopped

#### Message Operations
- `send(channel, data; timeout_ms=0)`: Send data
- `receive(channel, Type; timeout_ms=0)`: Receive typed data
- `receive(channel; timeout_ms=0)`: Receive with auto-detection

#### Metrics and Monitoring
- `enable_metrics!(channel, enable)`: Enable/disable metrics
- `get_metrics(channel)`: Get current metrics
- `reset_metrics!(channel)`: Reset metrics counters
- `throughput(metrics, duration)`: Calculate throughput
- `efficiency(metrics)`: Calculate efficiency metrics
- `health_check(channel)`: Perform health check
- `monitor(channel, duration; interval)`: Monitor over time

#### Compression
- `lz4_config(; kwargs...)`: Create LZ4 configuration
- `zstd_config(; kwargs...)`: Create Zstd configuration
- `snappy_config(; kwargs...)`: Create Snappy configuration
- `estimate_compression_ratio(data, config)`: Estimate compression
- `recommend_compression(data; priority)`: Get recommendations

## Examples

The `examples/` directory contains comprehensive examples:

- `basic_example.jl`: Fundamental usage patterns
- `performance_example.jl`: Performance monitoring and optimization
- `network_example.jl`: TCP client/server communication
- `producer_consumer_example.jl`: Multi-threaded patterns
- `array_operations_example.jl`: Scientific computing workflows

Run examples:
```bash
julia examples/basic_example.jl
julia -t 4 examples/producer_consumer_example.jl
julia examples/network_example.jl server  # Terminal 1
julia examples/network_example.jl client  # Terminal 2
```

## Testing

Run the test suite:
```bash
julia --project=. test/runtests.jl
# Or
julia --project=. -e "using Pkg; Pkg.test()"
```

## Performance Guidelines

### Channel Mode Selection
- **SPSC**: Fastest for single producer/consumer (lock-free)
- **MPSC**: Multiple producers, single consumer
- **SPMC**: Single producer, multiple consumers  
- **MPMC**: Full multi-threading support (highest overhead)

### Buffer Sizing
- Start with 1-4MB for most applications
- Larger buffers reduce blocking but use more memory
- Size based on message frequency and size

### Compression Guidelines
- **LZ4**: Fast compression/decompression, moderate ratios
- **Zstd**: Better compression ratios, slower
- **Snappy**: Balanced speed/ratio
- Use for network channels or when bandwidth is limited
- Set minimum size thresholds to avoid compressing small messages

### Memory Management
- Always close channels when done
- Use metrics to monitor performance
- Enable health checks in production
- Consider using `@async` for non-blocking operations

## Error Handling

```julia
try
    data = receive(ch, Vector{Float32}, timeout_ms=1000)
catch e
    if isa(e, PsyneError)
        if e.code == PSYNE_ERROR_TIMEOUT
            println("Receive timed out")
        elseif e.code == PSYNE_ERROR_NO_MESSAGE
            println("No message available")
        else
            println("Psyne error: $(e.message)")
        end
    else
        println("Unexpected error: $e")
        rethrow(e)
    end
end
```

## Troubleshooting

### Common Issues

1. **"Cannot find libpsyne"**
   ```bash
   # Linux
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   
   # macOS
   export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
   ```

2. **"Channel creation failed"**
   - Check URI format: `"memory://name"`, `"tcp://host:port"`
   - Ensure port is available for TCP channels
   - Verify sufficient memory for buffer size

3. **Poor performance**
   - Use SPSC mode when possible
   - Increase buffer size for large/frequent messages
   - Enable compression for network channels
   - Monitor metrics to identify bottlenecks

4. **Memory issues**
   - Close channels explicitly
   - Monitor channel health
   - Use appropriate buffer sizes

### Debug Information

```julia
# Enable verbose error reporting
ENV["JULIA_DEBUG"] = "Psyne"

# Check channel health
health = health_check(ch)
if health.status != :healthy
    println("Issues: ", health.issues)
    println("Recommendations: ", health.recommendations)
end

# Monitor metrics
snapshots = monitor(ch, 10.0, interval_seconds=1.0)
for snapshot in snapshots
    println("Metrics: ", snapshot)
end
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/your-repo/psyne.git
cd psyne/bindings/julia
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. test/runtests.jl
```

## License

This project is licensed under the same terms as the main Psyne library.

## Support

- Documentation: See `examples/` and `test/` directories
- Issues: Create GitHub issues for bugs or feature requests
- Performance: Use metrics and health checks for debugging

## Version History

- **1.2.0**: Current release
  - Core channel operations
  - Message types with Julia idioms
  - Compression support
  - Performance monitoring
  - Comprehensive examples and tests