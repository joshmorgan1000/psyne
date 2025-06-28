# Psyne Python Bindings

This directory contains Python bindings for the Psyne high-performance messaging library, built using [pybind11](https://pybind11.readthedocs.io/).

## Features

- **Zero-copy messaging**: Efficient data transfer between Python and C++
- **NumPy integration**: Seamless conversion between NumPy arrays and Psyne messages
- **Multiple transports**: IPC, TCP, Unix sockets, and UDP multicast
- **Compression support**: LZ4, Zstd, and Snappy compression algorithms
- **Metrics and monitoring**: Built-in performance metrics and debugging utilities
- **Thread-safe**: Support for different channel synchronization modes

## Installation

### Prerequisites

1. **Python 3.7+** with development headers
2. **NumPy** (`pip install numpy`)
3. **pybind11** (`pip install pybind11`)
4. **Psyne C++ library** (built in the parent directory)

### Building

#### Option 1: Using setuptools (recommended)

```bash
cd python
python setup.py build_ext --inplace
```

#### Option 2: Using CMake

```bash
# From the Psyne root directory
mkdir -p build
cd build
cmake .. -DPSYNE_PYTHON_BINDINGS=ON
make psyne_python
```

### Installation

```bash
cd python
pip install .
```

Or for development:

```bash
pip install -e .
```

## Quick Start

```python
import psyne
import numpy as np

# Create an IPC channel
channel = psyne.create_ipc_channel("my_channel")

# Create and send a NumPy array
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
msg = channel.create_float_vector(len(data))
msg.from_numpy(data)
msg.send()

# Receive data as NumPy array
received = channel.receive_float_vector()
if received is not None:
    print(f"Received: {received}")

# Clean up
channel.stop()
```

## API Reference

### Channel Creation

#### IPC Channels
```python
channel = psyne.create_ipc_channel(name, buffer_size=1024*1024, mode=psyne.ChannelMode.SPSC)
```

#### TCP Channels
```python
# Server
server = psyne.create_tcp_server(port, buffer_size=1024*1024)

# Client  
client = psyne.create_tcp_client(host, port, buffer_size=1024*1024)
```

#### Unix Domain Sockets
```python
channel = psyne.create_unix_channel(path, role, buffer_size=1024*1024)
# role: psyne.UnixSocketRole.Server or psyne.UnixSocketRole.Client
```

#### UDP Multicast
```python
# Publisher
publisher = psyne.create_multicast_publisher(address, port, buffer_size=1024*1024, compression_config)

# Subscriber
subscriber = psyne.create_multicast_subscriber(address, port, buffer_size=1024*1024, interface_address="")
```

### Message Types

#### FloatVector
```python
# Create message
msg = channel.create_float_vector(size=1024)

# From NumPy array
msg.from_numpy(numpy_array)

# To NumPy array
array = msg.to_numpy()

# Send message
msg.send()

# Access elements
msg[0] = 1.0
value = msg[0]
length = len(msg)
```

### Compression

```python
# Configure compression
config = psyne.CompressionConfig()
config.type = psyne.CompressionType.LZ4
config.level = 1
config.min_size_threshold = 100
config.enabled = True

# Use with channels that support it
publisher = psyne.create_multicast_publisher("239.255.0.1", 8080, compression_config=config)
```

### Metrics

```python
# Get channel metrics
metrics = channel.get_metrics()
print(f"Messages sent: {metrics.messages_sent}")
print(f"Bytes sent: {metrics.bytes_sent}")
print(f"Messages received: {metrics.messages_received}")

# Reset metrics
channel.reset_metrics()
```

### Channel Management

```python
# Get channel information
uri = channel.uri()
is_stopped = channel.is_stopped()

# Stop channel
channel.stop()
```

## Examples

### Basic IPC Communication

```python
import psyne
import numpy as np

# Create channel
channel = psyne.create_ipc_channel("example")

# Send data
data = np.random.randn(1000).astype(np.float32)
msg = channel.create_float_vector(len(data))
msg.from_numpy(data)
msg.send()

# Receive data
received = channel.receive_float_vector()
print(f"Received {len(received)} values")

channel.stop()
```

### TCP Client-Server

```python
import psyne
import numpy as np
import threading
import time

def server():
    server_ch = psyne.create_tcp_server(8888)
    
    # Wait for data
    data = server_ch.receive_float_vector()
    if data is not None:
        print(f"Server received: {data}")
    
    server_ch.stop()

def client():
    time.sleep(0.1)  # Let server start
    client_ch = psyne.create_tcp_client("localhost", 8888)
    
    # Send data
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    msg = client_ch.create_float_vector(len(data))
    msg.from_numpy(data)
    msg.send()
    
    client_ch.stop()

# Run server and client
threading.Thread(target=server).start()
threading.Thread(target=client).start()
```

### Multicast with Compression

```python
import psyne
import numpy as np

# Configure compression
config = psyne.CompressionConfig()
config.type = psyne.CompressionType.LZ4
config.enabled = True

# Create publisher
publisher = psyne.create_multicast_publisher("239.255.0.1", 9999, compression_config=config)

# Create subscriber
subscriber = psyne.create_multicast_subscriber("239.255.0.1", 9999)

# Send large compressible data
data = np.ones(10000, dtype=np.float32) * 42.0  # Highly compressible
msg = publisher.create_float_vector(len(data))
msg.from_numpy(data)
msg.send()

# Receive data
received = subscriber.receive_float_vector()
if received is not None:
    print(f"Received compressed data: {len(received)} values")

publisher.stop()
subscriber.stop()
```

## Performance Tips

1. **Use appropriate buffer sizes**: Larger buffers reduce system calls but use more memory
2. **Choose the right channel mode**: SPSC is fastest for single producer/consumer scenarios
3. **Enable compression for large, repetitive data**: Can significantly reduce network traffic
4. **Reuse FloatVector objects**: Avoid creating new messages for each send when possible
5. **Use NumPy dtypes correctly**: Ensure your arrays are `float32` for optimal performance

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure the C++ library is built and the Python bindings are compiled
2. **Channel Creation Fails**: Check that ports aren't already in use and you have proper permissions
3. **No Data Received**: Ensure sender and receiver are using compatible configurations
4. **Memory Issues**: Use appropriate buffer sizes and stop channels when done

### Debugging

```python
# Enable metrics to track performance
metrics = channel.get_metrics()
print(f"Send blocks: {metrics.send_blocks}")
print(f"Receive blocks: {metrics.receive_blocks}")

# Check channel status
print(f"Channel URI: {channel.uri()}")
print(f"Is stopped: {channel.is_stopped()}")
```

## Contributing

When contributing to the Python bindings:

1. Follow the existing code style and patterns
2. Add appropriate docstrings and type hints
3. Include tests for new functionality
4. Update this README for new features
5. Ensure compatibility with NumPy and common Python patterns

## License

The Python bindings follow the same license as the main Psyne library.