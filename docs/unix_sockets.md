# Unix Domain Sockets in Psyne

Unix domain sockets provide high-performance local IPC on Unix-like systems (Linux, macOS, BSD). They offer lower latency and higher throughput than TCP sockets for local communication.

## Usage

### Creating a Server

To create a Unix domain socket server, use the `unix://` URI scheme with an `@` prefix:

```cpp
// Server listens on /tmp/myapp.sock
auto server = psyne::create_channel("unix://@/tmp/myapp.sock");

// Wait for messages
auto msg = server->receive<FloatVector>(std::chrono::seconds(5));
```

### Creating a Client

To connect to a Unix domain socket server, use the `unix://` URI scheme without the `@` prefix:

```cpp
// Client connects to /tmp/myapp.sock
auto client = psyne::create_channel("unix:///tmp/myapp.sock");

// Send a message
FloatVector msg(*client);
msg.resize(100);
// ... fill data ...
msg.send();
```

## URI Format

- **Server (listening)**: `unix://@/path/to/socket` or `unix://@socket_name`
- **Client (connecting)**: `unix:///path/to/socket` or `unix://socket_name`

### Path Resolution

- Absolute paths (starting with `/`) are used as-is
- Relative paths are resolved relative to the current working directory
- The `@` prefix indicates server mode (creates and listens on the socket)

## Features

- **Zero-copy messaging**: Same efficient memory management as other Psyne channels
- **Message framing**: Automatic message framing with checksums
- **All channel modes**: Supports SPSC, SPMC, MPSC, and MPMC modes
- **Reliability features**: Works with acknowledgments, retries, and heartbeats
- **Cross-platform**: Works on Linux, macOS, and other Unix-like systems

## Performance

Unix domain sockets typically provide:
- **Lower latency** than TCP for local communication (often 2-3x faster)
- **Higher throughput** due to no network stack overhead
- **Lower CPU usage** compared to TCP sockets

## Example

See `examples/unix_socket_demo.cpp` for a complete working example:

```bash
# Terminal 1 - Start server
./build/examples/unix_socket_demo server

# Terminal 2 - Start client
./build/examples/unix_socket_demo client
```

## Best Practices

1. **Socket file cleanup**: The server automatically removes existing socket files on startup
2. **Permissions**: Socket files inherit directory permissions; place in appropriate locations
3. **Path length**: Keep socket paths under 108 characters (Unix limitation)
4. **Error handling**: Always check for connection errors, especially on client startup

## Comparison with Other Transports

| Transport | Use Case | Latency | Throughput |
|-----------|----------|---------|------------|
| memory:// | Same process | Lowest | Highest |
| unix://   | Same machine | Very Low | Very High |
| ipc://    | Same machine | Low | High |
| tcp://    | Network/Local | Higher | Lower |

## Security Considerations

- Unix domain sockets use file system permissions for access control
- Only processes with appropriate file permissions can connect
- No network exposure - purely local communication
- Consider socket file location carefully in multi-user systems