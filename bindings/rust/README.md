# Psyne Rust Bindings

<div align="center">
  <img src="../../docs/assets/psyne_logo.png" alt="Psyne Logo" width="120"/>
  
  **Safe Rust bindings for Psyne high-performance messaging**
  
  [![Crates.io](https://img.shields.io/badge/crates.io-psyne-orange.svg)](https://crates.io/crates/psyne)
  [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.rs/psyne)
  [![Rust Version](https://img.shields.io/badge/rust-1.70+-blue.svg)](https://www.rust-lang.org)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)
</div>

## 📋 Table of Contents

- [🌟 Features](#-features)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📚 Examples](#-examples)
  - [Basic Usage](#basic-usage)
  - [Builder Pattern](#builder-pattern)
  - [Async/Await](#asyncawait)
  - [Multi-threading](#multi-threading)
  - [Compression](#compression)
- [🔧 API Reference](#-api-reference)
- [⚡ Performance](#-performance)
- [🐛 Error Handling](#-error-handling)
- [🔍 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)

## 🌟 Features

- **🦀 Safe Rust API** - Zero-cost abstractions over the C API
- **⚡ Zero-Copy Performance** - Direct access to message buffers
- **🔒 Memory Safety** - Leverages Rust's ownership system
- **🚀 Async/Await Support** - First-class async support with Tokio
- **🔧 Builder Pattern** - Idiomatic Rust channel configuration
- **📊 Type Safety** - Compile-time guarantees for message types
- **🌐 All Transports** - Memory, IPC, TCP, Unix sockets, WebSocket, RDMA
- **🗜️ Compression** - LZ4, Zstd, and Snappy support
- **📈 Performance Monitoring** - Built-in metrics collection

## 📦 Installation

Add Psyne to your `Cargo.toml`:

```toml
[dependencies]
psyne = "1.0.0"
tokio = { version = "1.35", features = ["full"] }  # For async support
```

### System Requirements

- **Rust 1.70+** (2021 edition)
- **libpsyne** system library
- **pkg-config** for linking

### Installing System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install libpsyne-dev pkg-config
```

**macOS (Homebrew):**
```bash
brew install psyne pkg-config
```

**From Source:**
```bash
git clone https://github.com/joshmorgan1000/psyne.git
cd psyne
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

## 🚀 Quick Start

```rust
use psyne::{Channel, ChannelMode, ChannelType};

fn main() -> Result<(), psyne::Error> {
    // Initialize the library
    psyne::init()?;

    // Create a high-performance channel
    let channel = Channel::create(
        "memory://demo",
        1024 * 1024,  // 1MB buffer
        ChannelMode::Spsc,
        ChannelType::Multi,
    )?;

    // Send a message
    let data = b"Hello from Rust!";
    channel.send_data(data, 1)?;

    // Receive the message
    let mut buffer = vec![0u8; 1024];
    if let Some((size, msg_type)) = channel.receive_data(&mut buffer, 0)? {
        let message = std::str::from_utf8(&buffer[..size])?;
        println!("Received: {} (type: {})", message, msg_type);
    }

    Ok(())
}
```

## 📚 Examples

### Basic Usage

```rust
use psyne::{Channel, ChannelMode, ChannelType};

fn basic_example() -> Result<(), psyne::Error> {
    psyne::init()?;
    
    let channel = Channel::create(
        "memory://basic", 
        2 * 1024 * 1024, 
        ChannelMode::Spsc, 
        ChannelType::Multi
    )?;
    
    // Send structured data
    let data = serde_json::json!({
        "user_id": 12345,
        "action": "purchase",
        "amount": 99.99
    });
    
    let serialized = serde_json::to_vec(&data)?;
    channel.send_data(&serialized, 100)?;
    
    // Receive and deserialize
    let mut buffer = vec![0u8; 4096];
    if let Some((size, _)) = channel.receive_data(&mut buffer, 1000)? {
        let received: serde_json::Value = serde_json::from_slice(&buffer[..size])?;
        println!("Received JSON: {}", received);
    }
    
    Ok(())
}
```

### Builder Pattern

```rust
use psyne::{ChannelBuilder, ChannelMode, CompressionConfig, CompressionType};

fn builder_example() -> Result<(), psyne::Error> {
    psyne::init()?;
    
    let channel = ChannelBuilder::new("tcp://localhost:8080")
        .buffer_size(10 * 1024 * 1024)  // 10MB
        .mode(ChannelMode::Mpsc)
        .compression(CompressionConfig {
            compression_type: CompressionType::Lz4,
            level: 3,
            min_size_threshold: 1024,
            enable_checksum: true,
        })
        .build()?;
    
    println!("Created channel: {}", channel.uri()?);
    
    // Use the channel...
    Ok(())
}
```

### Async/Await

```rust
use psyne::{Channel, ChannelMode, ChannelType};
use tokio::time::{timeout, Duration};

#[tokio::main]
async fn async_example() -> Result<(), Box<dyn std::error::Error>> {
    psyne::init()?;
    
    let channel = Channel::create(
        "memory://async",
        4 * 1024 * 1024,
        ChannelMode::Spsc,
        ChannelType::Multi,
    )?;
    
    // Async send with timeout
    let data = b"Async message from Rust";
    timeout(Duration::from_secs(5), async {
        channel.send_data(data, 42)
    }).await??;
    
    // Async receive with timeout
    let mut buffer = vec![0u8; 1024];
    let result = timeout(Duration::from_secs(5), async {
        loop {
            if let Some(msg) = channel.receive_data(&mut buffer, 100)? {
                return Ok::<_, psyne::Error>(msg);
            }
            tokio::task::yield_now().await;
        }
    }).await??;
    
    if let (size, msg_type) = result {
        println!("Async received: {} (type: {})", 
                 std::str::from_utf8(&buffer[..size])?, msg_type);
    }
    
    Ok(())
}
```

### Multi-threading

```rust
use psyne::{Channel, ChannelMode, ChannelType};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn multithreaded_example() -> Result<(), psyne::Error> {
    psyne::init()?;
    
    let channel = Arc::new(Channel::create(
        "memory://multithread",
        8 * 1024 * 1024,
        ChannelMode::Mpsc,  // Multiple producers, single consumer
        ChannelType::Multi,
    )?);
    
    // Spawn producer threads
    let producers: Vec<_> = (0..4).map(|i| {
        let ch = Arc::clone(&channel);
        thread::spawn(move || -> Result<(), psyne::Error> {
            for j in 0..100 {
                let msg = format!("Message {} from producer {}", j, i);
                ch.send_data(msg.as_bytes(), i as u32)?;
                thread::sleep(Duration::from_millis(10));
            }
            Ok(())
        })
    }).collect();
    
    // Consumer thread
    let consumer = {
        let ch = Arc::clone(&channel);
        thread::spawn(move || -> Result<(), psyne::Error> {
            let mut received = 0;
            let mut buffer = vec![0u8; 1024];
            
            while received < 400 {  // 4 producers × 100 messages
                if let Some((size, producer_id)) = ch.receive_data(&mut buffer, 100)? {
                    let msg = std::str::from_utf8(&buffer[..size])?;
                    println!("Received from producer {}: {}", producer_id, msg);
                    received += 1;
                }
            }
            Ok(())
        })
    };
    
    // Wait for completion
    for producer in producers {
        producer.join().unwrap()?;
    }
    consumer.join().unwrap()?;
    
    Ok(())
}
```

### Compression

```rust
use psyne::{ChannelBuilder, CompressionConfig, CompressionType};

fn compression_example() -> Result<(), psyne::Error> {
    psyne::init()?;
    
    let config = CompressionConfig {
        compression_type: CompressionType::Zstd,
        level: 5,
        min_size_threshold: 512,
        enable_checksum: true,
    };
    
    let channel = ChannelBuilder::new("memory://compressed")
        .compression(config)
        .build()?;
    
    // Send large, compressible data
    let data = "A".repeat(10_000);  // 10KB of 'A's - highly compressible
    channel.send_data(data.as_bytes(), 0)?;
    
    println!("Sent {} bytes of compressible data", data.len());
    
    // Receive (decompression is automatic)
    let mut buffer = vec![0u8; 20_000];
    if let Some((size, _)) = channel.receive_data(&mut buffer, 1000)? {
        println!("Received {} bytes after decompression", size);
        assert_eq!(size, data.len());
    }
    
    Ok(())
}
```

## 🔧 API Reference

### Core Types

```rust
// Channel creation and management
pub struct Channel { /* ... */ }
pub struct ChannelBuilder { /* ... */ }

// Configuration
pub enum ChannelMode { Spsc, Spmc, Mpsc, Mpmc }
pub enum ChannelType { Single, Multi }
pub enum CompressionType { None, Lz4, Zstd, Snappy }

// Error handling
pub enum Error { /* ... */ }
pub type Result<T> = std::result::Result<T, Error>;

// Metrics and monitoring
pub struct Metrics { /* ... */ }
```

### Channel Operations

```rust
impl Channel {
    // Creation
    pub fn create(uri: &str, buffer_size: usize, mode: ChannelMode, 
                  channel_type: ChannelType) -> Result<Self>;
    
    // Basic operations
    pub fn send_data(&self, data: &[u8], msg_type: u32) -> Result<()>;
    pub fn receive_data(&self, buffer: &mut [u8], timeout: Duration) 
                       -> Result<Option<(usize, u32)>>;
    
    // Management
    pub fn stop(&self) -> Result<()>;
    pub fn is_stopped(&self) -> Result<bool>;
    pub fn uri(&self) -> Result<String>;
    pub fn metrics(&self) -> Result<Metrics>;
    pub fn reset_metrics(&self) -> Result<()>;
}
```

### Message Operations

```rust
pub struct Message<'a> { /* ... */ }

impl<'a> Message<'a> {
    pub fn reserve(channel: &'a Channel, size: usize) -> Result<Self>;
    pub fn data_mut(&mut self) -> Result<&mut [u8]>;
    pub fn send(self, msg_type: u32) -> Result<()>;
}
```

## ⚡ Performance

### Benchmarks (Release Build, x86_64)

| Operation | Latency | Throughput | Notes |
|-----------|---------|------------|-------|
| Memory Channel (SPSC) | ~80 ns | 1.2M msg/s | 64-byte messages |
| Memory Channel (1KB) | ~200 ns | 800 MB/s | With compression off |
| TCP Channel (localhost) | ~15 μs | 650 MB/s | 1KB messages |
| Unix Socket | ~8 μs | 950 MB/s | Local IPC |

### Performance Tips

1. **Use SPSC mode** when possible for maximum throughput
2. **Pre-allocate buffers** to avoid runtime allocations
3. **Enable compression** for network channels with large messages
4. **Use manual message reservation** for zero-copy workflows
5. **Pin threads to cores** for consistent latency

```rust
// High-performance example
use psyne::{Channel, ChannelMode, ChannelType};

fn high_performance_setup() -> Result<Channel, psyne::Error> {
    let channel = Channel::create(
        "memory://high_perf",
        16 * 1024 * 1024,  // Large buffer
        ChannelMode::Spsc,  // Single producer/consumer
        ChannelType::Single, // Single message type
    )?;
    
    // Optionally set thread affinity
    // core_affinity::set_for_current(core_affinity::CoreId { id: 0 });
    
    Ok(channel)
}
```

## 🐛 Error Handling

Psyne uses Rust's `Result` type for comprehensive error handling:

```rust
use psyne::Error;

fn error_handling_example() {
    match psyne::init() {
        Ok(()) => println!("Psyne initialized successfully"),
        Err(Error::OutOfMemory) => eprintln!("Not enough memory"),
        Err(Error::InvalidArgument) => eprintln!("Invalid configuration"),
        Err(Error::IoError) => eprintln!("Network or file system error"),
        Err(e) => eprintln!("Other error: {}", e),
    }
}

// Convert to other error types
fn with_anyhow() -> anyhow::Result<()> {
    psyne::init()?;  // Automatic conversion
    Ok(())
}
```

### Error Types

```rust
pub enum Error {
    InvalidArgument,
    OutOfMemory,
    ChannelFull,
    NoMessage,
    ChannelStopped,
    Unsupported,
    IoError,
    Timeout,
    Unknown,
    InvalidUtf8(std::str::Utf8Error),
    NullPointer,
}
```

## 🔍 Troubleshooting

### Common Issues

**Q: "libpsyne not found" linking error**
```bash
# Install development libraries
sudo apt install libpsyne-dev pkg-config

# Or set PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

**Q: Compilation fails with "bindgen not found"**
```bash
cargo install bindgen-cli
```

**Q: Runtime panic on channel creation**
```rust
// Always call init() first
psyne::init()?;

// Check URI format
let uri = "memory://valid_name";  // ✓ Good
let uri = "invalid_uri";          // ✗ Bad
```

**Q: Messages not received**
```rust
// Check channel modes match between producer and consumer
// Use appropriate timeout values
let result = channel.receive_data(&mut buffer, 1000)?; // 1 second timeout
```

### Debug Mode

```toml
[dependencies]
psyne = { version = "1.0.0", features = ["debug"] }
```

```rust
// Enable debug logging
psyne::set_log_level(psyne::LogLevel::Debug);
```

### Performance Debugging

```rust
// Monitor channel performance
let metrics = channel.metrics()?;
println!("Messages sent: {}, received: {}", 
         metrics.messages_sent, metrics.messages_received);
println!("Send blocks: {}, receive blocks: {}", 
         metrics.send_blocks, metrics.receive_blocks);
```

## 🤝 Contributing

We welcome contributions to the Rust bindings! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

### Rust-Specific Guidelines

- **Follow Rust conventions** - Use `cargo fmt` and `cargo clippy`
- **Add tests** for new functionality in `tests/`
- **Update documentation** - Keep this README and rustdoc comments current
- **Consider safety** - Unsafe code requires careful review and documentation

### Development Commands

```bash
# Format code
cargo fmt

# Run lints
cargo clippy -- -D warnings

# Run tests
cargo test

# Generate documentation
cargo doc --open

# Run examples
cargo run --example basic
```

---

## 📚 Additional Resources

- **[Main Psyne Documentation](../../docs/)** - Complete project documentation
- **[API Documentation](https://docs.rs/psyne)** - Rust-specific API docs
- **[Examples Directory](examples/)** - More code examples
- **[Performance Guide](../../docs/performance.md)** - Optimization techniques
- **[GitHub Repository](https://github.com/joshmorgan1000/psyne)** - Source code and issues

---

<div align="center">
  
**Built with ❤️ for high-performance Rust applications**

[🏠 Main Documentation](../../README.md) • [🐛 Report Issues](https://github.com/joshmorgan1000/psyne/issues) • [💬 Discussions](https://github.com/joshmorgan1000/psyne/discussions)

</div>