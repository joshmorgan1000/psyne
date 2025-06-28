use psyne::{Channel, ChannelBuilder, ChannelMode, ChannelType, CompressionConfig, CompressionType};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Psyne
    psyne::init()?;
    println!("Psyne version: {}", psyne::version());

    // Example 1: Basic channel usage
    basic_example()?;
    
    // Example 2: Builder pattern
    builder_example()?;
    
    // Example 3: Compression
    compression_example()?;
    
    // Example 4: Multi-threaded
    multithreaded_example()?;
    
    // Example 5: Manual messages
    manual_message_example()?;

    // Cleanup
    psyne::cleanup();
    Ok(())
}

fn basic_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Basic Example ===");
    
    let channel = Channel::create(
        "memory://basic",
        1024 * 1024,
        ChannelMode::Spsc,
        ChannelType::Multi,
    )?;
    
    // Send a message
    let message = "Hello from Rust!";
    channel.send_data(message.as_bytes(), 1)?;
    println!("Sent: {}", message);
    
    // Receive the message
    let mut buffer = vec![0u8; 1024];
    if let Some((size, msg_type)) = channel.receive_data(&mut buffer, Duration::from_secs(1))? {
        let received = std::str::from_utf8(&buffer[..size])?;
        println!("Received: {} (type: {})", received, msg_type);
    }
    
    // Check metrics
    let metrics = channel.metrics()?;
    println!("Metrics: {:?}", metrics);
    
    Ok(())
}

fn builder_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Builder Pattern Example ===");
    
    let channel = ChannelBuilder::new("memory://builder")
        .buffer_size(2 * 1024 * 1024)  // 2MB
        .mode(ChannelMode::Mpsc)
        .channel_type(ChannelType::Multi)
        .build()?;
    
    println!("Created channel: {}", channel.uri()?);
    
    // Send from multiple threads
    let handles: Vec<_> = (0..3)
        .map(|i| {
            let ch = &channel;
            thread::spawn(move || {
                let msg = format!("Message from thread {}", i);
                ch.send_data(msg.as_bytes(), i as u32).unwrap();
                println!("Thread {} sent message", i);
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Receive all messages
    let mut buffer = vec![0u8; 1024];
    for _ in 0..3 {
        if let Some((size, msg_type)) = channel.receive_data(&mut buffer, Duration::from_secs(1))? {
            let msg = std::str::from_utf8(&buffer[..size])?;
            println!("Received: {} (type: {})", msg, msg_type);
        }
    }
    
    Ok(())
}

fn compression_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Compression Example ===");
    
    let compression = CompressionConfig {
        compression_type: CompressionType::Lz4,
        level: 1,
        min_size_threshold: 100,
        enable_checksum: true,
    };
    
    let channel = ChannelBuilder::new("memory://compressed")
        .compression(compression)
        .build()?;
    
    // Send a large, compressible message
    let data = "A".repeat(10000);  // 10KB of 'A's
    channel.send_data(data.as_bytes(), 0)?;
    println!("Sent {} bytes of compressible data", data.len());
    
    // Receive
    let mut buffer = vec![0u8; 20000];
    if let Some((size, _)) = channel.receive_data(&mut buffer, Duration::from_secs(1))? {
        println!("Received {} bytes", size);
        assert_eq!(size, data.len());
        assert_eq!(&buffer[..100], &data.as_bytes()[..100]);
    }
    
    Ok(())
}

fn multithreaded_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Multi-threaded Example ===");
    
    let channel = Channel::create(
        "memory://multithread",
        4 * 1024 * 1024,
        ChannelMode::Spsc,
        ChannelType::Multi,
    )?;
    
    // Producer thread
    let producer = {
        let ch = channel.clone();
        thread::spawn(move || {
            for i in 0..10 {
                let msg = format!("Message {}", i);
                ch.send_data(msg.as_bytes(), i).unwrap();
                println!("Producer sent: {}", msg);
                thread::sleep(Duration::from_millis(100));
            }
        })
    };
    
    // Consumer thread
    let consumer = {
        let ch = channel;
        thread::spawn(move || {
            let mut buffer = vec![0u8; 1024];
            let mut received = 0;
            
            while received < 10 {
                if let Ok(Some((size, msg_type))) = 
                    ch.receive_data(&mut buffer, Duration::from_millis(200)) {
                    let msg = std::str::from_utf8(&buffer[..size]).unwrap();
                    println!("Consumer received: {} (type: {})", msg, msg_type);
                    received += 1;
                }
            }
        })
    };
    
    producer.join().unwrap();
    consumer.join().unwrap();
    
    Ok(())
}

fn manual_message_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Manual Message Example ===");
    
    let channel = Channel::create(
        "memory://manual",
        1024 * 1024,
        ChannelMode::Spsc,
        ChannelType::Multi,
    )?;
    
    // Reserve and send a message
    {
        let mut msg = psyne::Message::reserve(&channel, 100)?;
        let data = msg.data_mut()?;
        
        // Write directly to message buffer
        let text = b"Direct buffer write from Rust!";
        data[..text.len()].copy_from_slice(text);
        
        // Send with type 99
        msg.send(99)?;
        println!("Sent manual message");
    }
    
    // Receive
    let mut buffer = vec![0u8; 1024];
    if let Some((size, msg_type)) = channel.receive_data(&mut buffer, Duration::from_secs(1))? {
        let msg = std::str::from_utf8(&buffer[..size])?;
        println!("Received manual message: {} (type: {})", msg, msg_type);
    }
    
    Ok(())
}

// Implement Clone for Channel to allow sharing between threads
impl Clone for Channel {
    fn clone(&self) -> Self {
        // This is safe because the C API manages reference counting internally
        // In a real implementation, we'd need to increment a reference count
        Self {
            ptr: self.ptr,
            _phantom: PhantomData,
        }
    }
}