package com.psyne.examples;

import com.psyne.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

/**
 * Example demonstrating the builder pattern and various channel configurations.
 */
public class BuilderPatternExample {
    
    public static void main(String[] args) {
        try {
            // Initialize the Psyne library
            Psyne.init();
            
            // Example 1: Basic channel with defaults
            basicChannel();
            
            // Example 2: Channel with custom configuration
            customChannel();
            
            // Example 3: Compressed channel with all options
            compressedChannel();
            
            // Example 4: Zero-copy message handling
            zeroCopyExample();
            
        } catch (PsyneException e) {
            System.err.println("Psyne error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void basicChannel() throws PsyneException {
        System.out.println("=== Basic Channel Example ===");
        
        // Minimal configuration - only URI is required
        try (Channel channel = Channel.builder()
                .uri("memory://basic")
                .build()) {
            
            System.out.println("Created channel with defaults:");
            System.out.println("  URI: " + channel.getUri());
            System.out.println("  Buffer size: " + channel.getBufferSize() + " bytes");
            
            // Simple send/receive
            channel.send("Hello, World!".getBytes(StandardCharsets.UTF_8), 1);
            
            Channel.ReceivedMessage msg = channel.receive();
            if (msg != null) {
                try (Message message = msg.getMessage()) {
                    System.out.println("  Received: " + new String(message.toByteArray()));
                }
            }
        }
        System.out.println();
    }
    
    private static void customChannel() throws PsyneException {
        System.out.println("=== Custom Channel Example ===");
        
        // Channel with all custom options
        try (Channel channel = Channel.builder()
                .uri("memory://custom")
                .bufferSize(4 * 1024 * 1024) // 4MB buffer
                .mode(ChannelMode.MPMC)      // Multi-producer, multi-consumer
                .type(ChannelType.MULTI)      // Support multiple message types
                .build()) {
            
            System.out.println("Created custom channel:");
            System.out.println("  URI: " + channel.getUri());
            System.out.println("  Buffer size: " + channel.getBufferSize() + " bytes");
            
            // Enable metrics
            channel.enableMetrics(true);
            
            // Send different message types
            channel.send("Text message".getBytes(), 1);
            channel.send("{\"type\":\"json\"}".getBytes(), 2);
            channel.send("<xml/>".getBytes(), 3);
            
            // Receive with timeout
            for (int i = 0; i < 3; i++) {
                Channel.ReceivedMessage msg = channel.receive(100, TimeUnit.MILLISECONDS);
                if (msg != null) {
                    try (Message message = msg.getMessage()) {
                        System.out.println("  Received type " + msg.getType() + 
                                         ": " + new String(message.toByteArray()));
                    }
                }
            }
            
            // Show metrics
            Metrics metrics = channel.getMetrics();
            System.out.println("  Messages sent/received: " + 
                             metrics.getMessagesSent() + "/" + metrics.getMessagesReceived());
        }
        System.out.println();
    }
    
    private static void compressedChannel() throws PsyneException {
        System.out.println("=== Compressed Channel Example ===");
        
        // Build compression configuration
        CompressionConfig compression = CompressionConfig.builder()
                .type(CompressionType.ZSTD)
                .level(6) // Medium compression level
                .minSizeThreshold(512) // Only compress messages > 512 bytes
                .enableChecksum(true)
                .build();
        
        // Create channel with compression
        try (Channel channel = Channel.builder()
                .uri("memory://compressed")
                .bufferSize(2 * 1024 * 1024)
                .mode(ChannelMode.SPSC)
                .compression(compression)
                .build()) {
            
            System.out.println("Created compressed channel with ZSTD");
            
            // Send a large, compressible message
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 100; i++) {
                sb.append("This is a highly compressible line of text. ");
            }
            String largeMessage = sb.toString();
            
            System.out.println("  Original message size: " + largeMessage.length() + " bytes");
            channel.send(largeMessage.getBytes(StandardCharsets.UTF_8), 1);
            
            // Receive and check size
            Channel.ReceivedMessage msg = channel.receive();
            if (msg != null) {
                try (Message message = msg.getMessage()) {
                    System.out.println("  Received message size: " + message.getSize() + " bytes");
                    System.out.println("  Content matches: " + 
                                     largeMessage.equals(new String(message.toByteArray())));
                }
            }
        }
        System.out.println();
    }
    
    private static void zeroCopyExample() throws PsyneException {
        System.out.println("=== Zero-Copy Example ===");
        
        try (Channel channel = Channel.builder()
                .uri("memory://zerocopy")
                .bufferSize(1024 * 1024)
                .mode(ChannelMode.SPSC)
                .build()) {
            
            // Reserve space for a message
            try (Message message = channel.reserve(1024)) {
                // Get direct ByteBuffer (zero-copy access to native memory)
                ByteBuffer buffer = message.getData();
                
                // Write directly to the buffer
                buffer.put("Direct buffer write: ".getBytes(StandardCharsets.UTF_8));
                
                // Write some binary data
                buffer.putInt(42);
                buffer.putDouble(3.14159);
                buffer.putLong(System.currentTimeMillis());
                
                // Record how much we wrote
                int bytesWritten = buffer.position();
                
                // Send the message
                message.send(10);
                System.out.println("  Sent " + bytesWritten + " bytes using zero-copy");
            }
            
            // Receive and read directly from buffer
            Channel.ReceivedMessage received = channel.receive();
            if (received != null) {
                try (Message msg = received.getMessage()) {
                    ByteBuffer buffer = msg.getData();
                    
                    // Read the string part
                    byte[] stringBytes = new byte[21]; // "Direct buffer write: "
                    buffer.get(stringBytes);
                    System.out.println("  String: " + new String(stringBytes));
                    
                    // Read binary data
                    System.out.println("  Int: " + buffer.getInt());
                    System.out.println("  Double: " + buffer.getDouble());
                    System.out.println("  Timestamp: " + buffer.getLong());
                }
            }
        }
    }
}