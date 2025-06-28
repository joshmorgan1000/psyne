package com.psyne.examples;

import com.psyne.*;
import java.nio.charset.StandardCharsets;
import java.util.Random;

/**
 * Example demonstrating message compression with Psyne.
 */
public class CompressionExample {
    
    public static void main(String[] args) {
        try {
            // Initialize the Psyne library
            Psyne.init();
            
            // Configure compression
            CompressionConfig compression = CompressionConfig.builder()
                    .type(CompressionType.LZ4)
                    .level(9) // Medium compression
                    .minSizeThreshold(100) // Only compress messages > 100 bytes
                    .enableChecksum(true)
                    .build();
            
            // Create a channel with compression
            try (Channel channel = Channel.builder()
                    .uri("memory://compression-example")
                    .bufferSize(4 * 1024 * 1024) // 4MB
                    .mode(ChannelMode.SPSC)
                    .compression(compression)
                    .build()) {
                
                System.out.println("Created compressed channel: " + channel.getUri());
                
                // Enable metrics to see compression effects
                channel.enableMetrics(true);
                
                // Send various sized messages
                sendCompressibleData(channel, 50);    // Won't be compressed (below threshold)
                sendCompressibleData(channel, 1000);  // Will be compressed
                sendCompressibleData(channel, 10000); // Will be compressed
                sendRandomData(channel, 1000);        // Random data (poor compression)
                
                // Receive all messages
                System.out.println("\nReceiving messages...");
                Channel.ReceivedMessage received;
                int count = 0;
                while ((received = channel.receive()) != null) {
                    try (Message msg = received.getMessage()) {
                        System.out.println("Received message " + (++count) + 
                                         ", size: " + msg.getSize() + " bytes");
                    }
                }
                
                // Display compression metrics
                Metrics metrics = channel.getMetrics();
                System.out.println("\nCompression metrics:");
                System.out.println("  Total messages: " + metrics.getMessagesSent());
                System.out.println("  Total bytes sent: " + metrics.getBytesSent());
                System.out.println("  Average message size: " + metrics.getAverageSentMessageSize() + " bytes");
                
                // Note: Actual compression ratio would require comparing with uncompressed size
                // which would need to be tracked separately
            }
            
        } catch (PsyneException e) {
            System.err.println("Psyne error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void sendCompressibleData(Channel channel, int size) throws PsyneException {
        // Create highly compressible data (repeated pattern)
        StringBuilder sb = new StringBuilder(size);
        String pattern = "ABCDEFGHIJ";
        while (sb.length() < size) {
            sb.append(pattern);
        }
        sb.setLength(size);
        
        byte[] data = sb.toString().getBytes(StandardCharsets.UTF_8);
        channel.send(data, 1);
        System.out.println("Sent compressible data: " + size + " bytes");
    }
    
    private static void sendRandomData(Channel channel, int size) throws PsyneException {
        // Create random data (poor compression)
        Random random = new Random();
        byte[] data = new byte[size];
        random.nextBytes(data);
        
        channel.send(data, 2);
        System.out.println("Sent random data: " + size + " bytes");
    }
}