package com.psyne.examples;

import com.psyne.*;
import java.nio.charset.StandardCharsets;

/**
 * Basic example demonstrating simple message sending and receiving with Psyne.
 */
public class BasicExample {
    
    public static void main(String[] args) {
        try {
            // Initialize the Psyne library
            Psyne.init();
            System.out.println("Psyne version: " + Psyne.getVersion());
            
            // Create a memory channel
            try (Channel channel = Channel.builder()
                    .uri("memory://basic-example")
                    .bufferSize(1024 * 1024) // 1MB
                    .mode(ChannelMode.SPSC)
                    .build()) {
                
                System.out.println("Created channel: " + channel.getUri());
                
                // Send some messages
                for (int i = 0; i < 5; i++) {
                    String message = "Hello, Psyne! Message #" + i;
                    channel.send(message.getBytes(StandardCharsets.UTF_8), 1);
                    System.out.println("Sent: " + message);
                }
                
                // Receive messages
                System.out.println("\nReceiving messages...");
                Channel.ReceivedMessage received;
                while ((received = channel.receive()) != null) {
                    try (Message msg = received.getMessage()) {
                        byte[] data = msg.toByteArray();
                        String content = new String(data, StandardCharsets.UTF_8);
                        System.out.println("Received (type=" + received.getType() + "): " + content);
                    }
                }
                
                // Display metrics
                Metrics metrics = channel.getMetrics();
                System.out.println("\nChannel metrics:");
                System.out.println("  Messages sent: " + metrics.getMessagesSent());
                System.out.println("  Bytes sent: " + metrics.getBytesSent());
                System.out.println("  Average message size: " + metrics.getAverageSentMessageSize() + " bytes");
            }
            
        } catch (PsyneException e) {
            System.err.println("Psyne error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Cleanup is automatic via shutdown hook, but we can call it explicitly
            Psyne.cleanup();
        }
    }
}