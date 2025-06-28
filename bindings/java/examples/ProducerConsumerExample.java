package com.psyne.examples;

import com.psyne.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Example demonstrating producer-consumer pattern with Psyne channels.
 */
public class ProducerConsumerExample {
    
    private static final int MESSAGE_COUNT = 100;
    private static final String CHANNEL_URI = "memory://producer-consumer";
    
    public static void main(String[] args) {
        try {
            // Initialize the Psyne library
            Psyne.init();
            
            // Create a channel
            try (Channel channel = Channel.builder()
                    .uri(CHANNEL_URI)
                    .bufferSize(1024 * 1024) // 1MB
                    .mode(ChannelMode.SPSC)
                    .type(ChannelType.MULTI) // Support multiple message types
                    .build()) {
                
                System.out.println("Created channel: " + channel.getUri());
                channel.enableMetrics(true);
                
                // Synchronization
                CountDownLatch startLatch = new CountDownLatch(1);
                CountDownLatch doneLatch = new CountDownLatch(2);
                AtomicBoolean stopFlag = new AtomicBoolean(false);
                
                // Start producer thread
                Thread producer = new Thread(() -> {
                    try {
                        startLatch.await();
                        runProducer(channel, stopFlag);
                    } catch (Exception e) {
                        e.printStackTrace();
                    } finally {
                        doneLatch.countDown();
                    }
                });
                producer.setName("Producer");
                producer.start();
                
                // Start consumer thread
                Thread consumer = new Thread(() -> {
                    try {
                        startLatch.await();
                        runConsumer(channel, stopFlag);
                    } catch (Exception e) {
                        e.printStackTrace();
                    } finally {
                        doneLatch.countDown();
                    }
                });
                consumer.setName("Consumer");
                consumer.start();
                
                // Start both threads
                System.out.println("Starting producer and consumer...\n");
                startLatch.countDown();
                
                // Wait for completion
                if (!doneLatch.await(30, TimeUnit.SECONDS)) {
                    System.err.println("Timeout waiting for threads to complete");
                    stopFlag.set(true);
                }
                
                // Display final metrics
                Metrics metrics = channel.getMetrics();
                System.out.println("\n\nFinal metrics:");
                System.out.println("  Messages sent: " + metrics.getMessagesSent());
                System.out.println("  Messages received: " + metrics.getMessagesReceived());
                System.out.println("  Bytes transferred: " + metrics.getBytesSent());
                System.out.println("  Send blocks: " + metrics.getSendBlocks());
                System.out.println("  Receive blocks: " + metrics.getReceiveBlocks());
            }
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void runProducer(Channel channel, AtomicBoolean stopFlag) throws PsyneException {
        System.out.println("[Producer] Starting...");
        
        for (int i = 0; i < MESSAGE_COUNT && !stopFlag.get(); i++) {
            // Send different types of messages
            int messageType = (i % 3) + 1; // Types 1, 2, 3
            String content;
            
            switch (messageType) {
                case 1:
                    content = "Text message #" + i;
                    break;
                case 2:
                    content = "{ \"id\": " + i + ", \"type\": \"json\" }";
                    break;
                case 3:
                    content = "<message id=\"" + i + "\">XML content</message>";
                    break;
                default:
                    content = "Unknown type";
            }
            
            channel.send(content.getBytes(StandardCharsets.UTF_8), messageType);
            
            // Simulate some work
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        
        // Send termination message
        channel.send("STOP".getBytes(StandardCharsets.UTF_8), 99);
        System.out.println("[Producer] Finished sending " + MESSAGE_COUNT + " messages");
    }
    
    private static void runConsumer(Channel channel, AtomicBoolean stopFlag) throws PsyneException {
        System.out.println("[Consumer] Starting...");
        
        int receivedCount = 0;
        int[] typeCounts = new int[4]; // Count by type
        
        while (!stopFlag.get()) {
            Channel.ReceivedMessage received = channel.receive(100, TimeUnit.MILLISECONDS);
            
            if (received != null) {
                try (Message msg = received.getMessage()) {
                    String content = new String(msg.toByteArray(), StandardCharsets.UTF_8);
                    
                    // Check for termination message
                    if (received.getType() == 99 && "STOP".equals(content)) {
                        System.out.println("[Consumer] Received stop signal");
                        break;
                    }
                    
                    receivedCount++;
                    if (received.getType() >= 1 && received.getType() <= 3) {
                        typeCounts[received.getType()]++;
                    }
                    
                    // Log every 10th message
                    if (receivedCount % 10 == 0) {
                        System.out.println("[Consumer] Received " + receivedCount + 
                                         " messages so far...");
                    }
                }
            }
        }
        
        System.out.println("[Consumer] Finished. Received " + receivedCount + " messages");
        System.out.println("[Consumer] Type distribution:");
        for (int i = 1; i <= 3; i++) {
            System.out.println("  Type " + i + ": " + typeCounts[i] + " messages");
        }
    }
}