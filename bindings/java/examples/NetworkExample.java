package com.psyne.examples;

import com.psyne.*;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

/**
 * Example demonstrating network communication with Psyne channels.
 * This example can run as either a server or client.
 */
public class NetworkExample {
    
    private static final String SERVER_URI = "tcp://localhost:8888";
    private static final int BUFFER_SIZE = 1024 * 1024; // 1MB
    
    public static void main(String[] args) {
        if (args.length != 1 || (!args[0].equals("server") && !args[0].equals("client"))) {
            System.err.println("Usage: NetworkExample <server|client>");
            System.exit(1);
        }
        
        try {
            // Initialize the Psyne library
            Psyne.init();
            
            if (args[0].equals("server")) {
                runServer();
            } else {
                runClient();
            }
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void runServer() throws PsyneException {
        System.out.println("Starting Psyne server on " + SERVER_URI);
        
        try (Channel channel = Channel.builder()
                .uri(SERVER_URI)
                .bufferSize(BUFFER_SIZE)
                .mode(ChannelMode.SPSC)
                .build()) {
            
            System.out.println("Server listening...");
            System.out.println("Waiting for messages (press Ctrl+C to stop)");
            
            while (true) {
                Channel.ReceivedMessage received = channel.receive(1000, TimeUnit.MILLISECONDS);
                
                if (received != null) {
                    try (Message msg = received.getMessage()) {
                        String content = new String(msg.toByteArray(), StandardCharsets.UTF_8);
                        System.out.println("[Server] Received (type=" + received.getType() + "): " + content);
                        
                        // Echo the message back with a different type
                        String response = "Echo: " + content;
                        channel.send(response.getBytes(StandardCharsets.UTF_8), received.getType() + 100);
                    }
                }
            }
        }
    }
    
    private static void runClient() throws PsyneException {
        System.out.println("Connecting to Psyne server at " + SERVER_URI);
        
        try (Channel channel = Channel.builder()
                .uri(SERVER_URI)
                .bufferSize(BUFFER_SIZE)
                .mode(ChannelMode.SPSC)
                .build()) {
            
            System.out.println("Connected to server");
            System.out.println("Type messages to send (or 'quit' to exit):");
            
            Scanner scanner = new Scanner(System.in);
            
            // Start a thread to receive responses
            Thread receiver = new Thread(() -> {
                try {
                    while (!Thread.currentThread().isInterrupted()) {
                        Channel.ReceivedMessage received = channel.receive(100, TimeUnit.MILLISECONDS);
                        if (received != null) {
                            try (Message msg = received.getMessage()) {
                                String content = new String(msg.toByteArray(), StandardCharsets.UTF_8);
                                System.out.println("[Client] " + content);
                            }
                        }
                    }
                } catch (Exception e) {
                    if (!Thread.currentThread().isInterrupted()) {
                        e.printStackTrace();
                    }
                }
            });
            receiver.setDaemon(true);
            receiver.start();
            
            // Read and send messages
            int messageType = 1;
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                
                if ("quit".equalsIgnoreCase(line)) {
                    break;
                }
                
                channel.send(line.getBytes(StandardCharsets.UTF_8), messageType++);
                System.out.println("[Client] Sent: " + line);
                
                // Give receiver thread time to process response
                Thread.sleep(100);
            }
            
            receiver.interrupt();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}