package com.psyne;

import org.junit.jupiter.api.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Psyne Java bindings.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ChannelTest {
    
    @BeforeAll
    public static void setUp() throws PsyneException {
        // Initialize Psyne library once for all tests
        Psyne.init();
    }
    
    @AfterAll
    public static void tearDown() {
        Psyne.cleanup();
    }
    
    @Test
    @Order(1)
    public void testLibraryInitialization() {
        assertTrue(Psyne.isInitialized(), "Library should be initialized");
        assertNotNull(Psyne.getVersion(), "Version should not be null");
        System.out.println("Psyne version: " + Psyne.getVersion());
    }
    
    @Test
    @Order(2)
    public void testChannelCreation() throws PsyneException {
        try (Channel channel = Channel.builder()
                .uri("memory://test-channel")
                .bufferSize(1024 * 1024)
                .mode(ChannelMode.SPSC)
                .build()) {
            
            assertNotNull(channel, "Channel should be created");
            assertEquals("memory://test-channel", channel.getUri());
            assertEquals(1024 * 1024, channel.getBufferSize());
            assertFalse(channel.isStopped());
        }
    }
    
    @Test
    @Order(3)
    public void testSimpleMessageSendReceive() throws PsyneException {
        try (Channel channel = Channel.builder()
                .uri("memory://test-simple")
                .build()) {
            
            String testMessage = "Hello, Psyne!";
            byte[] data = testMessage.getBytes(StandardCharsets.UTF_8);
            
            // Send message
            channel.send(data, 1);
            
            // Receive message
            Channel.ReceivedMessage received = channel.receive();
            assertNotNull(received, "Should receive a message");
            assertEquals(1, received.getType(), "Message type should match");
            
            try (Message msg = received.getMessage()) {
                byte[] receivedData = msg.toByteArray();
                String receivedMessage = new String(receivedData, StandardCharsets.UTF_8);
                assertEquals(testMessage, receivedMessage, "Message content should match");
            }
        }
    }
    
    @Test
    @Order(4)
    public void testMultipleMessageTypes() throws PsyneException {
        try (Channel channel = Channel.builder()
                .uri("memory://test-multi")
                .type(ChannelType.MULTI)
                .build()) {
            
            // Send messages with different types
            channel.send("Type 1".getBytes(), 1);
            channel.send("Type 2".getBytes(), 2);
            channel.send("Type 3".getBytes(), 3);
            
            // Receive and verify
            for (int expectedType = 1; expectedType <= 3; expectedType++) {
                Channel.ReceivedMessage received = channel.receive();
                assertNotNull(received);
                assertEquals(expectedType, received.getType());
                
                try (Message msg = received.getMessage()) {
                    String content = new String(msg.toByteArray());
                    assertEquals("Type " + expectedType, content);
                }
            }
        }
    }
    
    @Test
    @Order(5)
    public void testZeroCopyOperations() throws PsyneException {
        try (Channel channel = Channel.builder()
                .uri("memory://test-zerocopy")
                .build()) {
            
            // Reserve and write directly
            try (Message message = channel.reserve(256)) {
                ByteBuffer buffer = message.getData();
                buffer.putInt(42);
                buffer.putDouble(3.14159);
                buffer.putLong(123456789L);
                message.send(10);
            }
            
            // Receive and read directly
            Channel.ReceivedMessage received = channel.receive();
            assertNotNull(received);
            assertEquals(10, received.getType());
            
            try (Message msg = received.getMessage()) {
                ByteBuffer buffer = msg.getData();
                assertEquals(42, buffer.getInt());
                assertEquals(3.14159, buffer.getDouble(), 0.00001);
                assertEquals(123456789L, buffer.getLong());
            }
        }
    }
    
    @Test
    @Order(6)
    public void testReceiveTimeout() throws PsyneException {
        try (Channel channel = Channel.builder()
                .uri("memory://test-timeout")
                .build()) {
            
            // Try to receive with timeout (no messages)
            long start = System.currentTimeMillis();
            Channel.ReceivedMessage received = channel.receive(100, TimeUnit.MILLISECONDS);
            long elapsed = System.currentTimeMillis() - start;
            
            assertNull(received, "Should not receive any message");
            assertTrue(elapsed >= 100, "Should wait at least 100ms");
            assertTrue(elapsed < 200, "Should not wait much longer than 100ms");
        }
    }
    
    @Test
    @Order(7)
    public void testMetrics() throws PsyneException {
        try (Channel channel = Channel.builder()
                .uri("memory://test-metrics")
                .build()) {
            
            channel.enableMetrics(true);
            
            // Send some messages
            for (int i = 0; i < 10; i++) {
                channel.send(("Message " + i).getBytes(), 1);
            }
            
            // Receive messages
            for (int i = 0; i < 10; i++) {
                Channel.ReceivedMessage received = channel.receive();
                assertNotNull(received);
                received.getMessage().close();
            }
            
            // Check metrics
            Metrics metrics = channel.getMetrics();
            assertEquals(10, metrics.getMessagesSent());
            assertEquals(10, metrics.getMessagesReceived());
            assertTrue(metrics.getBytesSent() > 0);
            assertTrue(metrics.getBytesReceived() > 0);
            
            // Reset metrics
            channel.resetMetrics();
            metrics = channel.getMetrics();
            assertEquals(0, metrics.getMessagesSent());
            assertEquals(0, metrics.getMessagesReceived());
        }
    }
    
    @Test
    @Order(8)
    public void testCompressionConfig() throws PsyneException {
        CompressionConfig config = CompressionConfig.builder()
                .type(CompressionType.LZ4)
                .level(9)
                .minSizeThreshold(512)
                .enableChecksum(true)
                .build();
        
        try (Channel channel = Channel.builder()
                .uri("memory://test-compression")
                .compression(config)
                .build()) {
            
            // Create compressible data
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 100; i++) {
                sb.append("This is a repetitive line of text. ");
            }
            byte[] data = sb.toString().getBytes();
            
            channel.send(data, 1);
            
            Channel.ReceivedMessage received = channel.receive();
            assertNotNull(received);
            
            try (Message msg = received.getMessage()) {
                byte[] receivedData = msg.toByteArray();
                assertArrayEquals(data, receivedData, "Data should match after decompression");
            }
        }
    }
    
    @Test
    @Order(9)
    public void testChannelStop() throws PsyneException {
        try (Channel channel = Channel.builder()
                .uri("memory://test-stop")
                .build()) {
            
            assertFalse(channel.isStopped());
            
            channel.stop();
            assertTrue(channel.isStopped());
            
            // Should not be able to send after stop
            assertThrows(PsyneException.class, () -> {
                channel.send("test".getBytes(), 1);
            });
        }
    }
    
    @Test
    @Order(10)
    public void testErrorHandling() {
        // Test invalid URI
        assertThrows(PsyneException.class, () -> {
            Channel.builder()
                    .uri("invalid://uri")
                    .build();
        });
        
        // Test missing URI
        assertThrows(IllegalArgumentException.class, () -> {
            Channel.builder().build();
        });
    }
}