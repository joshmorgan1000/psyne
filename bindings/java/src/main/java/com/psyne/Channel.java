package com.psyne;

import java.nio.ByteBuffer;
import java.util.concurrent.TimeUnit;

/**
 * Represents a Psyne communication channel.
 * 
 * This class provides the main interface for sending and receiving messages
 * through Psyne channels. It implements AutoCloseable to support try-with-resources
 * for automatic cleanup.
 * 
 * Example usage:
 * <pre>{@code
 * try (Channel channel = Channel.builder()
 *         .uri("tcp://localhost:8080")
 *         .bufferSize(1024 * 1024)
 *         .mode(ChannelMode.SPSC)
 *         .build()) {
 *     
 *     // Send a message
 *     channel.send("Hello, Psyne!".getBytes(), 1);
 *     
 *     // Receive a message
 *     ReceivedMessage msg = channel.receive();
 *     if (msg != null) {
 *         try (Message message = msg.getMessage()) {
 *             byte[] data = message.toByteArray();
 *             System.out.println("Received: " + new String(data));
 *         }
 *     }
 * }
 * }</pre>
 */
public class Channel implements AutoCloseable {
    
    private long nativeHandle;
    private boolean closed = false;
    
    /**
     * Creates a new Channel with the given native handle.
     * 
     * @param nativeHandle The native channel handle
     */
    private Channel(long nativeHandle) {
        this.nativeHandle = nativeHandle;
    }
    
    /**
     * Reserves space for a message in the channel.
     * 
     * @param size The size of the message in bytes
     * @return A Message object that can be filled with data and sent
     * @throws PsyneException if the reservation fails
     * @throws IllegalStateException if the channel is closed
     */
    public Message reserve(int size) throws PsyneException {
        checkNotClosed();
        long messageHandle = reserveNative(nativeHandle, size);
        if (messageHandle == 0) {
            throw new PsyneException(PsyneException.ErrorCode.UNKNOWN, 
                "Failed to reserve message");
        }
        return new Message(messageHandle, false);
    }
    
    /**
     * Sends data through the channel.
     * This is a convenience method that reserves a message, copies the data, and sends it.
     * 
     * @param data The data to send
     * @param type The message type ID
     * @throws PsyneException if the send operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public void send(byte[] data, int type) throws PsyneException {
        checkNotClosed();
        int result = sendDataNative(nativeHandle, data, data.length, type);
        if (result != 0) {
            throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(result));
        }
    }
    
    /**
     * Sends data from a ByteBuffer through the channel.
     * 
     * @param buffer The data to send
     * @param type The message type ID
     * @throws PsyneException if the send operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public void send(ByteBuffer buffer, int type) throws PsyneException {
        if (buffer.hasArray()) {
            byte[] array = buffer.array();
            int offset = buffer.arrayOffset() + buffer.position();
            int length = buffer.remaining();
            send(array, offset, length, type);
        } else {
            // For direct buffers or read-only buffers
            byte[] data = new byte[buffer.remaining()];
            buffer.get(data);
            send(data, type);
        }
    }
    
    /**
     * Sends a portion of a byte array through the channel.
     * 
     * @param data The data array
     * @param offset The starting offset in the array
     * @param length The number of bytes to send
     * @param type The message type ID
     * @throws PsyneException if the send operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public void send(byte[] data, int offset, int length, int type) throws PsyneException {
        checkNotClosed();
        if (offset == 0) {
            int result = sendDataNative(nativeHandle, data, length, type);
            if (result != 0) {
                throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(result));
            }
        } else {
            // Need to copy the sub-array
            byte[] subArray = new byte[length];
            System.arraycopy(data, offset, subArray, 0, length);
            send(subArray, type);
        }
    }
    
    /**
     * Receives a message from the channel (non-blocking).
     * 
     * @return A ReceivedMessage containing the message and type, or null if no message available
     * @throws PsyneException if the receive operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public ReceivedMessage receive() throws PsyneException {
        checkNotClosed();
        return receiveInternal(0);
    }
    
    /**
     * Receives a message from the channel with a timeout.
     * 
     * @param timeout The timeout value
     * @param unit The timeout unit
     * @return A ReceivedMessage containing the message and type, or null if timeout occurs
     * @throws PsyneException if the receive operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public ReceivedMessage receive(long timeout, TimeUnit unit) throws PsyneException {
        checkNotClosed();
        long timeoutMs = unit.toMillis(timeout);
        if (timeoutMs > Integer.MAX_VALUE) {
            timeoutMs = Integer.MAX_VALUE;
        }
        return receiveInternal((int) timeoutMs);
    }
    
    /**
     * Receives data from the channel into a byte array.
     * This is a convenience method that receives a message and copies its data.
     * 
     * @param buffer The buffer to receive data into
     * @return A ReceivedData object containing the received size and type, or null if no message
     * @throws PsyneException if the receive operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public ReceivedData receiveData(byte[] buffer) throws PsyneException {
        return receiveData(buffer, 0, TimeUnit.MILLISECONDS);
    }
    
    /**
     * Receives data from the channel into a byte array with a timeout.
     * 
     * @param buffer The buffer to receive data into
     * @param timeout The timeout value
     * @param unit The timeout unit
     * @return A ReceivedData object containing the received size and type, or null if timeout
     * @throws PsyneException if the receive operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public ReceivedData receiveData(byte[] buffer, long timeout, TimeUnit unit) throws PsyneException {
        checkNotClosed();
        long timeoutMs = unit.toMillis(timeout);
        if (timeoutMs > Integer.MAX_VALUE) {
            timeoutMs = Integer.MAX_VALUE;
        }
        
        int[] typeHolder = new int[1];
        int receivedSize = receiveDataNative(nativeHandle, buffer, buffer.length, 
                                            typeHolder, (int) timeoutMs);
        
        if (receivedSize < 0) {
            int errorCode = receivedSize;
            if (errorCode == PsyneException.ErrorCode.NO_MESSAGE.getCode() ||
                errorCode == PsyneException.ErrorCode.TIMEOUT.getCode()) {
                return null;
            }
            throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(errorCode));
        }
        
        return new ReceivedData(receivedSize, typeHolder[0]);
    }
    
    /**
     * Stops the channel.
     * After calling this method, no more messages can be sent or received.
     * 
     * @throws PsyneException if the stop operation fails
     */
    public void stop() throws PsyneException {
        if (closed) {
            return;
        }
        
        int result = stopNative(nativeHandle);
        if (result != 0) {
            throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(result));
        }
    }
    
    /**
     * Checks if the channel is stopped.
     * 
     * @return true if the channel is stopped, false otherwise
     * @throws PsyneException if the check fails
     * @throws IllegalStateException if the channel is closed
     */
    public boolean isStopped() throws PsyneException {
        checkNotClosed();
        return isStoppedNative(nativeHandle);
    }
    
    /**
     * Gets the channel URI.
     * 
     * @return The channel URI
     * @throws PsyneException if the operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public String getUri() throws PsyneException {
        checkNotClosed();
        return getUriNative(nativeHandle);
    }
    
    /**
     * Gets the channel buffer size.
     * 
     * @return The buffer size in bytes
     * @throws PsyneException if the operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public long getBufferSize() throws PsyneException {
        checkNotClosed();
        return getBufferSizeNative(nativeHandle);
    }
    
    /**
     * Gets the channel metrics.
     * 
     * @return The channel metrics
     * @throws PsyneException if the operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public Metrics getMetrics() throws PsyneException {
        checkNotClosed();
        long[] metrics = getMetricsNative(nativeHandle);
        return new Metrics(metrics[0], metrics[1], metrics[2], 
                          metrics[3], metrics[4], metrics[5]);
    }
    
    /**
     * Enables or disables metrics collection.
     * 
     * @param enable true to enable metrics, false to disable
     * @throws PsyneException if the operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public void enableMetrics(boolean enable) throws PsyneException {
        checkNotClosed();
        int result = enableMetricsNative(nativeHandle, enable);
        if (result != 0) {
            throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(result));
        }
    }
    
    /**
     * Resets the channel metrics.
     * 
     * @throws PsyneException if the operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public void resetMetrics() throws PsyneException {
        checkNotClosed();
        int result = resetMetricsNative(nativeHandle);
        if (result != 0) {
            throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(result));
        }
    }
    
    /**
     * Reserves space in the ring buffer and returns the offset (zero-copy API).
     * 
     * @param size Size of message to reserve
     * @return Offset within ring buffer, or 0xFFFFFFFF if buffer is full
     * @throws PsyneException if the operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public int reserveWriteSlot(int size) throws PsyneException {
        checkNotClosed();
        int result = reserveWriteSlotNative(nativeHandle, size);
        if (result < 0) {
            throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(result));
        }
        return result;
    }
    
    /**
     * Notifies the receiver that a message is ready at the specified offset (zero-copy API).
     * 
     * @param offset Offset within ring buffer where message data starts
     * @param size Size of the message
     * @throws PsyneException if the operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public void notifyMessageReady(int offset, int size) throws PsyneException {
        checkNotClosed();
        int result = notifyMessageReadyNative(nativeHandle, offset, size);
        if (result != 0) {
            throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(result));
        }
    }
    
    /**
     * Consumer advances read pointer after processing a message (zero-copy API).
     * 
     * @param size Size of message that was consumed
     * @throws PsyneException if the operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public void advanceReadPointer(int size) throws PsyneException {
        checkNotClosed();
        int result = advanceReadPointerNative(nativeHandle, size);
        if (result != 0) {
            throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(result));
        }
    }
    
    /**
     * Gets a direct ByteBuffer view of the ring buffer for zero-copy access.
     * 
     * @return A direct ByteBuffer over the ring buffer, or null if not available
     * @throws PsyneException if the operation fails
     * @throws IllegalStateException if the channel is closed
     */
    public ByteBuffer getBufferView() throws PsyneException {
        checkNotClosed();
        return getBufferViewNative(nativeHandle);
    }
    
    /**
     * Closes the channel and releases native resources.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        
        destroyNative(nativeHandle);
        nativeHandle = 0;
        closed = true;
    }
    
    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("Channel is closed");
        }
    }
    
    private ReceivedMessage receiveInternal(int timeoutMs) throws PsyneException {
        long[] handles = new long[1];
        int[] types = new int[1];
        
        int result = receiveNative(nativeHandle, timeoutMs, handles, types);
        
        if (result == PsyneException.ErrorCode.NO_MESSAGE.getCode() ||
            result == PsyneException.ErrorCode.TIMEOUT.getCode()) {
            return null;
        }
        
        if (result != 0) {
            throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(result));
        }
        
        return new ReceivedMessage(new Message(handles[0], true), types[0]);
    }
    
    @Override
    protected void finalize() throws Throwable {
        // Ensure cleanup if not properly closed
        if (!closed && nativeHandle != 0) {
            close();
        }
        super.finalize();
    }
    
    /**
     * Creates a new channel builder.
     * 
     * @return A new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Builder for creating Channel instances.
     */
    public static class Builder {
        private String uri;
        private long bufferSize = 1024 * 1024; // 1MB default
        private ChannelMode mode = ChannelMode.SPSC;
        private ChannelType type = ChannelType.SINGLE;
        private CompressionConfig compression = null;
        
        /**
         * Sets the channel URI.
         * Examples: "memory://buffer1", "tcp://localhost:8080", "ipc:///tmp/psyne.sock"
         * 
         * @param uri The channel URI
         * @return This builder
         */
        public Builder uri(String uri) {
            this.uri = uri;
            return this;
        }
        
        /**
         * Sets the channel buffer size.
         * 
         * @param size The buffer size in bytes
         * @return This builder
         */
        public Builder bufferSize(long size) {
            this.bufferSize = size;
            return this;
        }
        
        /**
         * Sets the channel synchronization mode.
         * 
         * @param mode The channel mode
         * @return This builder
         */
        public Builder mode(ChannelMode mode) {
            this.mode = mode;
            return this;
        }
        
        /**
         * Sets the channel type.
         * 
         * @param type The channel type
         * @return This builder
         */
        public Builder type(ChannelType type) {
            this.type = type;
            return this;
        }
        
        /**
         * Sets the compression configuration.
         * 
         * @param compression The compression configuration
         * @return This builder
         */
        public Builder compression(CompressionConfig compression) {
            this.compression = compression;
            return this;
        }
        
        /**
         * Configures the channel for UDP multicast communication.
         * 
         * @param multicastAddress The multicast group address (e.g., "239.255.0.1")
         * @param port The port number
         * @return This builder
         */
        public Builder multicast(String multicastAddress, int port) {
            return uri("udp://" + multicastAddress + ":" + port);
        }
        
        /**
         * Configures the channel for WebRTC peer-to-peer communication.
         * 
         * @param peerId The target peer identifier
         * @param signalingServerUri The WebSocket signaling server URI (default: ws://localhost:8080)
         * @return This builder
         */
        public Builder webrtc(String peerId, String signalingServerUri) {
            return uri("webrtc://" + peerId + "?signaling=" + signalingServerUri);
        }
        
        /**
         * Configures the channel for WebRTC peer-to-peer communication with default signaling server.
         * 
         * @param peerId The target peer identifier
         * @return This builder
         */
        public Builder webrtc(String peerId) {
            return webrtc(peerId, "ws://localhost:8080");
        }
        
        /**
         * Builds the channel.
         * 
         * @return A new Channel instance
         * @throws PsyneException if channel creation fails
         * @throws IllegalArgumentException if required parameters are missing
         */
        public Channel build() throws PsyneException {
            if (uri == null || uri.isEmpty()) {
                throw new IllegalArgumentException("URI is required");
            }
            
            long handle;
            if (compression != null) {
                handle = createCompressedNative(uri, bufferSize, mode.getValue(), 
                                              type.getValue(), compression);
            } else {
                handle = createNative(uri, bufferSize, mode.getValue(), type.getValue());
            }
            
            if (handle == 0) {
                throw new PsyneException(PsyneException.ErrorCode.UNKNOWN, 
                    "Failed to create channel");
            }
            
            return new Channel(handle);
        }
    }
    
    /**
     * Container for received message data.
     */
    public static class ReceivedMessage {
        private final Message message;
        private final int type;
        
        ReceivedMessage(Message message, int type) {
            this.message = message;
            this.type = type;
        }
        
        /**
         * Gets the received message.
         * 
         * @return The message
         */
        public Message getMessage() {
            return message;
        }
        
        /**
         * Gets the message type.
         * 
         * @return The message type ID
         */
        public int getType() {
            return type;
        }
    }
    
    /**
     * Container for received data information.
     */
    public static class ReceivedData {
        private final int size;
        private final int type;
        
        ReceivedData(int size, int type) {
            this.size = size;
            this.type = type;
        }
        
        /**
         * Gets the size of received data.
         * 
         * @return The data size in bytes
         */
        public int getSize() {
            return size;
        }
        
        /**
         * Gets the message type.
         * 
         * @return The message type ID
         */
        public int getType() {
            return type;
        }
    }
    
    // Native methods
    private static native long createNative(String uri, long bufferSize, int mode, int type);
    private static native long createCompressedNative(String uri, long bufferSize, int mode, 
                                                     int type, CompressionConfig compression);
    private static native void destroyNative(long handle);
    private static native int stopNative(long handle);
    private static native boolean isStoppedNative(long handle);
    private static native String getUriNative(long handle);
    private static native long getBufferSizeNative(long handle);
    private static native long[] getMetricsNative(long handle);
    private static native int enableMetricsNative(long handle, boolean enable);
    private static native int resetMetricsNative(long handle);
    private static native long reserveNative(long handle, int size);
    private static native int sendDataNative(long handle, byte[] data, int size, int type);
    private static native int receiveNative(long handle, int timeoutMs, long[] messageHandle, int[] type);
    private static native int receiveDataNative(long handle, byte[] buffer, int bufferSize, 
                                               int[] type, int timeoutMs);
    // Zero-copy API methods (v1.3.0)
    private static native int reserveWriteSlotNative(long handle, int size);
    private static native int notifyMessageReadyNative(long handle, int offset, int size);
    private static native int advanceReadPointerNative(long handle, int size);
    private static native ByteBuffer getBufferViewNative(long handle);
}