package com.psyne;

import java.nio.ByteBuffer;

/**
 * Represents a message in a Psyne channel.
 * 
 * This class provides access to message data and must be properly released
 * after use to avoid memory leaks. It implements AutoCloseable to support
 * try-with-resources for automatic cleanup.
 */
public class Message implements AutoCloseable {
    
    private long nativeHandle;
    private final boolean isReceived;
    private boolean released = false;
    
    /**
     * Creates a new Message wrapper for a native message handle.
     * 
     * @param nativeHandle The native message handle
     * @param isReceived True if this is a received message, false if reserved for sending
     */
    Message(long nativeHandle, boolean isReceived) {
        this.nativeHandle = nativeHandle;
        this.isReceived = isReceived;
    }
    
    /**
     * Gets the message data as a ByteBuffer.
     * The buffer is a direct buffer backed by native memory.
     * 
     * @return A ByteBuffer containing the message data
     * @throws IllegalStateException if the message has been released
     */
    public ByteBuffer getData() {
        checkNotReleased();
        return getDataNative(nativeHandle);
    }
    
    /**
     * Gets the size of the message data in bytes.
     * 
     * @return The size of the message data
     * @throws IllegalStateException if the message has been released
     */
    public int getSize() {
        checkNotReleased();
        return getSizeNative(nativeHandle);
    }
    
    /**
     * Copies the message data into a byte array.
     * This is a convenience method that allocates a new array and copies the data.
     * 
     * @return A byte array containing the message data
     * @throws IllegalStateException if the message has been released
     */
    public byte[] toByteArray() {
        ByteBuffer buffer = getData();
        byte[] data = new byte[buffer.remaining()];
        buffer.get(data);
        return data;
    }
    
    /**
     * Sends this message with the specified type.
     * This method can only be called on messages reserved for sending.
     * 
     * @param type The message type ID
     * @throws PsyneException if the send operation fails
     * @throws IllegalStateException if this is a received message or has been released
     */
    public void send(int type) throws PsyneException {
        checkNotReleased();
        if (isReceived) {
            throw new IllegalStateException("Cannot send a received message");
        }
        
        int result = sendNative(nativeHandle, type);
        if (result != 0) {
            throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(result));
        }
        
        // Message is consumed after sending
        nativeHandle = 0;
        released = true;
    }
    
    /**
     * Cancels this message without sending it.
     * This method can only be called on messages reserved for sending.
     * 
     * @throws IllegalStateException if this is a received message
     */
    public void cancel() {
        if (released) {
            return; // Already released
        }
        
        if (isReceived) {
            throw new IllegalStateException("Cannot cancel a received message");
        }
        
        cancelNative(nativeHandle);
        nativeHandle = 0;
        released = true;
    }
    
    /**
     * Releases this message.
     * For received messages, this returns the message buffer to the channel.
     * For reserved messages, this cancels the message without sending it.
     */
    @Override
    public void close() {
        if (released) {
            return;
        }
        
        if (isReceived) {
            releaseNative(nativeHandle);
        } else {
            cancelNative(nativeHandle);
        }
        
        nativeHandle = 0;
        released = true;
    }
    
    /**
     * Checks if this message has been released.
     * 
     * @return true if the message has been released, false otherwise
     */
    public boolean isReleased() {
        return released;
    }
    
    private void checkNotReleased() {
        if (released) {
            throw new IllegalStateException("Message has been released");
        }
    }
    
    @Override
    protected void finalize() throws Throwable {
        // Ensure cleanup if not properly closed
        if (!released && nativeHandle != 0) {
            close();
        }
        super.finalize();
    }
    
    // Native methods
    private static native ByteBuffer getDataNative(long handle);
    private static native int getSizeNative(long handle);
    private static native int sendNative(long handle, int type);
    private static native void cancelNative(long handle);
    private static native void releaseNative(long handle);
}