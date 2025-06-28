package com.psyne;

/**
 * Channel synchronization modes.
 * 
 * This enum defines the different synchronization modes available for Psyne channels,
 * controlling how producers and consumers can access the channel.
 */
public enum ChannelMode {
    /**
     * Single Producer, Single Consumer.
     * The most efficient mode with minimal synchronization overhead.
     */
    SPSC(0),
    
    /**
     * Single Producer, Multiple Consumer.
     * One producer can send messages to multiple consumers.
     */
    SPMC(1),
    
    /**
     * Multiple Producer, Single Consumer.
     * Multiple producers can send messages to a single consumer.
     */
    MPSC(2),
    
    /**
     * Multiple Producer, Multiple Consumer.
     * Multiple producers can send messages to multiple consumers.
     * This mode has the highest synchronization overhead.
     */
    MPMC(3);
    
    private final int value;
    
    ChannelMode(int value) {
        this.value = value;
    }
    
    /**
     * Gets the native value of this channel mode.
     * 
     * @return The native value
     */
    public int getValue() {
        return value;
    }
}