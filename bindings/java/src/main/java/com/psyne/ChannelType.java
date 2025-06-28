package com.psyne;

/**
 * Channel message type modes.
 * 
 * This enum defines whether a channel supports single or multiple message types.
 */
public enum ChannelType {
    /**
     * Single message type channel.
     * All messages in the channel have the same type.
     */
    SINGLE(0),
    
    /**
     * Multiple message types channel.
     * The channel can handle messages of different types.
     */
    MULTI(1);
    
    private final int value;
    
    ChannelType(int value) {
        this.value = value;
    }
    
    /**
     * Gets the native value of this channel type.
     * 
     * @return The native value
     */
    public int getValue() {
        return value;
    }
}