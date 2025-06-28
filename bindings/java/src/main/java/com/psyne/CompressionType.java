package com.psyne;

/**
 * Compression algorithms supported by Psyne.
 * 
 * This enum defines the different compression algorithms that can be used
 * to compress messages in Psyne channels.
 */
public enum CompressionType {
    /**
     * No compression.
     * Messages are sent as-is without any compression.
     */
    NONE(0),
    
    /**
     * LZ4 compression.
     * Fast compression algorithm with reasonable compression ratios.
     */
    LZ4(1),
    
    /**
     * Zstandard compression.
     * Modern compression algorithm with excellent compression ratios.
     */
    ZSTD(2),
    
    /**
     * Snappy compression.
     * Very fast compression algorithm optimized for speed.
     */
    SNAPPY(3);
    
    private final int value;
    
    CompressionType(int value) {
        this.value = value;
    }
    
    /**
     * Gets the native value of this compression type.
     * 
     * @return The native value
     */
    public int getValue() {
        return value;
    }
}