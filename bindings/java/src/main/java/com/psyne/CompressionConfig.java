package com.psyne;

/**
 * Configuration for message compression in Psyne channels.
 * 
 * This class provides a builder pattern for configuring compression settings
 * for a Psyne channel.
 */
public class CompressionConfig {
    
    private final CompressionType type;
    private final int level;
    private final long minSizeThreshold;
    private final boolean enableChecksum;
    
    private CompressionConfig(Builder builder) {
        this.type = builder.type;
        this.level = builder.level;
        this.minSizeThreshold = builder.minSizeThreshold;
        this.enableChecksum = builder.enableChecksum;
    }
    
    /**
     * Gets the compression type.
     * 
     * @return The compression type
     */
    public CompressionType getType() {
        return type;
    }
    
    /**
     * Gets the compression level.
     * 
     * @return The compression level
     */
    public int getLevel() {
        return level;
    }
    
    /**
     * Gets the minimum size threshold for compression.
     * Messages smaller than this threshold will not be compressed.
     * 
     * @return The minimum size threshold in bytes
     */
    public long getMinSizeThreshold() {
        return minSizeThreshold;
    }
    
    /**
     * Checks if checksum is enabled for compressed messages.
     * 
     * @return true if checksum is enabled, false otherwise
     */
    public boolean isChecksumEnabled() {
        return enableChecksum;
    }
    
    /**
     * Creates a new builder for CompressionConfig.
     * 
     * @return A new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Builder for CompressionConfig.
     */
    public static class Builder {
        private CompressionType type = CompressionType.NONE;
        private int level = -1; // Default compression level
        private long minSizeThreshold = 1024; // Default 1KB threshold
        private boolean enableChecksum = true;
        
        /**
         * Sets the compression type.
         * 
         * @param type The compression type
         * @return This builder
         */
        public Builder type(CompressionType type) {
            this.type = type;
            return this;
        }
        
        /**
         * Sets the compression level.
         * The meaning of the level depends on the compression type:
         * - LZ4: 0-16 (higher = better compression)
         * - ZSTD: 1-22 (higher = better compression)
         * - SNAPPY: ignored
         * 
         * @param level The compression level
         * @return This builder
         */
        public Builder level(int level) {
            this.level = level;
            return this;
        }
        
        /**
         * Sets the minimum size threshold for compression.
         * Messages smaller than this threshold will not be compressed.
         * 
         * @param threshold The minimum size in bytes
         * @return This builder
         */
        public Builder minSizeThreshold(long threshold) {
            this.minSizeThreshold = threshold;
            return this;
        }
        
        /**
         * Enables or disables checksum for compressed messages.
         * 
         * @param enable true to enable checksum, false to disable
         * @return This builder
         */
        public Builder enableChecksum(boolean enable) {
            this.enableChecksum = enable;
            return this;
        }
        
        /**
         * Builds the CompressionConfig.
         * 
         * @return A new CompressionConfig instance
         */
        public CompressionConfig build() {
            return new CompressionConfig(this);
        }
    }
}