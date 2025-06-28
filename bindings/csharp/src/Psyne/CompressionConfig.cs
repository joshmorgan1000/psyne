namespace Psyne
{
    /// <summary>
    /// Configuration for message compression.
    /// </summary>
    public class CompressionConfig
    {
        /// <summary>
        /// Gets or sets the compression algorithm to use.
        /// </summary>
        public CompressionType Type { get; set; } = CompressionType.None;

        /// <summary>
        /// Gets or sets the compression level.
        /// Higher levels typically provide better compression at the cost of speed.
        /// The valid range depends on the compression algorithm.
        /// </summary>
        public int Level { get; set; } = 1;

        /// <summary>
        /// Gets or sets the minimum message size threshold for compression.
        /// Messages smaller than this size will not be compressed.
        /// </summary>
        public long MinSizeThreshold { get; set; } = 1024;

        /// <summary>
        /// Gets or sets whether to enable checksum validation for compressed messages.
        /// </summary>
        public bool EnableChecksum { get; set; } = true;

        /// <summary>
        /// Creates a compression configuration for LZ4 with default settings.
        /// </summary>
        /// <returns>A CompressionConfig configured for LZ4.</returns>
        public static CompressionConfig Lz4() => new()
        {
            Type = CompressionType.LZ4,
            Level = 1,
            MinSizeThreshold = 1024,
            EnableChecksum = true
        };

        /// <summary>
        /// Creates a compression configuration for Zstd with default settings.
        /// </summary>
        /// <returns>A CompressionConfig configured for Zstd.</returns>
        public static CompressionConfig Zstd() => new()
        {
            Type = CompressionType.Zstd,
            Level = 3,
            MinSizeThreshold = 1024,
            EnableChecksum = true
        };

        /// <summary>
        /// Creates a compression configuration for Snappy with default settings.
        /// </summary>
        /// <returns>A CompressionConfig configured for Snappy.</returns>
        public static CompressionConfig Snappy() => new()
        {
            Type = CompressionType.Snappy,
            Level = 1,
            MinSizeThreshold = 512,
            EnableChecksum = true
        };

        /// <summary>
        /// Converts this configuration to the native format.
        /// </summary>
        internal Native.PsyneNative.CompressionConfig ToNative()
        {
            return new Native.PsyneNative.CompressionConfig
            {
                Type = (Native.PsyneNative.CompressionType)Type,
                Level = Level,
                MinSizeThreshold = (UIntPtr)MinSizeThreshold,
                EnableChecksum = EnableChecksum
            };
        }
    }
}