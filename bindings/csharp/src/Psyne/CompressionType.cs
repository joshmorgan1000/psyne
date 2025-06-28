namespace Psyne
{
    /// <summary>
    /// Specifies the compression algorithm to use for messages.
    /// </summary>
    public enum CompressionType
    {
        /// <summary>
        /// No compression applied.
        /// </summary>
        None = 0,

        /// <summary>
        /// LZ4 compression algorithm.
        /// Fast compression and decompression with good compression ratios.
        /// </summary>
        LZ4 = 1,

        /// <summary>
        /// Zstandard (Zstd) compression algorithm.
        /// Excellent compression ratios with good performance.
        /// </summary>
        Zstd = 2,

        /// <summary>
        /// Snappy compression algorithm.
        /// Very fast compression with moderate compression ratios.
        /// </summary>
        Snappy = 3
    }
}