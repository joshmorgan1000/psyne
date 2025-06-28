import CPsyne

/// Compression algorithms supported by Psyne
public enum CompressionType: CaseIterable, CustomStringConvertible {
    /// No compression
    case none
    /// Fast compression/decompression with LZ4
    case lz4
    /// Better compression ratio with Zstandard
    case zstd
    /// Google Snappy - balanced speed/ratio
    case snappy
    
    /// Convert to C API enum value
    internal var cValue: psyne_compression_type_t {
        switch self {
        case .none: return PSYNE_COMPRESSION_NONE
        case .lz4: return PSYNE_COMPRESSION_LZ4
        case .zstd: return PSYNE_COMPRESSION_ZSTD
        case .snappy: return PSYNE_COMPRESSION_SNAPPY
        }
    }
    
    /// Create from C API enum value
    internal init?(cValue: psyne_compression_type_t) {
        switch cValue {
        case PSYNE_COMPRESSION_NONE: self = .none
        case PSYNE_COMPRESSION_LZ4: self = .lz4
        case PSYNE_COMPRESSION_ZSTD: self = .zstd
        case PSYNE_COMPRESSION_SNAPPY: self = .snappy
        default: return nil
        }
    }
    
    public var description: String {
        switch self {
        case .none: return "None"
        case .lz4: return "LZ4 (fast)"
        case .zstd: return "Zstandard (high compression)"
        case .snappy: return "Snappy (balanced)"
        }
    }
}

/// Configuration for compression behavior
public struct CompressionConfig {
    /// The compression algorithm to use
    public var type: CompressionType
    
    /// Compression level (algorithm dependent, typically 1-9)
    public var level: Int32
    
    /// Don't compress messages smaller than this threshold
    public var minSizeThreshold: Int
    
    /// Add checksum for compressed data
    public var enableChecksum: Bool
    
    /// Create a compression configuration
    /// - Parameters:
    ///   - type: The compression algorithm to use
    ///   - level: Compression level (default: 1)
    ///   - minSizeThreshold: Minimum size threshold in bytes (default: 128)
    ///   - enableChecksum: Whether to enable checksums (default: true)
    public init(
        type: CompressionType = .none,
        level: Int32 = 1,
        minSizeThreshold: Int = 128,
        enableChecksum: Bool = true
    ) {
        self.type = type
        self.level = level
        self.minSizeThreshold = minSizeThreshold
        self.enableChecksum = enableChecksum
    }
    
    /// Convert to C API struct
    internal var cValue: psyne_compression_config_t {
        return psyne_compression_config_t(
            type: type.cValue,
            level: level,
            min_size_threshold: minSizeThreshold,
            enable_checksum: enableChecksum
        )
    }
    
    /// Create from C API struct
    internal init(cValue: psyne_compression_config_t) {
        self.type = CompressionType(cValue: cValue.type) ?? .none
        self.level = cValue.level
        self.minSizeThreshold = Int(cValue.min_size_threshold)
        self.enableChecksum = cValue.enable_checksum
    }
    
    /// No compression configuration
    public static let none = CompressionConfig(type: .none)
    
    /// Fast LZ4 compression configuration
    public static let lz4Fast = CompressionConfig(type: .lz4, level: 1)
    
    /// High compression Zstandard configuration
    public static let zstdHigh = CompressionConfig(type: .zstd, level: 9)
    
    /// Balanced Snappy compression configuration
    public static let snappy = CompressionConfig(type: .snappy)
}