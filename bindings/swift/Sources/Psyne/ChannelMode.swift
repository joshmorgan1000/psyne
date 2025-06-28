import CPsyne

/// Channel synchronization modes for different threading scenarios
public enum ChannelMode: CaseIterable, CustomStringConvertible {
    /// Single Producer, Single Consumer - Highest performance, lock-free
    case spsc
    /// Single Producer, Multiple Consumer - One writer, many readers
    case spmc
    /// Multiple Producer, Single Consumer - Many writers, one reader
    case mpsc
    /// Multiple Producer, Multiple Consumer - Full multi-threading support
    case mpmc
    
    /// Convert to C API enum value
    internal var cValue: psyne_channel_mode_t {
        switch self {
        case .spsc: return PSYNE_MODE_SPSC
        case .spmc: return PSYNE_MODE_SPMC
        case .mpsc: return PSYNE_MODE_MPSC
        case .mpmc: return PSYNE_MODE_MPMC
        }
    }
    
    /// Create from C API enum value
    internal init?(cValue: psyne_channel_mode_t) {
        switch cValue {
        case PSYNE_MODE_SPSC: self = .spsc
        case PSYNE_MODE_SPMC: self = .spmc
        case PSYNE_MODE_MPSC: self = .mpsc
        case PSYNE_MODE_MPMC: self = .mpmc
        default: return nil
        }
    }
    
    public var description: String {
        switch self {
        case .spsc: return "SPSC (Single Producer, Single Consumer)"
        case .spmc: return "SPMC (Single Producer, Multiple Consumer)"
        case .mpsc: return "MPSC (Multiple Producer, Single Consumer)"
        case .mpmc: return "MPMC (Multiple Producer, Multiple Consumer)"
        }
    }
    
    /// True if this mode supports multiple producers
    public var supportsMultipleProducers: Bool {
        switch self {
        case .spsc, .spmc: return false
        case .mpsc, .mpmc: return true
        }
    }
    
    /// True if this mode supports multiple consumers
    public var supportsMultipleConsumers: Bool {
        switch self {
        case .spsc, .mpsc: return false
        case .spmc, .mpmc: return true
        }
    }
}