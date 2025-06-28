import CPsyne

/// Channel type definitions for single or multiple message types
public enum ChannelType: CaseIterable, CustomStringConvertible {
    /// Optimized for single message type (no type metadata overhead)
    case singleType
    /// Supports multiple message types (small metadata overhead)
    case multiType
    
    /// Convert to C API enum value
    internal var cValue: psyne_channel_type_t {
        switch self {
        case .singleType: return PSYNE_TYPE_SINGLE
        case .multiType: return PSYNE_TYPE_MULTI
        }
    }
    
    /// Create from C API enum value
    internal init?(cValue: psyne_channel_type_t) {
        switch cValue {
        case PSYNE_TYPE_SINGLE: self = .singleType
        case PSYNE_TYPE_MULTI: self = .multiType
        default: return nil
        }
    }
    
    public var description: String {
        switch self {
        case .singleType: return "Single Type (optimized for one message type)"
        case .multiType: return "Multi Type (supports multiple message types)"
        }
    }
    
    /// True if this channel type supports multiple message types
    public var supportsMultipleTypes: Bool {
        switch self {
        case .singleType: return false
        case .multiType: return true
        }
    }
}