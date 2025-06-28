import Foundation
import CPsyne

/// Errors that can occur when using Psyne
public enum PsyneError: Error, CustomStringConvertible, LocalizedError {
    case invalidArgument(String)
    case outOfMemory
    case channelFull
    case noMessage
    case channelStopped
    case unsupported(String)
    case ioError(String)
    case timeout
    case unknown(Int32)
    
    /// Create a PsyneError from a C API error code
    internal init(code: psyne_error_t, context: String = "") {
        switch code {
        case PSYNE_OK:
            self = .unknown(0) // This shouldn't happen
        case PSYNE_ERROR_INVALID_ARGUMENT:
            self = .invalidArgument(context.isEmpty ? "Invalid argument" : context)
        case PSYNE_ERROR_OUT_OF_MEMORY:
            self = .outOfMemory
        case PSYNE_ERROR_CHANNEL_FULL:
            self = .channelFull
        case PSYNE_ERROR_NO_MESSAGE:
            self = .noMessage
        case PSYNE_ERROR_CHANNEL_STOPPED:
            self = .channelStopped
        case PSYNE_ERROR_UNSUPPORTED:
            self = .unsupported(context.isEmpty ? "Unsupported operation" : context)
        case PSYNE_ERROR_IO:
            self = .ioError(context.isEmpty ? "I/O error" : context)
        case PSYNE_ERROR_TIMEOUT:
            self = .timeout
        default:
            self = .unknown(code.rawValue)
        }
    }
    
    public var description: String {
        switch self {
        case .invalidArgument(let message):
            return "Invalid argument: \(message)"
        case .outOfMemory:
            return "Out of memory"
        case .channelFull:
            return "Channel is full"
        case .noMessage:
            return "No message available"
        case .channelStopped:
            return "Channel is stopped"
        case .unsupported(let message):
            return "Unsupported operation: \(message)"
        case .ioError(let message):
            return "I/O error: \(message)"
        case .timeout:
            return "Operation timed out"
        case .unknown(let code):
            return "Unknown error (code: \(code))"
        }
    }
    
    public var errorDescription: String? {
        return description
    }
}

/// Throws a PsyneError if the given error code indicates failure
internal func throwOnError(_ code: psyne_error_t, context: String = "") throws {
    guard code == PSYNE_OK else {
        throw PsyneError(code: code, context: context)
    }
}