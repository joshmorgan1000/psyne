import Foundation
import CPsyne

/// Main entry point for the Psyne high-performance messaging library
public enum Psyne {
    private static let initLock = NSLock()
    private static var isInitialized = false
    
    /// Gets the version of the Psyne library
    public static var version: String {
        let ptr = psyne_version()
        guard let version = ptr.flatMap({ String(cString: $0) }) else {
            return "Unknown"
        }
        return version
    }
    
    /// Initializes the Psyne library. This must be called before using any other Psyne functionality.
    /// - Throws: `PsyneError` if initialization fails
    public static func initialize() throws {
        initLock.lock()
        defer { initLock.unlock() }
        
        guard !isInitialized else { return }
        
        let result = psyne_init()
        try throwOnError(result, context: "Failed to initialize Psyne library")
        isInitialized = true
    }
    
    /// Cleans up the Psyne library. Call this when you're done using Psyne.
    public static func cleanup() {
        initLock.lock()
        defer { initLock.unlock() }
        
        if isInitialized {
            psyne_cleanup()
            isInitialized = false
        }
    }
    
    /// Creates a new channel builder for configuring and creating channels
    /// - Returns: A new `ChannelBuilder` instance
    /// - Throws: `PsyneError` if the library is not initialized
    public static func createChannel() throws -> ChannelBuilder {
        try ensureInitialized()
        return ChannelBuilder()
    }
    
    /// Creates a simple memory channel with default settings
    /// - Parameters:
    ///   - name: The memory channel name
    ///   - bufferSize: The buffer size in bytes (default: 1MB)
    /// - Returns: A new `Channel` instance
    /// - Throws: `PsyneError` if creation fails
    public static func createMemoryChannel(
        name: String,
        bufferSize: Int = 1024 * 1024
    ) throws -> Channel {
        return try createChannel()
            .memory(name)
            .withBufferSize(bufferSize)
            .build()
    }
    
    /// Creates a simple TCP channel with default settings
    /// - Parameters:
    ///   - host: The host address
    ///   - port: The port number
    ///   - bufferSize: The buffer size in bytes (default: 1MB)
    /// - Returns: A new `Channel` instance
    /// - Throws: `PsyneError` if creation fails
    public static func createTcpChannel(
        host: String,
        port: Int,
        bufferSize: Int = 1024 * 1024
    ) throws -> Channel {
        return try createChannel()
            .tcp(host: host, port: port)
            .withBufferSize(bufferSize)
            .build()
    }
    
    /// Creates a simple Unix socket channel with default settings
    /// - Parameters:
    ///   - path: The Unix socket path
    ///   - bufferSize: The buffer size in bytes (default: 1MB)
    /// - Returns: A new `Channel` instance
    /// - Throws: `PsyneError` if creation fails
    public static func createUnixSocketChannel(
        path: String,
        bufferSize: Int = 1024 * 1024
    ) throws -> Channel {
        return try createChannel()
            .unixSocket(path)
            .withBufferSize(bufferSize)
            .build()
    }
    
    /// Gets the description for an error code
    /// - Parameter errorCode: The error code
    /// - Returns: A human-readable description of the error
    public static func getErrorDescription(_ errorCode: psyne_error_t) -> String {
        let ptr = psyne_error_string(errorCode)
        guard let description = ptr.flatMap({ String(cString: $0) }) else {
            return "Unknown error (\(errorCode.rawValue))"
        }
        return description
    }
    
    /// Ensures the library is initialized, throwing an error if not
    private static func ensureInitialized() throws {
        if !isInitialized {
            try initialize()
        }
    }
}