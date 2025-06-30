import Foundation
import CPsyne

/// A builder pattern for creating and configuring channels
public final class ChannelBuilder {
    private var uri: String = ""
    private var bufferSize: Int = 1024 * 1024 // 1MB default
    private var mode: ChannelMode = .spsc
    private var type: ChannelType = .multiType
    private var compression: CompressionConfig? = nil
    
    /// Create a new channel builder
    public init() {}
    
    /// Set the channel to use memory transport
    /// - Parameter name: The name of the memory channel
    /// - Returns: Self for chaining
    public func memory(_ name: String) -> ChannelBuilder {
        self.uri = "memory://\(name)"
        return self
    }
    
    /// Set the channel to use IPC (inter-process communication) transport
    /// - Parameter name: The name of the IPC channel
    /// - Returns: Self for chaining
    public func ipc(_ name: String) -> ChannelBuilder {
        self.uri = "ipc://\(name)"
        return self
    }
    
    /// Set the channel to use TCP transport as a client
    /// - Parameters:
    ///   - host: The host to connect to
    ///   - port: The port to connect to
    /// - Returns: Self for chaining
    public func tcp(host: String, port: Int) -> ChannelBuilder {
        self.uri = "tcp://\(host):\(port)"
        return self
    }
    
    /// Set the channel to use TCP transport as a server
    /// - Parameter port: The port to listen on
    /// - Returns: Self for chaining
    public func tcpServer(port: Int) -> ChannelBuilder {
        self.uri = "tcp://:\(port)"
        return self
    }
    
    /// Set the channel to use Unix domain socket transport as a client
    /// - Parameter path: The path to the Unix socket
    /// - Returns: Self for chaining
    public func unixSocket(_ path: String) -> ChannelBuilder {
        self.uri = "unix://\(path)"
        return self
    }
    
    /// Set the channel to use Unix domain socket transport as a server
    /// - Parameter path: The path to the Unix socket
    /// - Returns: Self for chaining
    public func unixSocketServer(_ path: String) -> ChannelBuilder {
        self.uri = "unix://@\(path)"
        return self
    }
    
    /// Set the channel to use WebSocket transport as a client
    /// - Parameters:
    ///   - host: The host to connect to
    ///   - port: The port to connect to
    /// - Returns: Self for chaining
    public func webSocket(host: String, port: Int) -> ChannelBuilder {
        self.uri = "ws://\(host):\(port)"
        return self
    }
    
    /// Set the channel to use WebSocket transport as a server
    /// - Parameter port: The port to listen on
    /// - Returns: Self for chaining
    public func webSocketServer(port: Int) -> ChannelBuilder {
        self.uri = "ws://:\(port)"
        return self
    }
    
    /// Set the channel to use secure WebSocket transport as a client
    /// - Parameters:
    ///   - host: The host to connect to
    ///   - port: The port to connect to
    /// - Returns: Self for chaining
    public func secureWebSocket(host: String, port: Int) -> ChannelBuilder {
        self.uri = "wss://\(host):\(port)"
        return self
    }
    
    // ===================================================================
    // v1.3.0 Transport Methods
    // ===================================================================
    
    /// Set the channel to use UDP multicast transport
    /// - Parameters:
    ///   - multicastAddress: The multicast group address (e.g., "239.255.0.1")
    ///   - port: The port number
    /// - Returns: Self for chaining
    public func multicast(multicastAddress: String, port: Int) -> ChannelBuilder {
        self.uri = "udp://\(multicastAddress):\(port)"
        return self
    }
    
    /// Set the channel to use WebRTC transport for peer-to-peer communication
    /// - Parameters:
    ///   - peerId: The target peer identifier
    ///   - signalingServerUri: The WebSocket signaling server URI (default: ws://localhost:8080)
    /// - Returns: Self for chaining
    public func webrtc(peerId: String, signalingServerUri: String = "ws://localhost:8080") -> ChannelBuilder {
        self.uri = "webrtc://\(peerId)?signaling=\(signalingServerUri)"
        return self
    }
    
    /// Set a custom URI for the channel
    /// - Parameter uri: The custom URI
    /// - Returns: Self for chaining
    public func withUri(_ uri: String) -> ChannelBuilder {
        self.uri = uri
        return self
    }
    
    /// Set the buffer size for the channel
    /// - Parameter size: The buffer size in bytes
    /// - Returns: Self for chaining
    public func withBufferSize(_ size: Int) -> ChannelBuilder {
        self.bufferSize = size
        return self
    }
    
    /// Set the channel mode (threading configuration)
    /// - Parameter mode: The channel mode
    /// - Returns: Self for chaining
    public func withMode(_ mode: ChannelMode) -> ChannelBuilder {
        self.mode = mode
        return self
    }
    
    /// Set the channel type (single or multi-type messages)
    /// - Parameter type: The channel type
    /// - Returns: Self for chaining
    public func withType(_ type: ChannelType) -> ChannelBuilder {
        self.type = type
        return self
    }
    
    /// Set compression configuration for the channel
    /// - Parameter config: The compression configuration
    /// - Returns: Self for chaining
    public func withCompression(_ config: CompressionConfig) -> ChannelBuilder {
        self.compression = config
        return self
    }
    
    /// Enable LZ4 compression with default settings
    /// - Returns: Self for chaining
    public func withLZ4Compression() -> ChannelBuilder {
        self.compression = .lz4Fast
        return self
    }
    
    /// Enable Zstandard compression with high compression ratio
    /// - Returns: Self for chaining
    public func withZstdCompression() -> ChannelBuilder {
        self.compression = .zstdHigh
        return self
    }
    
    /// Enable Snappy compression with balanced performance
    /// - Returns: Self for chaining
    public func withSnappyCompression() -> ChannelBuilder {
        self.compression = .snappy
        return self
    }
    
    /// Configure for single producer, single consumer (highest performance)
    /// - Returns: Self for chaining
    public func singleProducerSingleConsumer() -> ChannelBuilder {
        self.mode = .spsc
        return self
    }
    
    /// Configure for single producer, multiple consumers
    /// - Returns: Self for chaining
    public func singleProducerMultipleConsumers() -> ChannelBuilder {
        self.mode = .spmc
        return self
    }
    
    /// Configure for multiple producers, single consumer
    /// - Returns: Self for chaining
    public func multipleProducersSingleConsumer() -> ChannelBuilder {
        self.mode = .mpsc
        return self
    }
    
    /// Configure for multiple producers, multiple consumers
    /// - Returns: Self for chaining
    public func multipleProducersMultipleConsumers() -> ChannelBuilder {
        self.mode = .mpmc
        return self
    }
    
    /// Configure for single message type (optimized)
    /// - Returns: Self for chaining
    public func singleMessageType() -> ChannelBuilder {
        self.type = .singleType
        return self
    }
    
    /// Configure for multiple message types
    /// - Returns: Self for chaining
    public func multipleMessageTypes() -> ChannelBuilder {
        self.type = .multiType
        return self
    }
    
    /// Build the channel with the configured settings
    /// - Returns: A new `Channel` instance
    /// - Throws: `PsyneError` if channel creation fails
    public func build() throws -> Channel {
        guard !uri.isEmpty else {
            throw PsyneError.invalidArgument("Channel URI must be set")
        }
        
        guard bufferSize > 0 else {
            throw PsyneError.invalidArgument("Buffer size must be greater than 0")
        }
        
        var channelHandle: UnsafeMutablePointer<psyne_channel_t>?
        let result: psyne_error_t
        
        if let compression = compression {
            var compressionConfig = compression.cValue
            result = psyne_channel_create_compressed(
                uri,
                bufferSize,
                mode.cValue,
                type.cValue,
                &compressionConfig,
                &channelHandle
            )
        } else {
            result = psyne_channel_create(
                uri,
                bufferSize,
                mode.cValue,
                type.cValue,
                &channelHandle
            )
        }
        
        try throwOnError(result, context: "Failed to create channel with URI: \(uri)")
        
        guard let handle = channelHandle else {
            throw PsyneError.unknown(-1)
        }
        
        return Channel(handle: handle)
    }
    
    /// Get a description of the current configuration
    public var description: String {
        var components = [String]()
        
        if !uri.isEmpty {
            components.append("URI: \(uri)")
        }
        
        components.append("Buffer: \(bufferSize) bytes")
        components.append("Mode: \(mode)")
        components.append("Type: \(type)")
        
        if let compression = compression {
            components.append("Compression: \(compression.type)")
        }
        
        return "ChannelBuilder(\(components.joined(separator: ", ")))"
    }
}