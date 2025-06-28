// 
// Psyne+All.swift
// 
// Main module file that exports all public Swift APIs for Psyne
// This file ensures all public APIs are available when importing Psyne
//

// Re-export all public types and functions

// Core library management
@_exported import struct Psyne.Psyne

// Error handling
@_exported import enum Psyne.PsyneError

// Channel types and configuration
@_exported import enum Psyne.ChannelMode
@_exported import enum Psyne.ChannelType
@_exported import enum Psyne.CompressionType
@_exported import struct Psyne.CompressionConfig

// Core messaging classes
@_exported import class Psyne.Channel
@_exported import class Psyne.ChannelBuilder
@_exported import class Psyne.Message
@_exported import class Psyne.ReceivedMessage

// Message type protocol and built-in types
@_exported import protocol Psyne.MessageType
@_exported import enum Psyne.MessageTypes

// Metrics and monitoring
@_exported import struct Psyne.ChannelMetrics

// Async support (when available)
#if canImport(_Concurrency)
@_exported import class Psyne.AsyncChannel
#endif

/// Psyne Swift Bindings
/// 
/// High-performance zero-copy messaging library for Swift with support for:
/// - Multiple transport types (memory, IPC, TCP, Unix sockets, WebSockets)
/// - Configurable threading models (SPSC, SPMC, MPSC, MPMC)
/// - Built-in compression (LZ4, Zstandard, Snappy)
/// - Swift concurrency (async/await, AsyncSequence)
/// - Type-safe message protocols
/// - Comprehensive error handling
/// - Performance metrics and monitoring
/// 
/// ## Quick Start
/// 
/// ```swift
/// import Psyne
/// 
/// // Initialize the library
/// try Psyne.initialize()
/// defer { Psyne.cleanup() }
/// 
/// // Create a channel
/// let channel = try Psyne.createMemoryChannel(name: "my_channel")
/// 
/// // Send and receive messages
/// try channel.sendText("Hello, Psyne!")
/// if let message = try channel.receiveMessage() {
///     if let textMsg = message.asTextMessage() {
///         print("Received: \(textMsg.text)")
///     }
/// }
/// ```
/// 
/// ## Builder Pattern
/// 
/// ```swift
/// let channel = try Psyne.createChannel()
///     .tcp(host: "localhost", port: 8080)
///     .withBufferSize(2 * 1024 * 1024)
///     .multipleProducersMultipleConsumers()
///     .withLZ4Compression()
///     .build()
/// ```
/// 
/// ## Async/Await Support
/// 
/// ```swift
/// // Send asynchronously
/// try await channel.sendDataAsync("Hello".data(using: .utf8)!)
/// 
/// // Receive using async sequence
/// for await (data, messageType) in channel.dataSequence() {
///     // Process received data
/// }
/// ```
/// 
/// For more examples and documentation, see the Examples/ directory and README.md
public enum PsyneModule {
    /// The current version of the Psyne Swift bindings
    public static let version = "1.0.0"
    
    /// Information about the Swift bindings
    public static let info = """
        Psyne Swift Bindings v\(version)
        
        Features:
        - High-performance zero-copy messaging
        - Multiple transport types and threading models
        - Built-in compression support
        - Swift concurrency (async/await)
        - Type-safe message protocols
        - Comprehensive error handling
        - Performance metrics and monitoring
        
        For more information: https://github.com/joshmorgan1000/psyne
        """
}