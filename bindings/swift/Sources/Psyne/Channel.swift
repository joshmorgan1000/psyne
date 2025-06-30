import Foundation
import CPsyne

/// A high-performance, zero-copy messaging channel
///
/// Channels are the core abstraction in Psyne for inter-process and
/// inter-thread communication. They provide zero-copy message passing
/// with configurable synchronization modes.
public final class Channel {
    /// The underlying C channel handle
    private let handle: UnsafeMutablePointer<psyne_channel_t>
    
    /// Whether the channel has been stopped
    private var _isStopped = false
    
    /// Lock for thread safety
    private let lock = NSLock()
    
    /// Create a channel from a C handle (internal use)
    internal init(handle: UnsafeMutablePointer<psyne_channel_t>) {
        self.handle = handle
    }
    
    deinit {
        stop()
        psyne_channel_destroy(handle)
    }
    
    /// The channel's URI
    public lazy var uri: String = {
        var buffer = [CChar](repeating: 0, count: 1024)
        let result = psyne_channel_get_uri(handle, &buffer, buffer.count)
        
        guard result == PSYNE_OK else {
            return "unknown"
        }
        
        return String(cString: buffer)
    }()
    
    /// Whether the channel is stopped
    public var isStopped: Bool {
        lock.lock()
        defer { lock.unlock() }
        
        var stopped: Bool = false
        let result = psyne_channel_is_stopped(handle, &stopped)
        
        if result == PSYNE_OK {
            _isStopped = stopped
        }
        
        return _isStopped
    }
    
    /// Stop the channel
    /// - Throws: `PsyneError` if stopping fails
    public func stop() throws {
        lock.lock()
        defer { lock.unlock() }
        
        guard !_isStopped else { return }
        
        let result = psyne_channel_stop(handle)
        try throwOnError(result, context: "Failed to stop channel")
        _isStopped = true
    }
    
    /// Send raw data through the channel
    /// - Parameters:
    ///   - data: The data to send
    ///   - messageType: The message type identifier
    /// - Throws: `PsyneError` if sending fails
    public func sendData<T>(_ data: T, messageType: UInt32 = 0) throws {
        try data.withUnsafeBytes { bytes in
            let result = psyne_send_data(
                handle,
                bytes.baseAddress,
                bytes.count,
                messageType
            )
            try throwOnError(result, context: "Failed to send data")
        }
    }
    
    /// Send raw bytes through the channel
    /// - Parameters:
    ///   - bytes: The bytes to send
    ///   - messageType: The message type identifier
    /// - Throws: `PsyneError` if sending fails
    public func sendBytes(_ bytes: Data, messageType: UInt32 = 0) throws {
        try bytes.withUnsafeBytes { buffer in
            let result = psyne_send_data(
                handle,
                buffer.baseAddress,
                buffer.count,
                messageType
            )
            try throwOnError(result, context: "Failed to send bytes")
        }
    }
    
    /// Receive raw data from the channel
    /// - Parameters:
    ///   - maxSize: Maximum size of data to receive
    ///   - timeout: Timeout in seconds (nil for blocking)
    /// - Returns: A tuple containing the received data and message type
    /// - Throws: `PsyneError` if receiving fails
    public func receiveData(
        maxSize: Int = 1024 * 1024,
        timeout: TimeInterval? = nil
    ) throws -> (data: Data, messageType: UInt32) {
        var buffer = Data(count: maxSize)
        var receivedSize: Int = 0
        var messageType: UInt32 = 0
        
        let timeoutMs: UInt32 = {
            if let timeout = timeout {
                return UInt32(timeout * 1000)
            } else {
                return UInt32.max // Blocking
            }
        }()
        
        let result = buffer.withUnsafeMutableBytes { bytes in
            psyne_receive_data(
                handle,
                bytes.baseAddress,
                bytes.count,
                &receivedSize,
                &messageType,
                timeoutMs
            )
        }
        
        try throwOnError(result, context: "Failed to receive data")
        
        // Trim the buffer to the actual received size
        buffer.count = receivedSize
        
        return (data: buffer, messageType: messageType)
    }
    
    /// Reserve space for a message
    /// - Parameter size: Size of the message data
    /// - Returns: A `Message` object for writing data
    /// - Throws: `PsyneError` if reservation fails
    public func reserveMessage(size: Int) throws -> Message {
        var messageHandle: UnsafeMutablePointer<psyne_message_t>?
        
        let result = psyne_message_reserve(handle, size, &messageHandle)
        try throwOnError(result, context: "Failed to reserve message")
        
        guard let handle = messageHandle else {
            throw PsyneError.unknown(-1)
        }
        
        return Message(handle: handle, size: size)
    }
    
    /// Receive a message from the channel
    /// - Parameter timeout: Timeout in seconds (nil for blocking)
    /// - Returns: A received `Message` or nil if no message available
    /// - Throws: `PsyneError` if receiving fails (except for timeout/no message)
    public func receiveMessage(timeout: TimeInterval? = nil) throws -> ReceivedMessage? {
        var messageHandle: UnsafeMutablePointer<psyne_message_t>?
        var messageType: UInt32 = 0
        
        let result: psyne_error_t
        
        if let timeout = timeout {
            let timeoutMs = UInt32(timeout * 1000)
            result = psyne_message_receive_timeout(handle, timeoutMs, &messageHandle, &messageType)
        } else {
            result = psyne_message_receive(handle, &messageHandle, &messageType)
        }
        
        // Handle the case where no message is available
        if result == PSYNE_ERROR_NO_MESSAGE || result == PSYNE_ERROR_TIMEOUT {
            return nil
        }
        
        try throwOnError(result, context: "Failed to receive message")
        
        guard let handle = messageHandle else {
            return nil
        }
        
        return ReceivedMessage(handle: handle, messageType: messageType)
    }
    
    /// Get channel metrics
    /// - Returns: Current channel metrics
    /// - Throws: `PsyneError` if getting metrics fails
    public func getMetrics() throws -> ChannelMetrics {
        var metrics = psyne_metrics_t()
        let result = psyne_channel_get_metrics(handle, &metrics)
        try throwOnError(result, context: "Failed to get channel metrics")
        return ChannelMetrics(cValue: metrics)
    }
    
    /// Enable or disable metrics collection
    /// - Parameter enabled: Whether to enable metrics
    /// - Throws: `PsyneError` if changing metrics state fails
    public func setMetricsEnabled(_ enabled: Bool) throws {
        let result = psyne_channel_enable_metrics(handle, enabled)
        try throwOnError(result, context: "Failed to set metrics state")
    }
    
    /// Reset channel metrics to zero
    /// - Throws: `PsyneError` if resetting metrics fails
    public func resetMetrics() throws {
        let result = psyne_channel_reset_metrics(handle)
        try throwOnError(result, context: "Failed to reset metrics")
    }
    
    /// Get the channel's buffer size
    /// - Returns: Buffer size in bytes
    /// - Throws: `PsyneError` if getting buffer size fails
    public func getBufferSize() throws -> Int {
        var size: Int = 0
        let result = psyne_channel_get_buffer_size(handle, &size)
        try throwOnError(result, context: "Failed to get buffer size")
        return size
    }
    
    // ===================================================================
    // Zero-copy API methods (v1.3.0)
    // ===================================================================
    
    /// Reserve space in ring buffer and return offset (zero-copy API)
    /// - Parameter size: Size of message to reserve
    /// - Returns: Offset within ring buffer, or UInt32.max if buffer is full
    /// - Throws: `PsyneError` if reservation fails
    public func reserveWriteSlot(size: Int) throws -> UInt32 {
        var offset: UInt32 = 0
        let result = psyne_channel_reserve_write_slot(handle, size, &offset)
        try throwOnError(result, context: "Failed to reserve write slot")
        return offset
    }
    
    /// Notify receiver that message is ready at offset (zero-copy API)
    /// - Parameters:
    ///   - offset: Offset within ring buffer where message data starts
    ///   - size: Size of the message
    /// - Throws: `PsyneError` if notification fails
    public func notifyMessageReady(offset: UInt32, size: Int) throws {
        let result = psyne_channel_notify_message_ready(handle, offset, size)
        try throwOnError(result, context: "Failed to notify message ready")
    }
    
    /// Consumer advances read pointer after processing message (zero-copy API)
    /// - Parameter size: Size of message that was consumed
    /// - Throws: `PsyneError` if advancing read pointer fails
    public func advanceReadPointer(size: Int) throws {
        let result = psyne_channel_advance_read_pointer(handle, size)
        try throwOnError(result, context: "Failed to advance read pointer")
    }
    
    /// Get a buffer pointer for zero-copy access to the ring buffer
    /// - Returns: An `UnsafeMutableBufferPointer` to the ring buffer, or nil if not available
    /// - Throws: `PsyneError` if getting buffer view fails
    /// - Warning: The returned buffer is only valid while the channel exists and
    ///   the ring buffer is not reallocated. Use with extreme caution.
    public func getBufferView() throws -> UnsafeMutableBufferPointer<UInt8>? {
        var ptr: UnsafeMutableRawPointer?
        var size: Int = 0
        
        let result = psyne_channel_get_buffer_span(handle, &ptr, &size)
        try throwOnError(result, context: "Failed to get buffer view")
        
        guard let baseAddress = ptr?.assumingMemoryBound(to: UInt8.self), size > 0 else {
            return nil
        }
        
        return UnsafeMutableBufferPointer(start: baseAddress, count: size)
    }
}