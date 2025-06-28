import Foundation
import CPsyne

/// Protocol for message types that can be sent through channels
public protocol MessageType {
    /// The unique identifier for this message type
    static var messageTypeID: UInt32 { get }
}

/// A message for sending data through a channel
public final class Message {
    /// The underlying C message handle
    private let handle: UnsafeMutablePointer<psyne_message_t>
    
    /// The size of the message data
    public let size: Int
    
    /// Whether the message has been sent or cancelled
    private var isFinalized = false
    
    /// Create a message with a C handle (internal use)
    internal init(handle: UnsafeMutablePointer<psyne_message_t>, size: Int) {
        self.handle = handle
        self.size = size
    }
    
    deinit {
        if !isFinalized {
            cancel()
        }
    }
    
    /// Get a pointer to the message data for writing
    /// - Returns: An unsafe mutable raw pointer to the message data
    /// - Throws: `PsyneError` if getting data pointer fails
    public func dataPointer() throws -> UnsafeMutableRawPointer {
        guard !isFinalized else {
            throw PsyneError.invalidArgument("Message has already been sent or cancelled")
        }
        
        var dataPtr: UnsafeMutableRawPointer?
        var dataSize: Int = 0
        
        let result = psyne_message_get_data(handle, &dataPtr, &dataSize)
        try throwOnError(result, context: "Failed to get message data pointer")
        
        guard let ptr = dataPtr else {
            throw PsyneError.unknown(-1)
        }
        
        return ptr
    }
    
    /// Write data to the message
    /// - Parameters:
    ///   - data: The data to write
    ///   - offset: Offset in the message buffer (default: 0)
    /// - Throws: `PsyneError` if writing fails
    public func writeData<T>(_ data: T, at offset: Int = 0) throws {
        let dataPtr = try dataPointer()
        
        try data.withUnsafeBytes { bytes in
            guard offset + bytes.count <= size else {
                throw PsyneError.invalidArgument("Data size (\(bytes.count)) + offset (\(offset)) exceeds message size (\(size))")
            }
            
            let targetPtr = dataPtr.advanced(by: offset)
            targetPtr.copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
    }
    
    /// Write bytes to the message
    /// - Parameters:
    ///   - bytes: The bytes to write
    ///   - offset: Offset in the message buffer (default: 0)
    /// - Throws: `PsyneError` if writing fails
    public func writeBytes(_ bytes: Data, at offset: Int = 0) throws {
        let dataPtr = try dataPointer()
        
        guard offset + bytes.count <= size else {
            throw PsyneError.invalidArgument("Data size (\(bytes.count)) + offset (\(offset)) exceeds message size (\(size))")
        }
        
        bytes.withUnsafeBytes { buffer in
            let targetPtr = dataPtr.advanced(by: offset)
            targetPtr.copyMemory(from: buffer.baseAddress!, byteCount: buffer.count)
        }
    }
    
    /// Send the message with a specific message type
    /// - Parameter messageType: The message type identifier
    /// - Throws: `PsyneError` if sending fails
    public func send(messageType: UInt32 = 0) throws {
        guard !isFinalized else {
            throw PsyneError.invalidArgument("Message has already been sent or cancelled")
        }
        
        let result = psyne_message_send(handle, messageType)
        try throwOnError(result, context: "Failed to send message")
        isFinalized = true
    }
    
    /// Send the message with a typed message type
    /// - Parameter type: The message type conforming to MessageType
    /// - Throws: `PsyneError` if sending fails
    public func send<T: MessageType>(as type: T.Type) throws {
        try send(messageType: T.messageTypeID)
    }
    
    /// Cancel the message without sending
    public func cancel() {
        guard !isFinalized else { return }
        
        psyne_message_cancel(handle)
        isFinalized = true
    }
}

/// A received message from a channel
public final class ReceivedMessage {
    /// The underlying C message handle
    private let handle: UnsafeMutablePointer<psyne_message_t>
    
    /// The message type identifier
    public let messageType: UInt32
    
    /// Whether the message has been released
    private var isReleased = false
    
    /// Create a received message with a C handle (internal use)
    internal init(handle: UnsafeMutablePointer<psyne_message_t>, messageType: UInt32) {
        self.handle = handle
        self.messageType = messageType
    }
    
    deinit {
        release()
    }
    
    /// Get a pointer to the message data for reading
    /// - Returns: An unsafe raw pointer to the message data and its size
    /// - Throws: `PsyneError` if getting data pointer fails
    public func dataPointer() throws -> (pointer: UnsafeRawPointer, size: Int) {
        guard !isReleased else {
            throw PsyneError.invalidArgument("Message has already been released")
        }
        
        var dataPtr: UnsafeMutableRawPointer?
        var dataSize: Int = 0
        
        let result = psyne_message_get_data(handle, &dataPtr, &dataSize)
        try throwOnError(result, context: "Failed to get message data pointer")
        
        guard let ptr = dataPtr else {
            throw PsyneError.unknown(-1)
        }
        
        return (pointer: UnsafeRawPointer(ptr), size: dataSize)
    }
    
    /// Read data from the message
    /// - Parameters:
    ///   - type: The type to read as
    ///   - offset: Offset in the message buffer (default: 0)
    /// - Returns: The data read from the message
    /// - Throws: `PsyneError` if reading fails
    public func readData<T>(as type: T.Type, at offset: Int = 0) throws -> T {
        let (dataPtr, dataSize) = try dataPointer()
        
        guard offset + MemoryLayout<T>.size <= dataSize else {
            throw PsyneError.invalidArgument("Type size (\(MemoryLayout<T>.size)) + offset (\(offset)) exceeds message size (\(dataSize))")
        }
        
        let typedPtr = dataPtr.advanced(by: offset).assumingMemoryBound(to: T.self)
        return typedPtr.pointee
    }
    
    /// Read bytes from the message
    /// - Parameters:
    ///   - count: Number of bytes to read (nil to read all remaining data)
    ///   - offset: Offset in the message buffer (default: 0)
    /// - Returns: The bytes read from the message
    /// - Throws: `PsyneError` if reading fails
    public func readBytes(count: Int? = nil, at offset: Int = 0) throws -> Data {
        let (dataPtr, dataSize) = try dataPointer()
        
        let bytesToRead = count ?? (dataSize - offset)
        
        guard offset + bytesToRead <= dataSize else {
            throw PsyneError.invalidArgument("Byte count (\(bytesToRead)) + offset (\(offset)) exceeds message size (\(dataSize))")
        }
        
        let sourcePtr = dataPtr.advanced(by: offset)
        return Data(bytes: sourcePtr, count: bytesToRead)
    }
    
    /// Copy all message data as bytes
    /// - Returns: All message data as Data
    /// - Throws: `PsyneError` if reading fails
    public func allBytes() throws -> Data {
        return try readBytes()
    }
    
    /// The size of the message data
    public var size: Int {
        do {
            let (_, size) = try dataPointer()
            return size
        } catch {
            return 0
        }
    }
    
    /// Check if this message is of a specific type
    /// - Parameter type: The message type to check against
    /// - Returns: True if the message is of the specified type
    public func isType<T: MessageType>(_ type: T.Type) -> Bool {
        return messageType == T.messageTypeID
    }
    
    /// Release the message and free its resources
    public func release() {
        guard !isReleased else { return }
        
        psyne_message_release(handle)
        isReleased = true
    }
}