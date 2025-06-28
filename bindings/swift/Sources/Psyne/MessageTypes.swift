import Foundation

/// Common message types for Psyne
public enum MessageTypes {
    
    /// A simple text message
    public struct TextMessage: MessageType, Codable {
        public static let messageTypeID: UInt32 = 1
        
        public let text: String
        public let timestamp: Date
        
        public init(text: String, timestamp: Date = Date()) {
            self.text = text
            self.timestamp = timestamp
        }
    }
    
    /// A binary data message
    public struct BinaryMessage: MessageType {
        public static let messageTypeID: UInt32 = 2
        
        public let data: Data
        public let contentType: String
        public let timestamp: Date
        
        public init(data: Data, contentType: String = "application/octet-stream", timestamp: Date = Date()) {
            self.data = data
            self.contentType = contentType
            self.timestamp = timestamp
        }
    }
    
    /// A numeric array message for ML/AI applications
    public struct FloatArrayMessage: MessageType, Codable {
        public static let messageTypeID: UInt32 = 3
        
        public let values: [Float]
        public let shape: [Int]
        public let timestamp: Date
        
        public init(values: [Float], shape: [Int] = [], timestamp: Date = Date()) {
            self.values = values
            self.shape = shape.isEmpty ? [values.count] : shape
            self.timestamp = timestamp
        }
        
        /// Create a 1D array
        public static func array1D(_ values: [Float]) -> FloatArrayMessage {
            return FloatArrayMessage(values: values, shape: [values.count])
        }
        
        /// Create a 2D matrix
        public static func matrix(_ values: [Float], rows: Int, cols: Int) -> FloatArrayMessage? {
            guard values.count == rows * cols else { return nil }
            return FloatArrayMessage(values: values, shape: [rows, cols])
        }
        
        /// Get element at index (1D)
        public subscript(index: Int) -> Float? {
            guard shape.count == 1 && index >= 0 && index < values.count else { return nil }
            return values[index]
        }
        
        /// Get element at row, col (2D)
        public subscript(row: Int, col: Int) -> Float? {
            guard shape.count == 2 && shape[0] > row && shape[1] > col else { return nil }
            let index = row * shape[1] + col
            return values[index]
        }
    }
    
    /// A JSON message for structured data
    public struct JSONMessage: MessageType {
        public static let messageTypeID: UInt32 = 4
        
        public let json: [String: Any]
        public let timestamp: Date
        
        public init(json: [String: Any], timestamp: Date = Date()) {
            self.json = json
            self.timestamp = timestamp
        }
        
        /// Create from a Codable object
        public init<T: Codable>(from object: T, timestamp: Date = Date()) throws {
            let encoder = JSONEncoder()
            let data = try encoder.encode(object)
            let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])
            
            guard let dictionary = jsonObject as? [String: Any] else {
                throw PsyneError.invalidArgument("Object cannot be serialized to JSON dictionary")
            }
            
            self.json = dictionary
            self.timestamp = timestamp
        }
        
        /// Decode to a Codable object
        public func decode<T: Codable>(as type: T.Type) throws -> T {
            let data = try JSONSerialization.data(withJSONObject: json, options: [])
            let decoder = JSONDecoder()
            return try decoder.decode(type, from: data)
        }
    }
    
    /// A heartbeat/ping message
    public struct HeartbeatMessage: MessageType, Codable {
        public static let messageTypeID: UInt32 = 100
        
        public let timestamp: Date
        public let sequenceNumber: UInt64
        public let nodeID: String
        
        public init(sequenceNumber: UInt64, nodeID: String = UUID().uuidString, timestamp: Date = Date()) {
            self.timestamp = timestamp
            self.sequenceNumber = sequenceNumber
            self.nodeID = nodeID
        }
    }
    
    /// A control message for channel management
    public struct ControlMessage: MessageType, Codable {
        public static let messageTypeID: UInt32 = 101
        
        public enum Command: String, Codable {
            case start
            case stop
            case pause
            case resume
            case reset
            case status
        }
        
        public let command: Command
        public let parameters: [String: String]
        public let timestamp: Date
        
        public init(command: Command, parameters: [String: String] = [:], timestamp: Date = Date()) {
            self.command = command
            self.parameters = parameters
            self.timestamp = timestamp
        }
    }
}

/// Convenience extensions for sending typed messages
extension Channel {
    
    /// Send a text message
    /// - Parameter text: The text to send
    /// - Throws: `PsyneError` if sending fails
    public func sendText(_ text: String) throws {
        let message = MessageTypes.TextMessage(text: text)
        try sendCodable(message)
    }
    
    /// Send a float array
    /// - Parameter values: The float values to send
    /// - Throws: `PsyneError` if sending fails
    public func sendFloatArray(_ values: [Float]) throws {
        let message = MessageTypes.FloatArrayMessage.array1D(values)
        try sendCodable(message)
    }
    
    /// Send a JSON object
    /// - Parameter json: The JSON dictionary to send
    /// - Throws: `PsyneError` if sending fails
    public func sendJSON(_ json: [String: Any]) throws {
        let message = MessageTypes.JSONMessage(json: json)
        try sendCodable(message)
    }
    
    /// Send a Codable object as JSON
    /// - Parameter object: The Codable object to send
    /// - Throws: `PsyneError` if sending fails
    public func sendCodable<T: Codable>(_ object: T) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(object)
        try sendBytes(data, messageType: type(of: object).messageTypeID)
    }
    
    /// Send a heartbeat message
    /// - Parameters:
    ///   - sequenceNumber: The sequence number
    ///   - nodeID: The node identifier
    /// - Throws: `PsyneError` if sending fails
    public func sendHeartbeat(sequenceNumber: UInt64, nodeID: String = UUID().uuidString) throws {
        let message = MessageTypes.HeartbeatMessage(sequenceNumber: sequenceNumber, nodeID: nodeID)
        try sendCodable(message)
    }
    
    /// Send a control command
    /// - Parameters:
    ///   - command: The control command
    ///   - parameters: Optional parameters
    /// - Throws: `PsyneError` if sending fails
    public func sendControl(_ command: MessageTypes.ControlMessage.Command, parameters: [String: String] = [:]) throws {
        let message = MessageTypes.ControlMessage(command: command, parameters: parameters)
        try sendCodable(message)
    }
}

/// Convenience extensions for receiving typed messages
extension ReceivedMessage {
    
    /// Decode the message as a Codable type
    /// - Parameter type: The type to decode as
    /// - Returns: The decoded object
    /// - Throws: `PsyneError` or decoding errors
    public func decodeCodable<T: Codable>(as type: T.Type) throws -> T {
        let data = try allBytes()
        let decoder = JSONDecoder()
        return try decoder.decode(type, from: data)
    }
    
    /// Try to decode as a text message
    /// - Returns: The text message if successful
    public func asTextMessage() -> MessageTypes.TextMessage? {
        guard isType(MessageTypes.TextMessage.self) else { return nil }
        return try? decodeCodable(as: MessageTypes.TextMessage.self)
    }
    
    /// Try to decode as a float array message
    /// - Returns: The float array message if successful
    public func asFloatArrayMessage() -> MessageTypes.FloatArrayMessage? {
        guard isType(MessageTypes.FloatArrayMessage.self) else { return nil }
        return try? decodeCodable(as: MessageTypes.FloatArrayMessage.self)
    }
    
    /// Try to decode as a JSON message
    /// - Returns: The JSON message if successful
    public func asJSONMessage() -> MessageTypes.JSONMessage? {
        guard isType(MessageTypes.JSONMessage.self) else { return nil }
        
        do {
            let data = try allBytes()
            let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])
            guard let dictionary = jsonObject as? [String: Any] else { return nil }
            return MessageTypes.JSONMessage(json: dictionary)
        } catch {
            return nil
        }
    }
    
    /// Try to decode as a heartbeat message
    /// - Returns: The heartbeat message if successful
    public func asHeartbeatMessage() -> MessageTypes.HeartbeatMessage? {
        guard isType(MessageTypes.HeartbeatMessage.self) else { return nil }
        return try? decodeCodable(as: MessageTypes.HeartbeatMessage.self)
    }
    
    /// Try to decode as a control message
    /// - Returns: The control message if successful
    public func asControlMessage() -> MessageTypes.ControlMessage? {
        guard isType(MessageTypes.ControlMessage.self) else { return nil }
        return try? decodeCodable(as: MessageTypes.ControlMessage.self)
    }
}

// Extension to add messageTypeID to Codable types
extension MessageTypes.TextMessage: MessageType {}
extension MessageTypes.FloatArrayMessage: MessageType {}
extension MessageTypes.HeartbeatMessage: MessageType {}
extension MessageTypes.ControlMessage: MessageType {}