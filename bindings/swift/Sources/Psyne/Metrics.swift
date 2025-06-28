import CPsyne

/// Channel performance metrics for debugging and monitoring
public struct ChannelMetrics: CustomStringConvertible {
    /// Number of messages sent through the channel
    public let messagesSent: UInt64
    
    /// Total bytes sent through the channel
    public let bytesSent: UInt64
    
    /// Number of messages received from the channel
    public let messagesReceived: UInt64
    
    /// Total bytes received from the channel
    public let bytesReceived: UInt64
    
    /// Number of times send() blocked waiting for space
    public let sendBlocks: UInt64
    
    /// Number of times receive() blocked waiting for data
    public let receiveBlocks: UInt64
    
    /// Create metrics from C API struct
    internal init(cValue: psyne_metrics_t) {
        self.messagesSent = cValue.messages_sent
        self.bytesSent = cValue.bytes_sent
        self.messagesReceived = cValue.messages_received
        self.bytesReceived = cValue.bytes_received
        self.sendBlocks = cValue.send_blocks
        self.receiveBlocks = cValue.receive_blocks
    }
    
    /// Create empty metrics
    public init() {
        self.messagesSent = 0
        self.bytesSent = 0
        self.messagesReceived = 0
        self.bytesReceived = 0
        self.sendBlocks = 0
        self.receiveBlocks = 0
    }
    
    public var description: String {
        return """
        Channel Metrics:
          Messages: \(messagesSent) sent, \(messagesReceived) received
          Bytes: \(bytesSent) sent, \(bytesReceived) received
          Blocks: \(sendBlocks) send, \(receiveBlocks) receive
        """
    }
    
    /// Average message size for sent messages
    public var averageSentMessageSize: Double {
        guard messagesSent > 0 else { return 0 }
        return Double(bytesSent) / Double(messagesSent)
    }
    
    /// Average message size for received messages
    public var averageReceivedMessageSize: Double {
        guard messagesReceived > 0 else { return 0 }
        return Double(bytesReceived) / Double(messagesReceived)
    }
    
    /// Send blocking rate (0.0 to 1.0)
    public var sendBlockingRate: Double {
        guard messagesSent > 0 else { return 0 }
        return Double(sendBlocks) / Double(messagesSent)
    }
    
    /// Receive blocking rate (0.0 to 1.0)
    public var receiveBlockingRate: Double {
        guard messagesReceived > 0 else { return 0 }
        return Double(receiveBlocks) / Double(messagesReceived)
    }
    
    /// Total throughput in bytes
    public var totalThroughput: UInt64 {
        return bytesSent + bytesReceived
    }
    
    /// Total message count
    public var totalMessages: UInt64 {
        return messagesSent + messagesReceived
    }
}