import XCTest
@testable import Psyne

final class PsyneTests: XCTestCase {
    
    override func setUpWithError() throws {
        // Initialize Psyne before each test
        try Psyne.initialize()
    }
    
    override func tearDownWithError() throws {
        // Cleanup after each test
        Psyne.cleanup()
    }
    
    func testVersionString() throws {
        // Test that we can get a version string
        let version = Psyne.version
        XCTAssertFalse(version.isEmpty)
        XCTAssertNotEqual(version, "Unknown")
        print("Psyne version: \(version)")
    }
    
    func testMemoryChannelCreation() throws {
        // Test creating a simple memory channel
        let channel = try Psyne.createMemoryChannel(name: "test_channel")
        XCTAssertTrue(channel.uri.contains("memory://test_channel"))
        XCTAssertFalse(channel.isStopped)
        
        // Test buffer size
        let bufferSize = try channel.getBufferSize()
        XCTAssertEqual(bufferSize, 1024 * 1024) // Default 1MB
    }
    
    func testChannelBuilder() throws {
        // Test the builder pattern
        let channel = try Psyne.createChannel()
            .memory("builder_test")
            .withBufferSize(2048)
            .singleProducerSingleConsumer()
            .singleMessageType()
            .build()
        
        XCTAssertTrue(channel.uri.contains("memory://builder_test"))
        
        let bufferSize = try channel.getBufferSize()
        XCTAssertEqual(bufferSize, 2048)
    }
    
    func testBasicMessaging() throws {
        // Test basic send and receive
        let channel = try Psyne.createMemoryChannel(name: "messaging_test")
        
        let testData = "Hello, Psyne!".data(using: .utf8)!
        try channel.sendBytes(testData, messageType: 42)
        
        let (receivedData, messageType) = try channel.receiveData(timeout: 1.0)
        XCTAssertEqual(receivedData, testData)
        XCTAssertEqual(messageType, 42)
    }
    
    func testTextMessage() throws {
        // Test text message helper
        let channel = try Psyne.createMemoryChannel(name: "text_test")
        
        let text = "Test message"
        try channel.sendText(text)
        
        if let message = try channel.receiveMessage(timeout: 1.0) {
            XCTAssertTrue(message.isType(MessageTypes.TextMessage.self))
            
            if let textMsg = message.asTextMessage() {
                XCTAssertEqual(textMsg.text, text)
            } else {
                XCTFail("Failed to decode text message")
            }
        } else {
            XCTFail("No message received")
        }
    }
    
    func testFloatArrayMessage() throws {
        // Test float array message
        let channel = try Psyne.createMemoryChannel(name: "float_test")
        
        let floatArray: [Float] = [1.0, 2.5, 3.14, 4.2]
        try channel.sendFloatArray(floatArray)
        
        if let message = try channel.receiveMessage(timeout: 1.0) {
            XCTAssertTrue(message.isType(MessageTypes.FloatArrayMessage.self))
            
            if let floatMsg = message.asFloatArrayMessage() {
                XCTAssertEqual(floatMsg.values, floatArray)
                XCTAssertEqual(floatMsg.shape, [floatArray.count])
            } else {
                XCTFail("Failed to decode float array message")
            }
        } else {
            XCTFail("No message received")
        }
    }
    
    func testJSONMessage() throws {
        // Test JSON message
        let channel = try Psyne.createMemoryChannel(name: "json_test")
        
        let jsonData: [String: Any] = [
            "key": "value",
            "number": 42,
            "bool": true
        ]
        
        try channel.sendJSON(jsonData)
        
        if let message = try channel.receiveMessage(timeout: 1.0) {
            if let jsonMsg = message.asJSONMessage() {
                XCTAssertEqual(jsonMsg.json["key"] as? String, "value")
                XCTAssertEqual(jsonMsg.json["number"] as? Int, 42)
                XCTAssertEqual(jsonMsg.json["bool"] as? Bool, true)
            } else {
                XCTFail("Failed to decode JSON message")
            }
        } else {
            XCTFail("No message received")
        }
    }
    
    func testHeartbeatMessage() throws {
        // Test heartbeat message
        let channel = try Psyne.createMemoryChannel(name: "heartbeat_test")
        
        let sequenceNumber: UInt64 = 123
        let nodeID = "test-node"
        
        try channel.sendHeartbeat(sequenceNumber: sequenceNumber, nodeID: nodeID)
        
        if let message = try channel.receiveMessage(timeout: 1.0) {
            XCTAssertTrue(message.isType(MessageTypes.HeartbeatMessage.self))
            
            if let heartbeat = message.asHeartbeatMessage() {
                XCTAssertEqual(heartbeat.sequenceNumber, sequenceNumber)
                XCTAssertEqual(heartbeat.nodeID, nodeID)
            } else {
                XCTFail("Failed to decode heartbeat message")
            }
        } else {
            XCTFail("No message received")
        }
    }
    
    func testMetrics() throws {
        // Test metrics functionality
        let channel = try Psyne.createMemoryChannel(name: "metrics_test")
        
        // Enable metrics
        try channel.setMetricsEnabled(true)
        
        // Send some data
        let testData = "Metrics test data".data(using: .utf8)!
        try channel.sendBytes(testData)
        
        // Receive the data
        let _ = try channel.receiveData(timeout: 1.0)
        
        // Check metrics
        let metrics = try channel.getMetrics()
        XCTAssertEqual(metrics.messagesSent, 1)
        XCTAssertEqual(metrics.messagesReceived, 1)
        XCTAssertGreaterThan(metrics.bytesSent, 0)
        XCTAssertGreaterThan(metrics.bytesReceived, 0)
        
        // Reset metrics
        try channel.resetMetrics()
        let resetMetrics = try channel.getMetrics()
        XCTAssertEqual(resetMetrics.messagesSent, 0)
        XCTAssertEqual(resetMetrics.messagesReceived, 0)
    }
    
    func testChannelStop() throws {
        // Test stopping a channel
        let channel = try Psyne.createMemoryChannel(name: "stop_test")
        
        XCTAssertFalse(channel.isStopped)
        
        try channel.stop()
        XCTAssertTrue(channel.isStopped)
    }
    
    func testErrorHandling() throws {
        // Test various error conditions
        let channel = try Psyne.createMemoryChannel(name: "error_test")
        
        // Stop the channel
        try channel.stop()
        
        // Try to send on stopped channel - should throw error
        XCTAssertThrowsError(try channel.sendText("This should fail")) { error in
            if let psyneError = error as? PsyneError {
                XCTAssertEqual(psyneError, .channelStopped)
            } else {
                XCTFail("Expected PsyneError.channelStopped, got \(error)")
            }
        }
    }
    
    func testCompressionConfig() throws {
        // Test compression configuration
        let config1 = CompressionConfig.lz4Fast
        XCTAssertEqual(config1.type, .lz4)
        XCTAssertEqual(config1.level, 1)
        
        let config2 = CompressionConfig.zstdHigh
        XCTAssertEqual(config2.type, .zstd)
        XCTAssertEqual(config2.level, 9)
        
        let customConfig = CompressionConfig(
            type: .snappy,
            level: 3,
            minSizeThreshold: 512,
            enableChecksum: false
        )
        XCTAssertEqual(customConfig.type, .snappy)
        XCTAssertEqual(customConfig.level, 3)
        XCTAssertEqual(customConfig.minSizeThreshold, 512)
        XCTAssertFalse(customConfig.enableChecksum)
    }
    
    func testChannelModes() throws {
        // Test channel mode properties
        XCTAssertTrue(ChannelMode.mpsc.supportsMultipleProducers)
        XCTAssertFalse(ChannelMode.mpsc.supportsMultipleConsumers)
        
        XCTAssertFalse(ChannelMode.spmc.supportsMultipleProducers)
        XCTAssertTrue(ChannelMode.spmc.supportsMultipleConsumers)
        
        XCTAssertTrue(ChannelMode.mpmc.supportsMultipleProducers)
        XCTAssertTrue(ChannelMode.mpmc.supportsMultipleConsumers)
        
        XCTAssertFalse(ChannelMode.spsc.supportsMultipleProducers)
        XCTAssertFalse(ChannelMode.spsc.supportsMultipleConsumers)
    }
    
    func testChannelTypes() throws {
        // Test channel type properties
        XCTAssertFalse(ChannelType.singleType.supportsMultipleTypes)
        XCTAssertTrue(ChannelType.multiType.supportsMultipleTypes)
    }
    
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    func testAsyncMessaging() async throws {
        // Test async messaging
        let channel = try Psyne.createMemoryChannel(name: "async_test")
        
        let testData = "Async test data".data(using: .utf8)!
        
        // Send asynchronously
        try await channel.sendBytesAsync(testData, messageType: 100)
        
        // Receive asynchronously
        let (receivedData, messageType) = try await channel.receiveDataAsync(timeout: 1.0)
        
        XCTAssertEqual(receivedData, testData)
        XCTAssertEqual(messageType, 100)
    }
    
    static var allTests = [
        ("testVersionString", testVersionString),
        ("testMemoryChannelCreation", testMemoryChannelCreation),
        ("testChannelBuilder", testChannelBuilder),
        ("testBasicMessaging", testBasicMessaging),
        ("testTextMessage", testTextMessage),
        ("testFloatArrayMessage", testFloatArrayMessage),
        ("testJSONMessage", testJSONMessage),
        ("testHeartbeatMessage", testHeartbeatMessage),
        ("testMetrics", testMetrics),
        ("testChannelStop", testChannelStop),
        ("testErrorHandling", testErrorHandling),
        ("testCompressionConfig", testCompressionConfig),
        ("testChannelModes", testChannelModes),
        ("testChannelTypes", testChannelTypes),
        ("testAsyncMessaging", testAsyncMessaging),
    ]
}