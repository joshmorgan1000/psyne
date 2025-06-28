import Foundation
import Psyne

/// Example demonstrating the builder pattern for channel creation
@main
struct BuilderPatternExample {
    static func main() async {
        do {
            // Initialize the Psyne library
            try Psyne.initialize()
            defer { Psyne.cleanup() }
            
            print("Psyne Swift Builder Pattern Example")
            print("Version: \(Psyne.version)")
            print()
            
            // Example 1: Memory channel with custom configuration
            print("Creating memory channel with custom configuration...")
            let memoryChannel = try Psyne.createChannel()
                .memory("custom_memory")
                .withBufferSize(2 * 1024 * 1024) // 2MB buffer
                .multipleProducersMultipleConsumers()
                .multipleMessageTypes()
                .withLZ4Compression()
                .build()
            
            print("Memory channel created: \(memoryChannel.uri)")
            print("Buffer size: \(try memoryChannel.getBufferSize()) bytes")
            
            // Example 2: TCP server channel
            print("\nCreating TCP server channel...")
            let tcpServer = try Psyne.createChannel()
                .tcpServer(port: 8080)
                .withBufferSize(512 * 1024) // 512KB buffer
                .singleProducerSingleConsumer()
                .singleMessageType()
                .withSnappyCompression()
                .build()
            
            print("TCP server created: \(tcpServer.uri)")
            
            // Example 3: Unix socket channel with compression
            print("\nCreating Unix socket channel...")
            let unixChannel = try Psyne.createChannel()
                .unixSocket("/tmp/psyne_example.sock")
                .withBufferSize(1024 * 1024) // 1MB buffer
                .multipleProducersSingleConsumer()
                .withZstdCompression()
                .build()
            
            print("Unix socket channel created: \(unixChannel.uri)")
            
            // Example 4: Custom URI with detailed configuration
            print("\nCreating channel with custom URI...")
            let customChannel = try Psyne.createChannel()
                .withUri("ipc://shared_memory_segment")
                .withBufferSize(4 * 1024 * 1024) // 4MB buffer
                .withMode(.mpmc)
                .withType(.multiType)
                .withCompression(CompressionConfig(
                    type: .zstd,
                    level: 9,
                    minSizeThreshold: 256,
                    enableChecksum: true
                ))
                .build()
            
            print("Custom channel created: \(customChannel.uri)")
            
            // Demonstrate sending messages through different channels
            print("\nTesting message sending...")
            
            // Send through memory channel
            try memoryChannel.sendText("Hello from memory channel!")
            
            // Send through custom channel
            let jsonData = [
                "message": "Hello from custom channel!",
                "timestamp": Date().timeIntervalSince1970,
                "channel": "custom"
            ]
            try customChannel.sendJSON(jsonData)
            
            // Send ML data through unix channel
            let mlData: [Float] = (0..<100).map { Float($0) * 0.1 }
            try unixChannel.sendFloatArray(mlData)
            
            // Receive messages
            print("\nReceiving messages...")
            
            if let message1 = try memoryChannel.receiveMessage(timeout: 1.0) {
                if let textMsg = message1.asTextMessage() {
                    print("Memory channel message: \(textMsg.text)")
                }
            }
            
            if let message2 = try customChannel.receiveMessage(timeout: 1.0) {
                if let jsonMsg = message2.asJSONMessage() {
                    print("Custom channel message: \(jsonMsg.json)")
                }
            }
            
            if let message3 = try unixChannel.receiveMessage(timeout: 1.0) {
                if let floatMsg = message3.asFloatArrayMessage() {
                    print("Unix channel message: \(floatMsg.values.count) float values")
                    print("First few values: \(Array(floatMsg.values.prefix(5)))")
                }
            }
            
            // Show metrics for each channel
            print("\nChannel Metrics:")
            
            try memoryChannel.setMetricsEnabled(true)
            let memoryMetrics = try memoryChannel.getMetrics()
            print("Memory Channel:")
            print("  Messages sent: \(memoryMetrics.messagesSent)")
            print("  Bytes sent: \(memoryMetrics.bytesSent)")
            
            try customChannel.setMetricsEnabled(true)
            let customMetrics = try customChannel.getMetrics()
            print("Custom Channel:")
            print("  Messages sent: \(customMetrics.messagesSent)")
            print("  Bytes sent: \(customMetrics.bytesSent)")
            
            try unixChannel.setMetricsEnabled(true)
            let unixMetrics = try unixChannel.getMetrics()
            print("Unix Channel:")
            print("  Messages sent: \(unixMetrics.messagesSent)")
            print("  Bytes sent: \(unixMetrics.bytesSent)")
            
            print("\nBuilder pattern example completed!")
            
        } catch {
            print("Error: \(error)")
        }
    }
}