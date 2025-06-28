import Foundation
import Psyne

/// Example demonstrating compression features
@main
struct CompressionExample {
    static func main() async {
        do {
            // Initialize the Psyne library
            try Psyne.initialize()
            defer { Psyne.cleanup() }
            
            print("Psyne Swift Compression Example")
            print("Version: \(Psyne.version)")
            print()
            
            // Create channels with different compression algorithms
            let noCompressionChannel = try Psyne.createChannel()
                .memory("no_compression")
                .withBufferSize(1024 * 1024)
                .build()
            
            let lz4Channel = try Psyne.createChannel()
                .memory("lz4_compression")
                .withBufferSize(1024 * 1024)
                .withLZ4Compression()
                .build()
            
            let zstdChannel = try Psyne.createChannel()
                .memory("zstd_compression")
                .withBufferSize(1024 * 1024)
                .withZstdCompression()
                .build()
            
            let snappyChannel = try Psyne.createChannel()
                .memory("snappy_compression")
                .withBufferSize(1024 * 1024)
                .withSnappyCompression()
                .build()
            
            // Custom compression configuration
            let customCompressionChannel = try Psyne.createChannel()
                .memory("custom_compression")
                .withBufferSize(1024 * 1024)
                .withCompression(CompressionConfig(
                    type: .zstd,
                    level: 6,
                    minSizeThreshold: 64,
                    enableChecksum: true
                ))
                .build()
            
            print("Created channels with different compression settings")
            
            // Enable metrics for all channels
            let channels = [
                ("No Compression", noCompressionChannel),
                ("LZ4", lz4Channel),
                ("Zstandard", zstdChannel),
                ("Snappy", snappyChannel),
                ("Custom Zstd", customCompressionChannel)
            ]
            
            for (_, channel) in channels {
                try channel.setMetricsEnabled(true)
            }
            
            // Generate test data - repetitive data that compresses well
            let testString = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 100)
            let testData = testString.data(using: .utf8)!
            
            print("Test data size: \(testData.count) bytes")
            print("Test data content: Repetitive text (compresses well)")
            print()
            
            // Send the same data through all channels
            print("Sending test data through all channels...")
            for (name, channel) in channels {
                try channel.sendBytes(testData, messageType: 1)
                print("Sent through \(name) channel")
            }
            
            print()
            
            // Receive and verify data
            print("Receiving and verifying data...")
            for (name, channel) in channels {
                if let message = try channel.receiveMessage(timeout: 1.0) {
                    let receivedData = try message.allBytes()
                    let isCorrect = receivedData == testData
                    
                    print("\(name):")
                    print("  Received: \(receivedData.count) bytes")
                    print("  Data integrity: \(isCorrect ? "✓ PASS" : "✗ FAIL")")
                    
                    if let receivedString = String(data: receivedData, encoding: .utf8) {
                        let preview = String(receivedString.prefix(50))
                        print("  Preview: \(preview)...")
                    }
                } else {
                    print("\(name): No message received")
                }
            }
            
            print()
            
            // Show compression metrics
            print("Compression Metrics:")
            print(String(repeating: "=", count: 50))
            
            for (name, channel) in channels {
                let metrics = try channel.getMetrics()
                print("\(name):")
                print("  Messages sent: \(metrics.messagesSent)")
                print("  Bytes sent: \(metrics.bytesSent)")
                print("  Messages received: \(metrics.messagesReceived)")
                print("  Bytes received: \(metrics.bytesReceived)")
                
                if metrics.messagesSent > 0 {
                    let compressionRatio = Double(testData.count) / Double(metrics.bytesSent)
                    print("  Compression ratio: \(String(format: "%.2f", compressionRatio)):1")
                    print("  Space saved: \(String(format: "%.1f", (1.0 - 1.0/compressionRatio) * 100))%")
                }
                print()
            }
            
            // Test with different data types
            print("Testing compression with different data types...")
            print(String(repeating: "-", count: 50))
            
            // Random data (doesn't compress well)
            var randomData = Data(count: 1000)
            randomData.withUnsafeMutableBytes { bytes in
                arc4random_buf(bytes.baseAddress!, bytes.count)
            }
            
            // Float array (numerical data)
            let floatArray: [Float] = (0..<250).map { Float($0) * 0.01 }
            
            // JSON data
            let jsonObject = [
                "users": Array(0..<50).map { i in
                    [
                        "id": i,
                        "name": "User \(i)",
                        "email": "user\(i)@example.com",
                        "active": i % 2 == 0
                    ]
                }
            ]
            let jsonData = try JSONSerialization.data(withJSONObject: jsonObject, options: [])
            
            let testCases = [
                ("Random Data", randomData),
                ("JSON Data", jsonData)
            ]
            
            // Test each data type with LZ4 compression
            for (dataType, data) in testCases {
                print("\nTesting \(dataType) (\(data.count) bytes):")
                
                // Reset metrics
                try lz4Channel.resetMetrics()
                
                // Send data
                try lz4Channel.sendBytes(data, messageType: 2)
                
                // Receive data
                if let message = try lz4Channel.receiveMessage(timeout: 1.0) {
                    let receivedData = try message.allBytes()
                    let metrics = try lz4Channel.getMetrics()
                    let compressionRatio = Double(data.count) / Double(metrics.bytesSent)
                    
                    print("  Original size: \(data.count) bytes")
                    print("  Transmitted size: \(metrics.bytesSent) bytes")
                    print("  Compression ratio: \(String(format: "%.2f", compressionRatio)):1")
                    print("  Data integrity: \(receivedData == data ? "✓ PASS" : "✗ FAIL")")
                }
            }
            
            // Test float array
            print("\nTesting Float Array (\(floatArray.count * 4) bytes):")
            try lz4Channel.resetMetrics()
            try lz4Channel.sendFloatArray(floatArray)
            
            if let message = try lz4Channel.receiveMessage(timeout: 1.0) {
                let metrics = try lz4Channel.getMetrics()
                let originalSize = floatArray.count * 4
                let compressionRatio = Double(originalSize) / Double(metrics.bytesSent)
                
                print("  Original size: \(originalSize) bytes")
                print("  Transmitted size: \(metrics.bytesSent) bytes")
                print("  Compression ratio: \(String(format: "%.2f", compressionRatio)):1")
                
                if let floatMsg = message.asFloatArrayMessage() {
                    print("  Data integrity: \(floatMsg.values == floatArray ? "✓ PASS" : "✗ FAIL")")
                }
            }
            
            print("\nCompression example completed!")
            
        } catch {
            print("Error: \(error)")
        }
    }
}