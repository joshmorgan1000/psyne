import Foundation
import Psyne

/// Basic example showing simple message sending and receiving
@main
struct BasicExample {
    static func main() async {
        do {
            // Initialize the Psyne library
            try Psyne.initialize()
            defer { Psyne.cleanup() }
            
            print("Psyne Swift Basic Example")
            print("Version: \(Psyne.version)")
            print()
            
            // Create a simple memory channel
            let channel = try Psyne.createMemoryChannel(name: "example_channel")
            print("Created channel: \(channel.uri)")
            
            // Send some text data
            let message = "Hello, Psyne from Swift!"
            try channel.sendText(message)
            print("Sent text: \(message)")
            
            // Send some binary data
            let binaryData = "Binary data example".data(using: .utf8)!
            try channel.sendBytes(binaryData, messageType: 42)
            print("Sent binary data: \(binaryData.count) bytes")
            
            // Send a float array
            let floatArray: [Float] = [1.0, 2.5, 3.14, 4.2, 5.0]
            try channel.sendFloatArray(floatArray)
            print("Sent float array: \(floatArray)")
            
            // Receive messages
            print("\nReceiving messages:")
            
            for i in 0..<3 {
                if let receivedMessage = try channel.receiveMessage(timeout: 1.0) {
                    print("Message \(i + 1):")
                    print("  Type ID: \(receivedMessage.messageType)")
                    print("  Size: \(receivedMessage.size) bytes")
                    
                    // Try to decode as different message types
                    if let textMsg = receivedMessage.asTextMessage() {
                        print("  Content (Text): \(textMsg.text)")
                    } else if let floatMsg = receivedMessage.asFloatArrayMessage() {
                        print("  Content (Float Array): \(floatMsg.values)")
                    } else {
                        // Read as raw bytes
                        let data = try receivedMessage.allBytes()
                        if let string = String(data: data, encoding: .utf8) {
                            print("  Content (Raw String): \(string)")
                        } else {
                            print("  Content (Raw Bytes): \(data.count) bytes")
                        }
                    }
                    print()
                } else {
                    print("No message received (timeout)")
                }
            }
            
            // Get channel metrics
            try channel.setMetricsEnabled(true)
            let metrics = try channel.getMetrics()
            print("Channel Metrics:")
            print(metrics)
            
        } catch {
            print("Error: \(error)")
        }
    }
}