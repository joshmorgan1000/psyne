import Foundation
import Psyne

/// Example demonstrating async/await functionality
@main
struct AsyncExample {
    static func main() async {
        do {
            // Initialize the Psyne library
            try Psyne.initialize()
            defer { Psyne.cleanup() }
            
            print("Psyne Swift Async Example")
            print("Version: \(Psyne.version)")
            print()
            
            // Create a memory channel for demonstration
            let channel = try Psyne.createMemoryChannel(name: "async_channel")
            print("Created channel: \(channel.uri)")
            
            // Create a producer task
            let producerTask = Task {
                print("Producer starting...")
                
                for i in 1...10 {
                    let message = "Async message \(i)"
                    try await channel.sendDataAsync(message.data(using: .utf8)!)
                    print("Sent: \(message)")
                    
                    // Small delay between messages
                    try await Task.sleep(nanoseconds: 100_000_000) // 100ms
                }
                
                print("Producer finished")
            }
            
            // Create a consumer task using async sequence
            let consumerTask = Task {
                print("Consumer starting...")
                
                var messageCount = 0
                for await (data, messageType) in channel.dataSequence() {
                    messageCount += 1
                    
                    if let message = String(data: data, encoding: .utf8) {
                        print("Received (\(messageCount)): \(message) [type: \(messageType)]")
                    } else {
                        print("Received (\(messageCount)): \(data.count) bytes [type: \(messageType)]")
                    }
                    
                    // Stop after receiving 10 messages
                    if messageCount >= 10 {
                        break
                    }
                }
                
                print("Consumer finished")
            }
            
            // Wait for both tasks to complete
            try await producerTask.value
            try await consumerTask.value
            
            // Example of using async message sequence
            print("\nUsing async message sequence:")
            
            // Send some messages first
            for i in 1...3 {
                let heartbeat = MessageTypes.HeartbeatMessage(sequenceNumber: UInt64(i))
                try channel.sendCodable(heartbeat)
            }
            
            // Receive using async sequence
            var count = 0
            for await message in channel.messageSequence() {
                count += 1
                
                if let heartbeat = message.asHeartbeatMessage() {
                    print("Heartbeat #\(heartbeat.sequenceNumber) from \(heartbeat.nodeID)")
                }
                
                if count >= 3 {
                    break
                }
            }
            
            print("\nAsync example completed!")
            
        } catch {
            print("Error: \(error)")
        }
    }
}