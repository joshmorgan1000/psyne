import Foundation
import Psyne

/// Example demonstrating producer-consumer pattern with multiple threads
@main
struct ProducerConsumerExample {
    static func main() async {
        do {
            // Initialize the Psyne library
            try Psyne.initialize()
            defer { Psyne.cleanup() }
            
            print("Psyne Swift Producer-Consumer Example")
            print("Version: \(Psyne.version)")
            print()
            
            // Create a channel optimized for multiple producers and single consumer
            let channel = try Psyne.createChannel()
                .memory("producer_consumer")
                .withBufferSize(4 * 1024 * 1024) // 4MB buffer
                .multipleProducersSingleConsumer()
                .withLZ4Compression()
                .build()
            
            print("Created channel: \(channel.uri)")
            try channel.setMetricsEnabled(true)
            
            let numberOfProducers = 3
            let messagesPerProducer = 10
            let totalMessages = numberOfProducers * messagesPerProducer
            
            print("Starting \(numberOfProducers) producers, each sending \(messagesPerProducer) messages")
            print("Total expected messages: \(totalMessages)")
            print()
            
            // Create producer tasks
            let producerTasks = (0..<numberOfProducers).map { producerID in
                Task {
                    print("Producer \(producerID) starting...")
                    
                    for messageIndex in 0..<messagesPerProducer {
                        let messageData = ProducerMessage(
                            producerID: producerID,
                            messageIndex: messageIndex,
                            timestamp: Date(),
                            payload: "Data from producer \(producerID), message \(messageIndex)"
                        )
                        
                        try await channel.sendCodableAsync(messageData)
                        print("Producer \(producerID) sent message \(messageIndex)")
                        
                        // Random delay between messages (10-100ms)
                        let delayMs = Int.random(in: 10...100)
                        try await Task.sleep(nanoseconds: UInt64(delayMs * 1_000_000))
                    }
                    
                    print("Producer \(producerID) finished")
                }
            }
            
            // Create consumer task
            let consumerTask = Task {
                print("Consumer starting...")
                
                var receivedMessages: [ProducerMessage] = []
                var messagesByProducer: [Int: Int] = [:]
                
                while receivedMessages.count < totalMessages {
                    if let message = try await channel.receiveMessageAsync(timeout: 1.0) {
                        if let producerMessage = try? message.decodeCodable(as: ProducerMessage.self) {
                            receivedMessages.append(producerMessage)
                            messagesByProducer[producerMessage.producerID, default: 0] += 1
                            
                            print("Consumer received: Producer \(producerMessage.producerID), Message \(producerMessage.messageIndex)")
                            print("  Payload: \(producerMessage.payload)")
                            print("  Timestamp: \(producerMessage.timestamp)")
                            print("  Total received: \(receivedMessages.count)/\(totalMessages)")
                            print()
                        }
                    } else {
                        print("Consumer timeout - continuing...")
                    }
                }
                
                print("Consumer finished!")
                print("Messages received by producer:")
                for (producerID, count) in messagesByProducer.sorted(by: { $0.key < $1.key }) {
                    print("  Producer \(producerID): \(count) messages")
                }
                
                return receivedMessages
            }
            
            // Wait for all producers to complete
            print("Waiting for producers to complete...")
            for task in producerTasks {
                try await task.value
            }
            print("All producers completed!")
            
            // Wait for consumer to receive all messages
            print("Waiting for consumer to finish...")
            let receivedMessages = try await consumerTask.value
            
            // Analyze results
            print("\nResults Analysis:")
            print(String(repeating: "=", count: 50))
            
            let metrics = try channel.getMetrics()
            print("Channel Metrics:")
            print("  Messages sent: \(metrics.messagesSent)")
            print("  Bytes sent: \(metrics.bytesSent)")
            print("  Messages received: \(metrics.messagesReceived)")
            print("  Bytes received: \(metrics.bytesReceived)")
            print("  Send blocks: \(metrics.sendBlocks)")
            print("  Receive blocks: \(metrics.receiveBlocks)")
            print("  Average message size: \(String(format: "%.1f", metrics.averageSentMessageSize)) bytes")
            
            // Verify message ordering and completeness
            print("\nMessage Verification:")
            var messagesByProducer: [Int: [ProducerMessage]] = [:]
            
            for message in receivedMessages {
                messagesByProducer[message.producerID, default: []].append(message)
            }
            
            var allMessagesValid = true
            
            for producerID in 0..<numberOfProducers {
                let producerMessages = messagesByProducer[producerID] ?? []
                let sortedMessages = producerMessages.sorted { $0.messageIndex < $1.messageIndex }
                
                print("Producer \(producerID):")
                print("  Messages received: \(producerMessages.count)/\(messagesPerProducer)")
                
                // Check for message ordering within each producer
                var isOrdered = true
                for (index, message) in sortedMessages.enumerated() {
                    if message.messageIndex != index {
                        isOrdered = false
                        break
                    }
                }
                
                print("  Message ordering: \(isOrdered ? "✓ PASS" : "✗ FAIL")")
                print("  All messages received: \(producerMessages.count == messagesPerProducer ? "✓ PASS" : "✗ FAIL")")
                
                if producerMessages.count != messagesPerProducer || !isOrdered {
                    allMessagesValid = false
                }
            }
            
            print("\nOverall Result: \(allMessagesValid ? "✓ ALL TESTS PASSED" : "✗ SOME TESTS FAILED")")
            
            // Performance metrics
            let totalDataSent = metrics.bytesSent
            let totalDataReceived = metrics.bytesReceived
            
            print("\nPerformance Summary:")
            print("  Total data transmitted: \(totalDataSent) bytes")
            print("  Total data received: \(totalDataReceived) bytes")
            print("  Data integrity: \(totalDataSent == totalDataReceived ? "✓ PASS" : "✗ FAIL")")
            
            if let firstMessage = receivedMessages.first,
               let lastMessage = receivedMessages.last {
                let duration = lastMessage.timestamp.timeIntervalSince(firstMessage.timestamp)
                let throughput = Double(totalDataReceived) / duration / 1024 / 1024 // MB/s
                
                print("  Processing duration: \(String(format: "%.3f", duration)) seconds")
                print("  Throughput: \(String(format: "%.2f", throughput)) MB/s")
                print("  Messages per second: \(String(format: "%.1f", Double(totalMessages) / duration))")
            }
            
            print("\nProducer-Consumer example completed!")
            
        } catch {
            print("Error: \(error)")
        }
    }
}

/// Message structure for producer-consumer communication
struct ProducerMessage: Codable, MessageType {
    static let messageTypeID: UInt32 = 200
    
    let producerID: Int
    let messageIndex: Int
    let timestamp: Date
    let payload: String
}

/// Extension to add async Codable sending
extension Channel {
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    func sendCodableAsync<T: Codable>(_ object: T) async throws {
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                do {
                    try self.sendCodable(object)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}