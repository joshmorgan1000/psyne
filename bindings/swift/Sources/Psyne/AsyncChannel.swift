import Foundation
import CPsyne

/// Extension to Channel providing async/await support for Swift concurrency
extension Channel {
    
    /// Asynchronously send data through the channel
    /// - Parameters:
    ///   - data: The data to send
    ///   - messageType: The message type identifier
    /// - Throws: `PsyneError` if sending fails
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func sendDataAsync<T>(_ data: T, messageType: UInt32 = 0) async throws {
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                do {
                    try self.sendData(data, messageType: messageType)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Asynchronously send bytes through the channel
    /// - Parameters:
    ///   - bytes: The bytes to send
    ///   - messageType: The message type identifier
    /// - Throws: `PsyneError` if sending fails
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func sendBytesAsync(_ bytes: Data, messageType: UInt32 = 0) async throws {
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                do {
                    try self.sendBytes(bytes, messageType: messageType)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Asynchronously receive data from the channel
    /// - Parameters:
    ///   - maxSize: Maximum size of data to receive
    ///   - timeout: Timeout in seconds (nil for no timeout)
    /// - Returns: A tuple containing the received data and message type
    /// - Throws: `PsyneError` if receiving fails
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func receiveDataAsync(
        maxSize: Int = 1024 * 1024,
        timeout: TimeInterval? = nil
    ) async throws -> (data: Data, messageType: UInt32) {
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                do {
                    let result = try self.receiveData(maxSize: maxSize, timeout: timeout)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Asynchronously receive a message from the channel
    /// - Parameter timeout: Timeout in seconds (nil for no timeout)
    /// - Returns: A received `Message` or nil if no message available
    /// - Throws: `PsyneError` if receiving fails
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func receiveMessageAsync(timeout: TimeInterval? = nil) async throws -> ReceivedMessage? {
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                do {
                    let message = try self.receiveMessage(timeout: timeout)
                    continuation.resume(returning: message)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Create an async sequence for receiving messages
    /// - Parameter bufferPolicy: The buffer policy for the async sequence
    /// - Returns: An async sequence of received messages
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func messageSequence(
        bufferPolicy: AsyncChannel<ReceivedMessage>.BufferPolicy = .unbounded
    ) -> AsyncChannel<ReceivedMessage> {
        return AsyncChannel(bufferPolicy: bufferPolicy) { channel in
            while !self.isStopped {
                do {
                    if let message = try self.receiveMessage(timeout: 0.1) {
                        await channel.send(message)
                    }
                } catch {
                    // Log error but continue trying to receive
                    if case PsyneError.timeout = error {
                        // Timeout is expected, continue
                        continue
                    } else if case PsyneError.noMessage = error {
                        // No message is expected, continue
                        continue
                    } else {
                        // Other errors should break the loop
                        break
                    }
                }
            }
            channel.finish()
        }
    }
    
    /// Create an async sequence for receiving data
    /// - Parameters:
    ///   - maxSize: Maximum size of data to receive
    ///   - bufferPolicy: The buffer policy for the async sequence
    /// - Returns: An async sequence of received data and message types
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func dataSequence(
        maxSize: Int = 1024 * 1024,
        bufferPolicy: AsyncChannel<(data: Data, messageType: UInt32)>.BufferPolicy = .unbounded
    ) -> AsyncChannel<(data: Data, messageType: UInt32)> {
        return AsyncChannel(bufferPolicy: bufferPolicy) { channel in
            while !self.isStopped {
                do {
                    let result = try self.receiveData(maxSize: maxSize, timeout: 0.1)
                    await channel.send(result)
                } catch {
                    // Log error but continue trying to receive
                    if case PsyneError.timeout = error {
                        // Timeout is expected, continue
                        continue
                    } else if case PsyneError.noMessage = error {
                        // No message is expected, continue
                        continue
                    } else {
                        // Other errors should break the loop
                        break
                    }
                }
            }
            channel.finish()
        }
    }
}

/// An async sequence for streaming data from channels
@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
public final class AsyncChannel<Element>: AsyncSequence {
    public typealias AsyncIterator = Iterator
    
    /// Buffer policy for the async channel
    public enum BufferPolicy {
        case unbounded
        case bounded(Int)
    }
    
    private let bufferPolicy: BufferPolicy
    private let producer: (AsyncChannel<Element>) async -> Void
    private var continuation: AsyncStream<Element>.Continuation?
    private var stream: AsyncStream<Element>?
    
    /// Create an async channel with a producer function
    /// - Parameters:
    ///   - bufferPolicy: The buffer policy
    ///   - producer: The producer function that generates elements
    public init(
        bufferPolicy: BufferPolicy = .unbounded,
        producer: @escaping (AsyncChannel<Element>) async -> Void
    ) {
        self.bufferPolicy = bufferPolicy
        self.producer = producer
    }
    
    /// Send an element to the async channel
    /// - Parameter element: The element to send
    public func send(_ element: Element) async {
        continuation?.yield(element)
    }
    
    /// Finish the async channel (no more elements)
    public func finish() {
        continuation?.finish()
    }
    
    /// Create an async iterator
    /// - Returns: An async iterator for this sequence
    public func makeAsyncIterator() -> Iterator {
        let (stream, continuation) = AsyncStream<Element>.makeStream(
            bufferingPolicy: bufferPolicy == .unbounded ? 
                .unbounded : 
                .bufferingNewest(bufferPolicy.boundedSize)
        )
        
        self.stream = stream
        self.continuation = continuation
        
        // Start the producer
        Task {
            await producer(self)
        }
        
        return Iterator(stream: stream)
    }
    
    /// Iterator for the async channel
    public struct Iterator: AsyncIteratorProtocol {
        private var streamIterator: AsyncStream<Element>.AsyncIterator
        
        fileprivate init(stream: AsyncStream<Element>) {
            self.streamIterator = stream.makeAsyncIterator()
        }
        
        /// Get the next element
        /// - Returns: The next element or nil if finished
        public mutating func next() async -> Element? {
            return await streamIterator.next()
        }
    }
}

private extension AsyncChannel.BufferPolicy {
    var boundedSize: Int {
        switch self {
        case .bounded(let size):
            return size
        case .unbounded:
            return Int.max
        }
    }
}