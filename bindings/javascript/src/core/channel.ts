/**
 * @file channel.ts
 * @brief TypeScript wrapper for Psyne Channel functionality
 * 
 * Provides a high-level, Promise-based API for working with Psyne channels,
 * including EventEmitter integration for message streams.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

import { EventEmitter } from 'events';
import { ChannelMode, ChannelType, CompressionConfig, CompressionType } from '../types';
import { ChannelMetrics } from '../types/metrics';
import { PsyneError, ChannelClosedError, MessageError } from '../errors';

// Import native addon
const native = require('../../build/Release/psyne_native.node');

/**
 * @interface ChannelOptions
 * @brief Configuration options for creating a channel
 */
export interface ChannelOptions {
  /** Size of the internal buffer in bytes (default: 1MB) */
  bufferSize?: number;
  
  /** Channel synchronization mode (default: SPSC) */
  mode?: ChannelMode;
  
  /** Single or multi-type channel (default: MultiType) */
  type?: ChannelType;
  
  /** Enable metrics collection (default: false) */
  enableMetrics?: boolean;
  
  /** Compression configuration (optional) */
  compression?: CompressionConfig;
}

/**
 * @interface ReliabilityOptions
 * @brief Configuration for reliable channel features
 */
export interface ReliabilityOptions {
  /** Enable acknowledgment system (default: true) */
  enableAcknowledgments?: boolean;
  
  /** Enable automatic retries (default: true) */
  enableRetries?: boolean;
  
  /** Enable heartbeat monitoring (default: true) */
  enableHeartbeat?: boolean;
  
  /** Maximum number of retries (default: 3) */
  maxRetries?: number;
  
  /** Acknowledgment timeout in milliseconds (default: 1000) */
  ackTimeout?: number;
  
  /** Retry delay in milliseconds (default: 100) */
  retryDelay?: number;
  
  /** Heartbeat interval in milliseconds (default: 5000) */
  heartbeatInterval?: number;
}

/**
 * @interface MessageData
 * @brief Represents a received message
 */
export interface MessageData {
  /** Message type identifier */
  typeId: number;
  
  /** Message type name */
  type: string;
  
  /** Raw message size in bytes */
  size: number;
  
  /** Message payload data */
  data: any;
  
  /** Timestamp when message was received */
  timestamp?: Date;
}

/**
 * @class Channel
 * @brief High-level JavaScript wrapper for Psyne channels
 * 
 * Provides a Promise-based API with EventEmitter integration for
 * high-performance messaging. Supports zero-copy operations when
 * working with TypedArrays and Buffers.
 * 
 * @example
 * ```typescript
 * import { Channel, ChannelMode } from 'psyne';
 * 
 * // Create a channel
 * const channel = new Channel('memory://buffer1', {
 *   mode: ChannelMode.SPSC,
 *   enableMetrics: true
 * });
 * 
 * // Send data
 * await channel.send([1.0, 2.0, 3.0]);
 * 
 * // Receive data
 * const message = await channel.receive();
 * 
 * // Listen for messages
 * channel.on('message', (data) => {
 *   console.log('Received:', data);
 * });
 * 
 * channel.startListening();
 * ```
 */
export class Channel extends EventEmitter {
  private readonly nativeChannel: any;
  private listening = false;
  private _closed = false;

  /**
   * @brief Create a new channel
   * @param uri Channel URI (e.g., "memory://buffer1", "tcp://localhost:8080")
   * @param options Channel configuration options
   * @throws {PsyneError} If channel creation fails
   */
  constructor(uri: string, options: ChannelOptions = {}) {
    super();
    
    const {
      bufferSize = 1024 * 1024,
      mode = ChannelMode.SPSC,
      type = ChannelType.MultiType,
      enableMetrics = false,
      compression
    } = options;

    try {
      this.nativeChannel = native.createChannel(
        uri,
        bufferSize,
        mode,
        type,
        enableMetrics,
        compression
      );
    } catch (error) {
      throw new PsyneError(`Failed to create channel: ${error.message}`, error);
    }
  }

  /**
   * @brief Create a reliable channel with error recovery features
   * @param uri Channel URI
   * @param options Channel configuration options
   * @param reliabilityOptions Reliability feature configuration
   * @returns New Channel instance with reliability features
   */
  static createReliable(
    uri: string, 
    options: ChannelOptions = {},
    reliabilityOptions: ReliabilityOptions = {}
  ): Channel {
    const channel = Object.create(Channel.prototype);
    
    const {
      bufferSize = 1024 * 1024,
      mode = ChannelMode.SPSC
    } = options;

    try {
      channel.nativeChannel = native.createReliableChannel(
        uri,
        bufferSize,
        mode,
        reliabilityOptions
      );
      
      EventEmitter.call(channel);
      channel.listening = false;
      channel._closed = false;
      
      return channel;
    } catch (error) {
      throw new PsyneError(`Failed to create reliable channel: ${error.message}`, error);
    }
  }

  /**
   * @brief Send a message through the channel
   * @param data Message data (Array, TypedArray, Buffer, or structured object)
   * @param type Optional message type for structured data
   * @returns Promise that resolves when message is sent
   * @throws {ChannelClosedError} If channel is closed
   * @throws {MessageError} If message format is invalid
   * 
   * @example
   * ```typescript
   * // Send array of numbers
   * await channel.send([1.0, 2.0, 3.0]);
   * 
   * // Send typed array
   * const data = new Float32Array([1.0, 2.0, 3.0]);
   * await channel.send(data);
   * 
   * // Send structured message
   * await channel.send({
   *   type: 'doubleMatrix',
   *   data: {
   *     rows: 2,
   *     cols: 2,
   *     values: [1.0, 2.0, 3.0, 4.0]
   *   }
   * });
   * ```
   */
  async send(data: any, type?: string): Promise<void> {
    if (this._closed) {
      throw new ChannelClosedError('Cannot send on closed channel');
    }

    try {
      await this.nativeChannel.send(data, type);
      this.emit('sent', { data, type, timestamp: new Date() });
    } catch (error) {
      throw new MessageError(`Failed to send message: ${error.message}`, error);
    }
  }

  /**
   * @brief Receive a message from the channel
   * @param timeout Maximum time to wait in milliseconds (0 for non-blocking)
   * @returns Promise resolving to received message or null if none available
   * @throws {ChannelClosedError} If channel is closed
   * 
   * @example
   * ```typescript
   * // Non-blocking receive
   * const message = await channel.receive();
   * 
   * // Blocking receive with timeout
   * const message = await channel.receive(5000);
   * ```
   */
  async receive(timeout: number = 0): Promise<MessageData | null> {
    if (this._closed) {
      throw new ChannelClosedError('Cannot receive on closed channel');
    }

    try {
      const result = await this.nativeChannel.receive(timeout);
      
      if (result) {
        const messageData: MessageData = {
          ...result,
          timestamp: new Date()
        };
        
        this.emit('received', messageData);
        return messageData;
      }
      
      return null;
    } catch (error) {
      throw new PsyneError(`Failed to receive message: ${error.message}`, error);
    }
  }

  /**
   * @brief Start listening for incoming messages
   * @throws {ChannelClosedError} If channel is closed
   * 
   * Starts a background listener that emits 'message' events for
   * incoming messages. Use stopListening() to stop.
   */
  startListening(): void {
    if (this._closed) {
      throw new ChannelClosedError('Cannot listen on closed channel');
    }

    if (this.listening) {
      return; // Already listening
    }

    this.listening = true;
    
    this.nativeChannel.listen((message: any, error?: any) => {
      if (error) {
        this.emit('error', new PsyneError('Listener error', error));
        return;
      }

      if (message) {
        const messageData: MessageData = {
          ...message,
          timestamp: new Date()
        };
        
        this.emit('message', messageData);
        this.emit('received', messageData);
      }
    });

    this.emit('listening');
  }

  /**
   * @brief Stop listening for incoming messages
   */
  stopListening(): void {
    if (!this.listening) {
      return;
    }

    this.listening = false;
    this.nativeChannel.stopListening();
    this.emit('stopped');
  }

  /**
   * @brief Close the channel and clean up resources
   */
  close(): void {
    if (this._closed) {
      return;
    }

    this.stopListening();
    this.nativeChannel.stop();
    this._closed = true;
    
    this.emit('closed');
    this.removeAllListeners();
  }

  /**
   * @brief Check if the channel is closed
   * @returns True if channel is closed
   */
  get closed(): boolean {
    return this._closed || this.nativeChannel.isStopped();
  }

  /**
   * @brief Check if channel is currently listening
   * @returns True if listening for messages
   */
  get isListening(): boolean {
    return this.listening;
  }

  /**
   * @brief Get the channel URI
   * @returns Channel URI string
   */
  get uri(): string {
    return this.nativeChannel.getUri();
  }

  /**
   * @brief Get the channel mode
   * @returns Channel mode enum value
   */
  get mode(): ChannelMode {
    return this.nativeChannel.getMode();
  }

  /**
   * @brief Get the channel type
   * @returns Channel type enum value
   */
  get type(): ChannelType {
    return this.nativeChannel.getType();
  }

  /**
   * @brief Check if metrics are enabled
   * @returns True if metrics collection is enabled
   */
  get hasMetrics(): boolean {
    return this.nativeChannel.hasMetrics();
  }

  /**
   * @brief Get current channel metrics
   * @returns Metrics object or null if metrics disabled
   * 
   * @example
   * ```typescript
   * const metrics = channel.getMetrics();
   * if (metrics) {
   *   console.log(`Messages sent: ${metrics.messagesSent}`);
   *   console.log(`Throughput: ${metrics.bytesSent / 1024} KB`);
   * }
   * ```
   */
  getMetrics(): ChannelMetrics | null {
    if (!this.hasMetrics) {
      return null;
    }

    return this.nativeChannel.getMetrics();
  }

  /**
   * @brief Reset metrics counters to zero
   */
  resetMetrics(): void {
    this.nativeChannel.resetMetrics();
  }

  /**
   * @brief Create a fluent builder for channel configuration
   * @returns ChannelBuilder instance
   */
  static builder(): ChannelBuilder {
    return new ChannelBuilder();
  }
}

/**
 * @class ChannelBuilder
 * @brief Fluent builder for channel creation
 * 
 * Provides a fluent API for configuring and creating channels
 * with method chaining.
 * 
 * @example
 * ```typescript
 * const channel = Channel.builder()
 *   .uri('tcp://localhost:8080')
 *   .bufferSize(2048 * 1024)
 *   .mode(ChannelMode.MPSC)
 *   .enableMetrics()
 *   .compression({
 *     type: CompressionType.LZ4,
 *     level: 3
 *   })
 *   .build();
 * ```
 */
export class ChannelBuilder {
  private options: ChannelOptions = {};
  private channelUri?: string;

  /**
   * @brief Set the channel URI
   * @param uri Channel URI
   * @returns This builder for chaining
   */
  uri(uri: string): this {
    this.channelUri = uri;
    return this;
  }

  /**
   * @brief Set the buffer size
   * @param size Buffer size in bytes
   * @returns This builder for chaining
   */
  bufferSize(size: number): this {
    this.options.bufferSize = size;
    return this;
  }

  /**
   * @brief Set the channel mode
   * @param mode Channel mode
   * @returns This builder for chaining
   */
  mode(mode: ChannelMode): this {
    this.options.mode = mode;
    return this;
  }

  /**
   * @brief Set the channel type
   * @param type Channel type
   * @returns This builder for chaining
   */
  type(type: ChannelType): this {
    this.options.type = type;
    return this;
  }

  /**
   * @brief Enable metrics collection
   * @returns This builder for chaining
   */
  enableMetrics(): this {
    this.options.enableMetrics = true;
    return this;
  }

  /**
   * @brief Configure compression
   * @param config Compression configuration
   * @returns This builder for chaining
   */
  compression(config: CompressionConfig): this {
    this.options.compression = config;
    return this;
  }

  /**
   * @brief Build the channel with configured options
   * @returns New Channel instance
   * @throws {PsyneError} If URI not set or channel creation fails
   */
  build(): Channel {
    if (!this.channelUri) {
      throw new PsyneError('Channel URI must be set');
    }

    return new Channel(this.channelUri, this.options);
  }

  /**
   * @brief Build a reliable channel with configured options
   * @param reliabilityOptions Reliability configuration
   * @returns New Channel instance with reliability features
   */
  buildReliable(reliabilityOptions: ReliabilityOptions = {}): Channel {
    if (!this.channelUri) {
      throw new PsyneError('Channel URI must be set');
    }

    return Channel.createReliable(this.channelUri, this.options, reliabilityOptions);
  }
}