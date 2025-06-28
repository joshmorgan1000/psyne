/**
 * @file index.ts
 * @brief Main entry point for Psyne JavaScript bindings
 * 
 * Provides a comprehensive, high-level API for working with Psyne messaging
 * from JavaScript and TypeScript applications. Includes channel management,
 * message handling, performance monitoring, and utility functions.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 * 
 * @example Basic Usage
 * ```typescript
 * import { Psyne, ChannelMode } from 'psyne';
 * 
 * // Initialize Psyne
 * const psyne = new Psyne();
 * 
 * // Create a high-performance channel
 * const channel = psyne.createChannel('memory://buffer1', {
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
 * // Listen for incoming messages
 * channel.on('message', (data) => {
 *   console.log('Received:', data);
 * });
 * ```
 * 
 * @example Builder Pattern
 * ```typescript
 * import { Channel, ChannelMode, CompressionType } from 'psyne';
 * 
 * const channel = Channel.builder()
 *   .uri('tcp://localhost:8080')
 *   .mode(ChannelMode.MPSC)
 *   .bufferSize(2048 * 1024)
 *   .enableMetrics()
 *   .compression({
 *     type: CompressionType.LZ4,
 *     level: 3
 *   })
 *   .build();
 * ```
 * 
 * @example Message Types
 * ```typescript
 * import { FloatVector, DoubleMatrix, ByteVector } from 'psyne/messages';
 * 
 * // Send typed messages
 * await channel.send(new FloatVector([1.0, 2.0, 3.0]));
 * await channel.send(new DoubleMatrix(2, 2, [1, 2, 3, 4]));
 * await channel.send(new ByteVector(Buffer.from('hello')));
 * ```
 */

// Core exports
export { Channel, ChannelBuilder } from './core/channel';

// Type definitions
export * from './types';
export * from './types/metrics';

// Error classes
export * from './errors';

// Message types
export * from './messages';

// Utilities
export * from './utils';

// Version information
const native = require('../build/Release/psyne_native.node');

/**
 * @brief Get the version of the Psyne library
 * @returns Version string in format "major.minor.patch"
 */
export function getVersion(): string {
  return native.getVersion();
}

/**
 * @brief Print the Psyne banner to console
 */
export function printBanner(): void {
  native.printBanner();
}

/**
 * @class Psyne
 * @brief Main class for managing Psyne functionality
 * 
 * Provides a centralized interface for creating channels, managing
 * performance settings, and accessing library-wide functionality.
 * 
 * @example
 * ```typescript
 * import { Psyne, ChannelMode } from 'psyne';
 * 
 * const psyne = new Psyne({
 *   enablePerformanceOptimizations: true,
 *   defaultBufferSize: 2048 * 1024
 * });
 * 
 * const channel = psyne.createChannel('memory://buffer1', {
 *   mode: ChannelMode.SPSC
 * });
 * ```
 */
export class Psyne {
  private readonly options: PsyneOptions;
  private readonly channels = new Map<string, Channel>();

  /**
   * @brief Create a new Psyne instance
   * @param options Global configuration options
   */
  constructor(options: PsyneOptions = {}) {
    this.options = {
      enablePerformanceOptimizations: false,
      defaultBufferSize: 1024 * 1024,
      enableMetricsByDefault: false,
      ...options
    };

    // Apply global performance optimizations if enabled
    if (this.options.enablePerformanceOptimizations) {
      this.enablePerformanceOptimizations();
    }
  }

  /**
   * @brief Create a new channel
   * @param uri Channel URI
   * @param options Channel-specific options
   * @returns New Channel instance
   */
  createChannel(uri: string, options: import('./core/channel').ChannelOptions = {}): Channel {
    // Apply default options
    const channelOptions = {
      bufferSize: this.options.defaultBufferSize,
      enableMetrics: this.options.enableMetricsByDefault,
      ...options
    };

    const channel = new Channel(uri, channelOptions);
    
    // Track channel for management
    this.channels.set(uri, channel);
    
    // Clean up when channel is closed
    channel.once('closed', () => {
      this.channels.delete(uri);
    });

    return channel;
  }

  /**
   * @brief Create a reliable channel with error recovery
   * @param uri Channel URI
   * @param options Channel options
   * @param reliabilityOptions Reliability configuration
   * @returns New reliable Channel instance
   */
  createReliableChannel(
    uri: string, 
    options: import('./core/channel').ChannelOptions = {},
    reliabilityOptions: import('./core/channel').ReliabilityOptions = {}
  ): Channel {
    const channelOptions = {
      bufferSize: this.options.defaultBufferSize,
      enableMetrics: this.options.enableMetricsByDefault,
      ...options
    };

    const channel = Channel.createReliable(uri, channelOptions, reliabilityOptions);
    this.channels.set(uri, channel);
    
    channel.once('closed', () => {
      this.channels.delete(uri);
    });

    return channel;
  }

  /**
   * @brief Get an existing channel by URI
   * @param uri Channel URI
   * @returns Channel instance or undefined if not found
   */
  getChannel(uri: string): Channel | undefined {
    return this.channels.get(uri);
  }

  /**
   * @brief Get all active channels
   * @returns Array of all active channels
   */
  getChannels(): Channel[] {
    return Array.from(this.channels.values());
  }

  /**
   * @brief Close all channels and clean up
   */
  closeAll(): void {
    for (const channel of this.channels.values()) {
      channel.close();
    }
    this.channels.clear();
  }

  /**
   * @brief Enable global performance optimizations
   */
  enablePerformanceOptimizations(): void {
    // This would call native performance optimization functions
    // For now, this is a placeholder
  }

  /**
   * @brief Disable global performance optimizations
   */
  disablePerformanceOptimizations(): void {
    // This would disable native performance optimizations
    // For now, this is a placeholder
  }

  /**
   * @brief Get performance summary
   * @returns Human-readable performance summary
   */
  getPerformanceSummary(): string {
    // This would return native performance summary
    return 'Performance optimizations: ' + 
           (this.options.enablePerformanceOptimizations ? 'enabled' : 'disabled');
  }

  /**
   * @brief Run performance benchmarks
   * @param channelUri URI of channel to benchmark
   * @param messageSize Size of test messages
   * @param numMessages Number of messages to send
   * @returns Benchmark results
   */
  async runBenchmark(
    channelUri: string, 
    messageSize: number = 1024, 
    numMessages: number = 10000
  ): Promise<import('./types').BenchmarkResult> {
    const channel = this.getChannel(channelUri);
    if (!channel) {
      throw new Error(`Channel not found: ${channelUri}`);
    }

    const startTime = Date.now();
    const testData = new Uint8Array(messageSize);
    
    // Send messages
    for (let i = 0; i < numMessages; i++) {
      await channel.send(testData);
    }

    const duration = Date.now() - startTime;
    const totalBytes = messageSize * numMessages;
    const throughputMbps = (totalBytes / (1024 * 1024)) / (duration / 1000);

    return {
      throughputMbps,
      latencyUsP50: 0, // Would be calculated from actual measurements
      latencyUsP99: 0,
      latencyUsP999: 0,
      messagesSent: numMessages,
      durationMs: duration
    };
  }

  /**
   * @brief Get library version information
   * @returns Version string
   */
  static getVersion(): string {
    return getVersion();
  }

  /**
   * @brief Print library banner
   */
  static printBanner(): void {
    printBanner();
  }
}

/**
 * @interface PsyneOptions
 * @brief Global configuration options for Psyne
 */
export interface PsyneOptions {
  /** Enable performance optimizations globally */
  enablePerformanceOptimizations?: boolean;
  
  /** Default buffer size for new channels */
  defaultBufferSize?: number;
  
  /** Enable metrics by default for new channels */
  enableMetricsByDefault?: boolean;
  
  /** Maximum number of channels to track */
  maxChannels?: number;
  
  /** Global compression settings */
  defaultCompression?: import('./types').CompressionConfig;
}

/**
 * @brief Create a channel using the fluent builder pattern
 * @param uri Channel URI
 * @returns ChannelBuilder for method chaining
 * 
 * @example
 * ```typescript
 * import { createChannel, ChannelMode } from 'psyne';
 * 
 * const channel = createChannel('tcp://localhost:8080')
 *   .mode(ChannelMode.MPSC)
 *   .bufferSize(1024 * 1024)
 *   .enableMetrics()
 *   .build();
 * ```
 */
export function createChannel(uri: string): import('./core/channel').ChannelBuilder {
  return Channel.builder().uri(uri);
}

/**
 * @brief Create a simple channel with default options
 * @param uri Channel URI
 * @param options Optional channel configuration
 * @returns New Channel instance
 * 
 * @example
 * ```typescript
 * import { channel } from 'psyne';
 * 
 * const ch = channel('memory://buffer1');
 * await ch.send([1, 2, 3]);
 * ```
 */
export function channel(uri: string, options?: import('./core/channel').ChannelOptions): Channel {
  return new Channel(uri, options);
}

// Default export
export default Psyne;

// Re-export commonly used items for convenience
export {
  ChannelMode,
  ChannelType,
  CompressionType
} from './types';

export {
  PsyneError,
  ChannelError,
  MessageError
} from './errors';