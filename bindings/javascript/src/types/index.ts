/**
 * @file types/index.ts
 * @brief Type definitions for Psyne JavaScript bindings
 * 
 * Provides comprehensive TypeScript type definitions for all Psyne
 * functionality, including enums, interfaces, and utility types.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

/**
 * @enum ChannelMode
 * @brief Channel synchronization modes
 * 
 * Defines the thread-safety characteristics and performance
 * trade-offs for different channel configurations.
 */
export enum ChannelMode {
  /** Single Producer, Single Consumer - Highest performance, lock-free */
  SPSC = 0,
  
  /** Single Producer, Multiple Consumer - One writer, many readers */
  SPMC = 1,
  
  /** Multiple Producer, Single Consumer - Many writers, one reader */
  MPSC = 2,
  
  /** Multiple Producer, Multiple Consumer - Full multi-threading support */
  MPMC = 3
}

/**
 * @enum ChannelType
 * @brief Channel message type support
 */
export enum ChannelType {
  /** Optimized for single message type (no type metadata overhead) */
  SingleType = 0,
  
  /** Supports multiple message types (small metadata overhead) */
  MultiType = 1
}

/**
 * @enum CompressionType
 * @brief Supported compression algorithms
 */
export enum CompressionType {
  /** No compression */
  None = 0,
  
  /** LZ4 - Fast compression/decompression */
  LZ4 = 1,
  
  /** Zstandard - Better compression ratio */
  Zstd = 2,
  
  /** Google Snappy - Balanced speed/ratio */
  Snappy = 3
}

/**
 * @interface CompressionConfig
 * @brief Configuration for compression behavior
 */
export interface CompressionConfig {
  /** Compression algorithm to use */
  type: CompressionType;
  
  /** Compression level (algorithm dependent) */
  level?: number;
  
  /** Don't compress messages smaller than this (bytes) */
  minSizeThreshold?: number;
  
  /** Add checksum for compressed data */
  enableChecksum?: boolean;
}

/**
 * @interface ChannelMetrics
 * @brief Performance metrics for channel debugging
 */
export interface ChannelMetrics {
  /** Number of messages sent */
  messagesSent: number;
  
  /** Total bytes sent */
  bytesSent: number;
  
  /** Number of messages received */
  messagesReceived: number;
  
  /** Total bytes received */
  bytesReceived: number;
  
  /** Times send() blocked waiting for space */
  sendBlocks: number;
  
  /** Times receive() blocked waiting for data */
  receiveBlocks: number;
}

/**
 * @interface MessageType
 * @brief Base interface for all message types
 */
export interface MessageType {
  /** Unique type identifier */
  typeId: number;
  
  /** Human-readable type name */
  type: string;
  
  /** Message size in bytes */
  size: number;
  
  /** Message payload */
  data: any;
}

/**
 * @interface FloatVectorMessage
 * @brief Message containing array of floating-point values
 */
export interface FloatVectorMessage extends MessageType {
  type: 'floatVector';
  typeId: 1;
  data: number[] | Float32Array;
}

/**
 * @interface DoubleMatrixMessage
 * @brief Message containing 2D matrix of double-precision values
 */
export interface DoubleMatrixMessage extends MessageType {
  type: 'doubleMatrix';
  typeId: 2;
  data: {
    rows: number;
    cols: number;
    values: number[] | Float64Array;
  };
}

/**
 * @interface ByteVectorMessage
 * @brief Message containing raw binary data
 */
export interface ByteVectorMessage extends MessageType {
  type: 'byteVector';
  typeId: 10;
  data: Buffer | Uint8Array;
}

/**
 * @interface Vector3fMessage
 * @brief Message containing 3D vector
 */
export interface Vector3fMessage extends MessageType {
  type: 'vector3f';
  typeId: 103;
  data: {
    x: number;
    y: number;
    z: number;
  } | [number, number, number] | Float32Array;
}

/**
 * @interface Matrix4x4fMessage
 * @brief Message containing 4x4 transformation matrix
 */
export interface Matrix4x4fMessage extends MessageType {
  type: 'matrix4x4f';
  typeId: 101;
  data: number[] | Float32Array; // 16 elements in row-major order
}

/**
 * @interface ComplexVectorMessage
 * @brief Message containing array of complex numbers
 */
export interface ComplexVectorMessage extends MessageType {
  type: 'complexVector';
  typeId: 107;
  data: Array<{ real: number; imag: number }> | Float32Array; // Interleaved real/imag
}

/**
 * @interface MLTensorMessage
 * @brief Message containing multi-dimensional tensor for ML
 */
export interface MLTensorMessage extends MessageType {
  type: 'mlTensor';
  typeId: 108;
  data: {
    shape: number[];
    layout: 'NCHW' | 'NHWC' | 'CHW' | 'HWC' | 'Custom';
    values: number[] | Float32Array;
  };
}

/**
 * @interface SparseMatrixMessage
 * @brief Message containing sparse matrix in CSR format
 */
export interface SparseMatrixMessage extends MessageType {
  type: 'sparseMatrix';
  typeId: 109;
  data: {
    rows: number;
    cols: number;
    nnz: number; // Number of non-zero elements
    values: number[] | Float32Array;
    columnIndices: number[] | Uint32Array;
    rowPointers: number[] | Uint32Array;
  };
}

/**
 * @type SupportedMessage
 * @brief Union type of all supported message types
 */
export type SupportedMessage = 
  | FloatVectorMessage
  | DoubleMatrixMessage
  | ByteVectorMessage
  | Vector3fMessage
  | Matrix4x4fMessage
  | ComplexVectorMessage
  | MLTensorMessage
  | SparseMatrixMessage;

/**
 * @interface ChannelEventMap
 * @brief Event map for Channel EventEmitter
 */
export interface ChannelEventMap {
  /** Emitted when a message is received */
  message: (data: MessageType) => void;
  
  /** Emitted when a message is sent */
  sent: (data: { data: any; type?: string; timestamp: Date }) => void;
  
  /** Emitted when a message is received (same as 'message' but for consistency) */
  received: (data: MessageType) => void;
  
  /** Emitted when channel starts listening */
  listening: () => void;
  
  /** Emitted when channel stops listening */
  stopped: () => void;
  
  /** Emitted when channel is closed */
  closed: () => void;
  
  /** Emitted when an error occurs */
  error: (error: Error) => void;
  
  /** Emitted when connection is established (network channels) */
  connected: () => void;
  
  /** Emitted when connection is lost (network channels) */
  disconnected: () => void;
  
  /** Emitted when connection is reconnecting (reliable channels) */
  reconnecting: () => void;
}

/**
 * @interface PerformanceOptions
 * @brief Configuration for performance optimizations
 */
export interface PerformanceOptions {
  /** Enable huge pages for memory allocation */
  enableHugePages?: boolean;
  
  /** Enable CPU affinity pinning */
  enableCpuAffinity?: boolean;
  
  /** Enable memory prefetching */
  enablePrefetching?: boolean;
  
  /** Enable SIMD optimizations */
  enableSimd?: boolean;
  
  /** Cache line size for alignment */
  cacheLineSize?: number;
  
  /** CPU affinity mask */
  cpuAffinityMask?: number[];
}

/**
 * @interface BenchmarkResult
 * @brief Results from performance benchmarking
 */
export interface BenchmarkResult {
  /** Throughput in megabytes per second */
  throughputMbps: number;
  
  /** 50th percentile latency in microseconds */
  latencyUsP50: number;
  
  /** 99th percentile latency in microseconds */
  latencyUsP99: number;
  
  /** 99.9th percentile latency in microseconds */
  latencyUsP999: number;
  
  /** Total number of messages sent */
  messagesSent: number;
  
  /** Benchmark duration in milliseconds */
  durationMs: number;
}

/**
 * @interface NetworkOptions
 * @brief Configuration for network channels
 */
export interface NetworkOptions {
  /** Connection timeout in milliseconds */
  connectTimeout?: number;
  
  /** Read timeout in milliseconds */
  readTimeout?: number;
  
  /** Write timeout in milliseconds */
  writeTimeout?: number;
  
  /** Enable TCP_NODELAY (disable Nagle's algorithm) */
  nodelay?: boolean;
  
  /** TCP receive buffer size */
  receiveBufferSize?: number;
  
  /** TCP send buffer size */
  sendBufferSize?: number;
  
  /** Enable keep-alive */
  keepAlive?: boolean;
  
  /** Keep-alive interval in milliseconds */
  keepAliveInterval?: number;
}

/**
 * @interface MulticastOptions
 * @brief Configuration for UDP multicast channels
 */
export interface MulticastOptions extends NetworkOptions {
  /** Multicast TTL (Time To Live) */
  ttl?: number;
  
  /** Enable multicast loopback */
  loopback?: boolean;
  
  /** Local interface address to bind to */
  interfaceAddress?: string;
  
  /** Join additional multicast groups */
  additionalGroups?: string[];
}

/**
 * @interface UnixSocketOptions
 * @brief Configuration for Unix domain socket channels
 */
export interface UnixSocketOptions {
  /** Socket file permissions (octal) */
  permissions?: number;
  
  /** Remove existing socket file before binding */
  unlinkOnBind?: boolean;
  
  /** Remove socket file on close */
  unlinkOnClose?: boolean;
}

/**
 * @interface WebSocketOptions
 * @brief Configuration for WebSocket channels
 */
export interface WebSocketOptions extends NetworkOptions {
  /** WebSocket subprotocols */
  protocols?: string[];
  
  /** Additional HTTP headers */
  headers?: Record<string, string>;
  
  /** Maximum frame size */
  maxFrameSize?: number;
  
  /** Enable per-message compression */
  compression?: boolean;
}

/**
 * @type ChannelURI
 * @brief Type-safe channel URI strings
 */
export type ChannelURI = 
  | `memory://${string}`      // In-process memory
  | `ipc://${string}`         // Inter-process communication
  | `tcp://${string}:${number}` | `tcp://:${number}` // TCP client/server
  | `unix://${string}`        // Unix domain socket
  | `ws://${string}:${number}` | `ws://:${number}`   // WebSocket client/server
  | `wss://${string}:${number}` | `wss://:${number}` // Secure WebSocket
  | `multicast://${string}:${number}` // UDP multicast
  | `rdma://${string}:${number}`;     // RDMA/InfiniBand

/**
 * @interface TypedArrayConstructor
 * @brief Constructor interface for typed arrays
 */
export interface TypedArrayConstructor {
  new(length: number): TypedArray;
  new(buffer: ArrayBuffer, byteOffset?: number, length?: number): TypedArray;
  BYTES_PER_ELEMENT: number;
}

/**
 * @type TypedArray
 * @brief Union type for all typed array types
 */
export type TypedArray = 
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array
  | BigInt64Array
  | BigUint64Array;

/**
 * @type SendableData
 * @brief Union type for data that can be sent through channels
 */
export type SendableData = 
  | number[]
  | TypedArray
  | Buffer
  | ArrayBuffer
  | SupportedMessage
  | Record<string, any>;

/**
 * @interface ChannelFactory
 * @brief Factory interface for creating channels
 */
export interface ChannelFactory {
  /**
   * Create a channel from URI and options
   */
  create(uri: ChannelURI, options?: any): any;
  
  /**
   * Create a reliable channel
   */
  createReliable(uri: ChannelURI, options?: any, reliabilityOptions?: any): any;
  
  /**
   * Check if URI scheme is supported
   */
  supports(uri: string): boolean;
}

/**
 * @namespace MessageTypes
 * @brief Constants for message type IDs
 */
export namespace MessageTypes {
  export const FLOAT_VECTOR = 1;
  export const DOUBLE_MATRIX = 2;
  export const BYTE_VECTOR = 10;
  export const MATRIX_4X4F = 101;
  export const VECTOR_3F = 103;
  export const INT8_VECTOR = 105;
  export const COMPLEX_VECTOR = 107;
  export const ML_TENSOR = 108;
  export const SPARSE_MATRIX = 109;
}

/**
 * @type EventCallback
 * @brief Generic event callback type
 */
export type EventCallback<T = any> = (data: T) => void;

/**
 * @type AsyncEventCallback
 * @brief Async event callback type
 */
export type AsyncEventCallback<T = any> = (data: T) => Promise<void>;

/**
 * @interface Disposable
 * @brief Interface for objects that need cleanup
 */
export interface Disposable {
  /**
   * Dispose of resources and clean up
   */
  dispose(): void;
  
  /**
   * Check if object has been disposed
   */
  readonly disposed: boolean;
}