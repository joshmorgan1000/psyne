/**
 * @file types/metrics.ts
 * @brief Type definitions for Psyne metrics and performance monitoring
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

/**
 * @interface ChannelMetrics
 * @brief Performance metrics for channel operations
 */
export interface ChannelMetrics {
  /** Number of messages sent through the channel */
  messagesSent: number;
  
  /** Total bytes sent through the channel */
  bytesSent: number;
  
  /** Number of messages received from the channel */
  messagesReceived: number;
  
  /** Total bytes received from the channel */
  bytesReceived: number;
  
  /** Number of times send() blocked waiting for buffer space */
  sendBlocks: number;
  
  /** Number of times receive() blocked waiting for data */
  receiveBlocks: number;
  
  /** Timestamp when metrics were last reset */
  resetTimestamp?: Date;
  
  /** Calculated throughput in messages per second */
  readonly messageRate?: number;
  
  /** Calculated throughput in bytes per second */
  readonly byteRate?: number;
  
  /** Average message size in bytes */
  readonly averageMessageSize?: number;
}

/**
 * @interface PerformanceMetrics
 * @brief Extended performance metrics with timing information
 */
export interface PerformanceMetrics extends ChannelMetrics {
  /** Total time spent in send operations (microseconds) */
  totalSendTime: number;
  
  /** Total time spent in receive operations (microseconds) */
  totalReceiveTime: number;
  
  /** Average send latency (microseconds) */
  readonly averageSendLatency: number;
  
  /** Average receive latency (microseconds) */
  readonly averageReceiveLatency: number;
  
  /** Peak send latency (microseconds) */
  peakSendLatency: number;
  
  /** Peak receive latency (microseconds) */
  peakReceiveLatency: number;
  
  /** 99th percentile send latency (microseconds) */
  p99SendLatency?: number;
  
  /** 99th percentile receive latency (microseconds) */
  p99ReceiveLatency?: number;
}

/**
 * @interface NetworkMetrics
 * @brief Network-specific metrics for TCP/UDP channels
 */
export interface NetworkMetrics {
  /** Number of connection attempts */
  connectionAttempts: number;
  
  /** Number of successful connections */
  connectionsEstablished: number;
  
  /** Number of connection failures */
  connectionFailures: number;
  
  /** Number of times connection was lost */
  disconnections: number;
  
  /** Number of reconnection attempts */
  reconnectionAttempts: number;
  
  /** Total bytes sent over network */
  networkBytesSent: number;
  
  /** Total bytes received from network */
  networkBytesReceived: number;
  
  /** Number of network timeouts */
  networkTimeouts: number;
  
  /** Number of network errors */
  networkErrors: number;
  
  /** Current connection state */
  connectionState: 'disconnected' | 'connecting' | 'connected' | 'reconnecting';
  
  /** Remote endpoint address */
  remoteEndpoint?: string;
  
  /** Local endpoint address */
  localEndpoint?: string;
  
  /** Round-trip time in milliseconds */
  rtt?: number;
}

/**
 * @interface CompressionMetrics
 * @brief Metrics for compression operations
 */
export interface CompressionMetrics {
  /** Number of messages compressed */
  messagesCompressed: number;
  
  /** Number of messages decompressed */
  messagesDecompressed: number;
  
  /** Total bytes before compression */
  bytesBeforeCompression: number;
  
  /** Total bytes after compression */
  bytesAfterCompression: number;
  
  /** Total time spent compressing (microseconds) */
  compressionTime: number;
  
  /** Total time spent decompressing (microseconds) */
  decompressionTime: number;
  
  /** Number of compression errors */
  compressionErrors: number;
  
  /** Number of decompression errors */
  decompressionErrors: number;
  
  /** Overall compression ratio */
  readonly compressionRatio: number;
  
  /** Average compression speed (MB/s) */
  readonly compressionSpeed: number;
  
  /** Average decompression speed (MB/s) */
  readonly decompressionSpeed: number;
}

/**
 * @interface BufferMetrics
 * @brief Metrics for buffer usage and allocation
 */
export interface BufferMetrics {
  /** Total buffer size in bytes */
  bufferSize: number;
  
  /** Currently used buffer space in bytes */
  usedSpace: number;
  
  /** Available buffer space in bytes */
  readonly freeSpace: number;
  
  /** Buffer utilization percentage (0-100) */
  readonly utilization: number;
  
  /** Peak buffer usage in bytes */
  peakUsage: number;
  
  /** Number of buffer allocation failures */
  allocationFailures: number;
  
  /** Number of buffer reallocations */
  reallocations: number;
  
  /** Total number of allocations */
  totalAllocations: number;
  
  /** Total number of deallocations */
  totalDeallocations: number;
}

/**
 * @interface ReliabilityMetrics
 * @brief Metrics for reliability features
 */
export interface ReliabilityMetrics {
  /** Number of acknowledgments sent */
  acksSent: number;
  
  /** Number of acknowledgments received */
  acksReceived: number;
  
  /** Number of acknowledgment timeouts */
  ackTimeouts: number;
  
  /** Number of message retries */
  retries: number;
  
  /** Number of messages that failed all retries */
  permanentFailures: number;
  
  /** Number of duplicate messages detected */
  duplicatesDetected: number;
  
  /** Number of out-of-order messages */
  outOfOrderMessages: number;
  
  /** Number of heartbeats sent */
  heartbeatsSent: number;
  
  /** Number of heartbeats received */
  heartbeatsReceived: number;
  
  /** Number of heartbeat timeouts */
  heartbeatTimeouts: number;
  
  /** Size of replay buffer */
  replayBufferSize: number;
  
  /** Number of messages in replay buffer */
  replayBufferMessages: number;
}

/**
 * @interface AggregatedMetrics
 * @brief Combined metrics from all subsystems
 */
export interface AggregatedMetrics {
  /** Basic channel metrics */
  channel: ChannelMetrics;
  
  /** Performance metrics with timing */
  performance?: PerformanceMetrics;
  
  /** Network-specific metrics */
  network?: NetworkMetrics;
  
  /** Compression metrics */
  compression?: CompressionMetrics;
  
  /** Buffer usage metrics */
  buffer?: BufferMetrics;
  
  /** Reliability metrics */
  reliability?: ReliabilityMetrics;
  
  /** Timestamp when metrics were collected */
  timestamp: Date;
  
  /** Duration over which metrics were collected */
  collectionDuration?: number;
}

/**
 * @interface MetricsSnapshot
 * @brief Point-in-time snapshot of metrics
 */
export interface MetricsSnapshot {
  /** Snapshot timestamp */
  timestamp: Date;
  
  /** All metrics at this point in time */
  metrics: AggregatedMetrics;
  
  /** Optional snapshot label */
  label?: string;
}

/**
 * @interface MetricsDelta
 * @brief Difference between two metric snapshots
 */
export interface MetricsDelta {
  /** Time period this delta covers */
  period: {
    start: Date;
    end: Date;
    duration: number; // milliseconds
  };
  
  /** Change in channel metrics */
  channel: Partial<ChannelMetrics>;
  
  /** Change in other metrics */
  performance?: Partial<PerformanceMetrics>;
  network?: Partial<NetworkMetrics>;
  compression?: Partial<CompressionMetrics>;
  buffer?: Partial<BufferMetrics>;
  reliability?: Partial<ReliabilityMetrics>;
}

/**
 * @interface MetricsCollector
 * @brief Interface for collecting and managing metrics
 */
export interface MetricsCollector {
  /** Start collecting metrics */
  start(): void;
  
  /** Stop collecting metrics */
  stop(): void;
  
  /** Reset all metrics to zero */
  reset(): void;
  
  /** Get current metrics snapshot */
  getSnapshot(): MetricsSnapshot;
  
  /** Get metrics for a specific time period */
  getMetrics(start?: Date, end?: Date): AggregatedMetrics;
  
  /** Calculate delta between two snapshots */
  calculateDelta(before: MetricsSnapshot, after: MetricsSnapshot): MetricsDelta;
  
  /** Check if metrics collection is enabled */
  readonly enabled: boolean;
  
  /** Get collection interval in milliseconds */
  readonly interval: number;
}

/**
 * @type MetricsEventMap
 * @brief Event map for metrics events
 */
export interface MetricsEventMap {
  /** Emitted when metrics are updated */
  updated: (metrics: AggregatedMetrics) => void;
  
  /** Emitted when metrics are reset */
  reset: () => void;
  
  /** Emitted when collection starts */
  started: () => void;
  
  /** Emitted when collection stops */
  stopped: () => void;
  
  /** Emitted when a threshold is exceeded */
  threshold: (metric: string, value: number, threshold: number) => void;
}

/**
 * @interface MetricsThreshold
 * @brief Configuration for metric thresholds
 */
export interface MetricsThreshold {
  /** Metric name to monitor */
  metric: string;
  
  /** Threshold value */
  value: number;
  
  /** Comparison type */
  condition: 'greater_than' | 'less_than' | 'equal' | 'not_equal';
  
  /** Optional callback when threshold is crossed */
  callback?: (value: number) => void;
  
  /** Whether to trigger only once or repeatedly */
  triggerOnce?: boolean;
}

/**
 * @interface MetricsExporter
 * @brief Interface for exporting metrics to external systems
 */
export interface MetricsExporter {
  /** Export metrics in a specific format */
  export(metrics: AggregatedMetrics, format: 'json' | 'csv' | 'prometheus'): string;
  
  /** Send metrics to external endpoint */
  send(metrics: AggregatedMetrics, endpoint: string): Promise<void>;
  
  /** Configure export settings */
  configure(options: Record<string, any>): void;
}