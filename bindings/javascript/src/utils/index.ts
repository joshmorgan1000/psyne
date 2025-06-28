/**
 * @file utils/index.ts
 * @brief Utility functions for Psyne JavaScript bindings
 * 
 * Provides helper functions for working with channels, messages,
 * performance monitoring, and data conversion.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

import { ChannelURI, ChannelMetrics, BenchmarkResult } from '../types';
import { Channel } from '../core/channel';
import { PsyneError } from '../errors';

/**
 * @brief Parse a channel URI into its components
 * @param uri Channel URI to parse
 * @returns Parsed URI components
 * 
 * @example
 * ```typescript
 * const parsed = parseChannelURI('tcp://localhost:8080');
 * // { scheme: 'tcp', host: 'localhost', port: 8080 }
 * ```
 */
export function parseChannelURI(uri: string): ParsedChannelURI {
  const match = uri.match(/^([a-z]+):\/\/(.*)$/);
  if (!match) {
    throw new PsyneError(`Invalid channel URI format: ${uri}`);
  }

  const [, scheme, rest] = match;
  
  switch (scheme) {
    case 'memory':
    case 'ipc':
      return { scheme, name: rest };
      
    case 'tcp':
    case 'ws':
    case 'wss':
      return parseTcpURI(scheme, rest);
      
    case 'unix':
      return { scheme, path: rest };
      
    case 'multicast':
      return parseMulticastURI(scheme, rest);
      
    case 'rdma':
      return parseTcpURI(scheme, rest); // Same format as TCP
      
    default:
      throw new PsyneError(`Unsupported URI scheme: ${scheme}`);
  }
}

/**
 * @interface ParsedChannelURI
 * @brief Components of a parsed channel URI
 */
export interface ParsedChannelURI {
  scheme: string;
  name?: string;
  host?: string;
  port?: number;
  path?: string;
}

function parseTcpURI(scheme: string, rest: string): ParsedChannelURI {
  if (rest.startsWith(':')) {
    // Server mode: :port
    const port = parseInt(rest.substring(1), 10);
    if (isNaN(port)) {
      throw new PsyneError(`Invalid port in URI: ${rest}`);
    }
    return { scheme, port };
  } else {
    // Client mode: host:port
    const [host, portStr] = rest.split(':');
    const port = parseInt(portStr, 10);
    if (isNaN(port)) {
      throw new PsyneError(`Invalid port in URI: ${rest}`);
    }
    return { scheme, host, port };
  }
}

function parseMulticastURI(scheme: string, rest: string): ParsedChannelURI {
  const [host, portStr] = rest.split(':');
  const port = parseInt(portStr, 10);
  if (isNaN(port)) {
    throw new PsyneError(`Invalid port in multicast URI: ${rest}`);
  }
  return { scheme, host, port };
}

/**
 * @brief Validate a channel URI format
 * @param uri URI to validate
 * @returns True if URI is valid
 */
export function isValidChannelURI(uri: string): boolean {
  try {
    parseChannelURI(uri);
    return true;
  } catch {
    return false;
  }
}

/**
 * @brief Build a channel URI from components
 * @param components URI components
 * @returns Formatted URI string
 */
export function buildChannelURI(components: ParsedChannelURI): string {
  const { scheme } = components;
  
  switch (scheme) {
    case 'memory':
    case 'ipc':
      return `${scheme}://${components.name}`;
      
    case 'tcp':
    case 'ws':
    case 'wss':
    case 'rdma':
      if (components.host) {
        return `${scheme}://${components.host}:${components.port}`;
      } else {
        return `${scheme}://:${components.port}`;
      }
      
    case 'unix':
      return `${scheme}://${components.path}`;
      
    case 'multicast':
      return `${scheme}://${components.host}:${components.port}`;
      
    default:
      throw new PsyneError(`Unsupported URI scheme: ${scheme}`);
  }
}

/**
 * @brief Calculate metrics summary from channel metrics
 * @param metrics Channel metrics object
 * @param timeWindow Time window in milliseconds for rate calculations
 * @returns Summary with calculated rates and ratios
 */
export function calculateMetricsSummary(
  metrics: ChannelMetrics, 
  timeWindow: number = 1000
): MetricsSummary {
  const timeWindowSeconds = timeWindow / 1000;
  
  return {
    ...metrics,
    messageRate: metrics.messagesSent / timeWindowSeconds,
    byteRate: metrics.bytesSent / timeWindowSeconds,
    averageMessageSize: metrics.messagesSent > 0 ? 
      metrics.bytesSent / metrics.messagesSent : 0,
    efficiency: calculateEfficiency(metrics),
    throughputMbps: (metrics.bytesSent / (1024 * 1024)) / timeWindowSeconds
  };
}

/**
 * @interface MetricsSummary
 * @brief Extended metrics with calculated values
 */
export interface MetricsSummary extends ChannelMetrics {
  messageRate: number;
  byteRate: number;
  averageMessageSize: number;
  efficiency: number;
  throughputMbps: number;
}

function calculateEfficiency(metrics: ChannelMetrics): number {
  const totalOperations = metrics.messagesSent + metrics.messagesReceived;
  const totalBlocks = metrics.sendBlocks + metrics.receiveBlocks;
  
  if (totalOperations === 0) return 1.0;
  return Math.max(0, 1.0 - (totalBlocks / totalOperations));
}

/**
 * @brief Convert various data types to a format suitable for channel transmission
 * @param data Input data of various types
 * @returns Converted data ready for transmission
 */
export function prepareDataForTransmission(data: any): any {
  if (data === null || data === undefined) {
    throw new PsyneError('Cannot send null or undefined data');
  }

  // Handle different data types
  if (typeof data === 'number') {
    return new Float32Array([data]);
  }
  
  if (typeof data === 'string') {
    return Buffer.from(data, 'utf8');
  }
  
  if (Array.isArray(data)) {
    // Determine if array contains numbers
    if (data.length > 0 && typeof data[0] === 'number') {
      return new Float32Array(data);
    }
    // For other arrays, serialize as JSON
    return Buffer.from(JSON.stringify(data), 'utf8');
  }
  
  if (ArrayBuffer.isView(data)) {
    return data; // TypedArray or DataView
  }
  
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  
  if (Buffer.isBuffer(data)) {
    return data;
  }
  
  if (typeof data === 'object') {
    // Check if it's already a message type
    if (data.type && data.typeId && data.data) {
      return data; // Already a message
    }
    
    // Serialize object as JSON
    return Buffer.from(JSON.stringify(data), 'utf8');
  }
  
  throw new PsyneError(`Unsupported data type: ${typeof data}`);
}

/**
 * @brief Create a simple performance monitor for a channel
 * @param channel Channel to monitor
 * @param interval Monitoring interval in milliseconds
 * @returns Performance monitor instance
 */
export function createPerformanceMonitor(
  channel: Channel, 
  interval: number = 1000
): PerformanceMonitor {
  return new PerformanceMonitor(channel, interval);
}

/**
 * @class PerformanceMonitor
 * @brief Simple performance monitoring for channels
 */
export class PerformanceMonitor {
  private interval: NodeJS.Timeout | null = null;
  private lastMetrics: ChannelMetrics | null = null;
  private callbacks: Array<(summary: MetricsSummary) => void> = [];

  constructor(
    private channel: Channel,
    private intervalMs: number = 1000
  ) {}

  /**
   * @brief Start monitoring
   */
  start(): void {
    if (this.interval) return; // Already started

    this.interval = setInterval(() => {
      this.collectMetrics();
    }, this.intervalMs);
  }

  /**
   * @brief Stop monitoring
   */
  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  /**
   * @brief Add a callback for metric updates
   * @param callback Function to call with metric updates
   */
  onUpdate(callback: (summary: MetricsSummary) => void): void {
    this.callbacks.push(callback);
  }

  private collectMetrics(): void {
    if (!this.channel.hasMetrics) return;

    const currentMetrics = this.channel.getMetrics();
    if (!currentMetrics) return;

    const summary = calculateMetricsSummary(currentMetrics, this.intervalMs);
    
    this.callbacks.forEach(callback => {
      try {
        callback(summary);
      } catch (error) {
        console.error('Error in performance monitor callback:', error);
      }
    });

    this.lastMetrics = currentMetrics;
  }
}

/**
 * @brief Wait for a condition to be met with timeout
 * @param condition Function that returns true when condition is met
 * @param timeout Timeout in milliseconds
 * @param interval Check interval in milliseconds
 * @returns Promise that resolves when condition is met
 */
export async function waitFor(
  condition: () => boolean,
  timeout: number = 5000,
  interval: number = 100
): Promise<void> {
  const startTime = Date.now();
  
  while (!condition()) {
    if (Date.now() - startTime > timeout) {
      throw new PsyneError(`Timeout waiting for condition after ${timeout}ms`);
    }
    
    await new Promise(resolve => setTimeout(resolve, interval));
  }
}

/**
 * @brief Create a promise that resolves after specified delay
 * @param ms Delay in milliseconds
 * @returns Promise that resolves after delay
 */
export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * @brief Retry an async operation with exponential backoff
 * @param operation Operation to retry
 * @param maxRetries Maximum number of retries
 * @param baseDelay Base delay in milliseconds
 * @param backoffFactor Multiplication factor for delay
 * @returns Promise with operation result
 */
export async function retryWithBackoff<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 100,
  backoffFactor: number = 2
): Promise<T> {
  let lastError: Error;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      
      if (attempt === maxRetries) {
        throw new PsyneError(
          `Operation failed after ${maxRetries} retries`, 
          lastError
        );
      }
      
      const delayMs = baseDelay * Math.pow(backoffFactor, attempt);
      await delay(delayMs);
    }
  }
  
  throw lastError!;
}

/**
 * @brief Measure execution time of an async operation
 * @param operation Operation to measure
 * @returns Object with result and execution time
 */
export async function measureTime<T>(
  operation: () => Promise<T>
): Promise<{ result: T; timeMs: number }> {
  const startTime = process.hrtime.bigint();
  const result = await operation();
  const endTime = process.hrtime.bigint();
  
  const timeMs = Number(endTime - startTime) / 1000000; // Convert nanoseconds to milliseconds
  
  return { result, timeMs };
}

/**
 * @brief Create a debounced version of a function
 * @param func Function to debounce
 * @param delay Delay in milliseconds
 * @returns Debounced function
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
}

/**
 * @brief Create a throttled version of a function
 * @param func Function to throttle
 * @param interval Minimum interval between calls in milliseconds
 * @returns Throttled function
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  interval: number
): (...args: Parameters<T>) => void {
  let lastCallTime = 0;
  
  return (...args: Parameters<T>) => {
    const now = Date.now();
    if (now - lastCallTime >= interval) {
      lastCallTime = now;
      func(...args);
    }
  };
}

/**
 * @brief Format bytes into human-readable string
 * @param bytes Number of bytes
 * @param decimals Number of decimal places
 * @returns Formatted string (e.g., "1.5 MB")
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
}

/**
 * @brief Format duration into human-readable string
 * @param ms Duration in milliseconds
 * @returns Formatted string (e.g., "1.5s", "250ms")
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`;
  } else {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  }
}