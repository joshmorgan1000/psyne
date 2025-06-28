/**
 * @file errors.ts
 * @brief Custom error classes for Psyne JavaScript bindings
 * 
 * Provides specialized error types for different failure modes
 * in Psyne operations, following JavaScript error conventions.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

/**
 * @class PsyneError
 * @brief Base error class for all Psyne-related errors
 * 
 * Extends the standard Error class with additional context
 * and error categorization for Psyne operations.
 */
export class PsyneError extends Error {
  /** Error category/type */
  public readonly type: string;
  
  /** Original error that caused this error (if any) */
  public readonly cause?: Error;
  
  /** Additional error context */
  public readonly context?: Record<string, any>;

  /**
   * @brief Create a new PsyneError
   * @param message Error message
   * @param cause Original error that caused this error
   * @param context Additional error context
   */
  constructor(message: string, cause?: Error, context?: Record<string, any>) {
    super(message);
    
    this.name = 'PsyneError';
    this.type = 'psyne';
    this.cause = cause;
    this.context = context;
    
    // Ensure proper prototype chain for instanceof checks
    Object.setPrototypeOf(this, PsyneError.prototype);
    
    // Capture stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, PsyneError);
    }
  }

  /**
   * @brief Get a detailed error description
   * @returns Detailed error information including cause and context
   */
  getDetails(): string {
    let details = `${this.name}: ${this.message}`;
    
    if (this.cause) {
      details += `\nCaused by: ${this.cause.name}: ${this.cause.message}`;
    }
    
    if (this.context) {
      details += `\nContext: ${JSON.stringify(this.context, null, 2)}`;
    }
    
    return details;
  }

  /**
   * @brief Convert error to JSON representation
   * @returns JSON object with error details
   */
  toJSON(): Record<string, any> {
    return {
      name: this.name,
      type: this.type,
      message: this.message,
      stack: this.stack,
      cause: this.cause ? {
        name: this.cause.name,
        message: this.cause.message,
        stack: this.cause.stack
      } : undefined,
      context: this.context
    };
  }
}

/**
 * @class ChannelError
 * @brief Error related to channel operations
 */
export class ChannelError extends PsyneError {
  /** Channel URI where error occurred */
  public readonly channelUri?: string;

  constructor(message: string, channelUri?: string, cause?: Error) {
    super(message, cause, { channelUri });
    this.name = 'ChannelError';
    this.type = 'channel';
    this.channelUri = channelUri;
    Object.setPrototypeOf(this, ChannelError.prototype);
  }
}

/**
 * @class ChannelClosedError
 * @brief Error when attempting operations on a closed channel
 */
export class ChannelClosedError extends ChannelError {
  constructor(message: string = 'Channel is closed', channelUri?: string) {
    super(message, channelUri);
    this.name = 'ChannelClosedError';
    this.type = 'channel_closed';
    Object.setPrototypeOf(this, ChannelClosedError.prototype);
  }
}

/**
 * @class ConnectionError
 * @brief Error related to network connections
 */
export class ConnectionError extends ChannelError {
  /** Remote endpoint address */
  public readonly endpoint?: string;
  
  /** Connection timeout if applicable */
  public readonly timeout?: number;

  constructor(message: string, endpoint?: string, timeout?: number, cause?: Error) {
    super(message, endpoint, cause);
    this.name = 'ConnectionError';
    this.type = 'connection';
    this.endpoint = endpoint;
    this.timeout = timeout;
    Object.setPrototypeOf(this, ConnectionError.prototype);
  }
}

/**
 * @class MessageError
 * @brief Error related to message operations
 */
export class MessageError extends PsyneError {
  /** Message type that caused the error */
  public readonly messageType?: string;
  
  /** Message size if applicable */
  public readonly messageSize?: number;

  constructor(
    message: string, 
    cause?: Error, 
    messageType?: string, 
    messageSize?: number
  ) {
    super(message, cause, { messageType, messageSize });
    this.name = 'MessageError';
    this.type = 'message';
    this.messageType = messageType;
    this.messageSize = messageSize;
    Object.setPrototypeOf(this, MessageError.prototype);
  }
}

/**
 * @class SerializationError
 * @brief Error during message serialization/deserialization
 */
export class SerializationError extends MessageError {
  /** Serialization format that failed */
  public readonly format?: string;

  constructor(
    message: string, 
    format?: string, 
    cause?: Error, 
    messageType?: string
  ) {
    super(message, cause, messageType);
    this.name = 'SerializationError';
    this.type = 'serialization';
    this.format = format;
    Object.setPrototypeOf(this, SerializationError.prototype);
  }
}

/**
 * @class CompressionError
 * @brief Error during compression/decompression operations
 */
export class CompressionError extends MessageError {
  /** Compression algorithm that failed */
  public readonly algorithm?: string;
  
  /** Compression level if applicable */
  public readonly level?: number;

  constructor(
    message: string, 
    algorithm?: string, 
    level?: number, 
    cause?: Error
  ) {
    super(message, cause);
    this.name = 'CompressionError';
    this.type = 'compression';
    this.algorithm = algorithm;
    this.level = level;
    Object.setPrototypeOf(this, CompressionError.prototype);
  }
}

/**
 * @class TimeoutError
 * @brief Error when operations exceed timeout limits
 */
export class TimeoutError extends PsyneError {
  /** Timeout duration in milliseconds */
  public readonly timeout: number;
  
  /** Operation that timed out */
  public readonly operation?: string;

  constructor(message: string, timeout: number, operation?: string) {
    super(message, undefined, { timeout, operation });
    this.name = 'TimeoutError';
    this.type = 'timeout';
    this.timeout = timeout;
    this.operation = operation;
    Object.setPrototypeOf(this, TimeoutError.prototype);
  }
}

/**
 * @class ConfigurationError
 * @brief Error in channel or library configuration
 */
export class ConfigurationError extends PsyneError {
  /** Configuration parameter that caused the error */
  public readonly parameter?: string;
  
  /** Invalid value that was provided */
  public readonly value?: any;

  constructor(message: string, parameter?: string, value?: any) {
    super(message, undefined, { parameter, value });
    this.name = 'ConfigurationError';
    this.type = 'configuration';
    this.parameter = parameter;
    this.value = value;
    Object.setPrototypeOf(this, ConfigurationError.prototype);
  }
}

/**
 * @class BufferOverflowError
 * @brief Error when buffer capacity is exceeded
 */
export class BufferOverflowError extends MessageError {
  /** Current buffer size */
  public readonly bufferSize: number;
  
  /** Requested size that caused overflow */
  public readonly requestedSize: number;

  constructor(bufferSize: number, requestedSize: number) {
    const message = `Buffer overflow: requested ${requestedSize} bytes, but buffer size is ${bufferSize} bytes`;
    super(message, undefined, undefined, requestedSize);
    this.name = 'BufferOverflowError';
    this.type = 'buffer_overflow';
    this.bufferSize = bufferSize;
    this.requestedSize = requestedSize;
    Object.setPrototypeOf(this, BufferOverflowError.prototype);
  }
}

/**
 * @class UnsupportedOperationError
 * @brief Error when attempting unsupported operations
 */
export class UnsupportedOperationError extends PsyneError {
  /** Operation that is not supported */
  public readonly operation: string;
  
  /** Reason why operation is not supported */
  public readonly reason?: string;

  constructor(operation: string, reason?: string) {
    const message = `Unsupported operation: ${operation}${reason ? ` (${reason})` : ''}`;
    super(message, undefined, { operation, reason });
    this.name = 'UnsupportedOperationError';
    this.type = 'unsupported_operation';
    this.operation = operation;
    this.reason = reason;
    Object.setPrototypeOf(this, UnsupportedOperationError.prototype);
  }
}

/**
 * @brief Check if an error is a Psyne-related error
 * @param error Error to check
 * @returns True if error is a PsyneError or subclass
 */
export function isPsyneError(error: any): error is PsyneError {
  return error instanceof PsyneError;
}

/**
 * @brief Check if an error is channel-related
 * @param error Error to check
 * @returns True if error is a ChannelError or subclass
 */
export function isChannelError(error: any): error is ChannelError {
  return error instanceof ChannelError;
}

/**
 * @brief Check if an error is message-related
 * @param error Error to check
 * @returns True if error is a MessageError or subclass
 */
export function isMessageError(error: any): error is MessageError {
  return error instanceof MessageError;
}

/**
 * @brief Create a PsyneError from a generic error
 * @param error Original error
 * @param message Optional custom message
 * @returns PsyneError wrapping the original error
 */
export function wrapError(error: any, message?: string): PsyneError {
  if (isPsyneError(error)) {
    return error;
  }
  
  const errorMessage = message || (error?.message) || 'Unknown error';
  return new PsyneError(errorMessage, error instanceof Error ? error : undefined);
}

/**
 * @brief Handle errors in async operations with proper error wrapping
 * @param operation Async operation to execute
 * @param context Additional context for error reporting
 * @returns Promise that resolves with operation result or rejects with wrapped error
 */
export async function handleAsync<T>(
  operation: () => Promise<T>,
  context?: Record<string, any>
): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    const wrappedError = wrapError(error);
    if (context) {
      wrappedError.context = { ...wrappedError.context, ...context };
    }
    throw wrappedError;
  }
}