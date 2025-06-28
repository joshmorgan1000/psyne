/**
 * @file messages/index.ts
 * @brief Message type implementations for Psyne JavaScript bindings
 * 
 * Provides typed message classes that correspond to the C++ message types,
 * with convenient JavaScript APIs for creating and manipulating message data.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

import { 
  FloatVectorMessage, 
  DoubleMatrixMessage, 
  ByteVectorMessage,
  Vector3fMessage,
  Matrix4x4fMessage,
  ComplexVectorMessage,
  MLTensorMessage,
  SparseMatrixMessage,
  MessageTypes
} from '../types';

/**
 * @class FloatVector
 * @brief JavaScript wrapper for FloatVector messages
 * 
 * Represents a dynamic array of single-precision floating-point values.
 * Optimized for high-performance mathematical computations.
 * 
 * @example
 * ```typescript
 * const vector = new FloatVector([1.0, 2.0, 3.0]);
 * await channel.send(vector);
 * 
 * // Or create from typed array
 * const data = new Float32Array([1.0, 2.0, 3.0]);
 * const vector2 = FloatVector.fromTypedArray(data);
 * ```
 */
export class FloatVector implements FloatVectorMessage {
  public readonly type = 'floatVector' as const;
  public readonly typeId = MessageTypes.FLOAT_VECTOR;
  public readonly size: number;
  public readonly data: Float32Array;

  /**
   * @brief Create a new FloatVector
   * @param data Array or typed array of float values
   */
  constructor(data: number[] | Float32Array) {
    if (Array.isArray(data)) {
      this.data = new Float32Array(data);
    } else {
      this.data = data;
    }
    this.size = this.data.byteLength;
  }

  /**
   * @brief Create FloatVector from typed array
   * @param array Float32Array to wrap
   * @returns New FloatVector instance
   */
  static fromTypedArray(array: Float32Array): FloatVector {
    return new FloatVector(array);
  }

  /**
   * @brief Create FloatVector from buffer
   * @param buffer ArrayBuffer containing float data
   * @param offset Byte offset into buffer
   * @param length Number of float elements
   * @returns New FloatVector instance
   */
  static fromBuffer(buffer: ArrayBuffer, offset: number = 0, length?: number): FloatVector {
    const array = new Float32Array(buffer, offset, length);
    return new FloatVector(array);
  }

  /**
   * @brief Get element at index
   * @param index Element index
   * @returns Float value at index
   */
  get(index: number): number {
    return this.data[index];
  }

  /**
   * @brief Set element at index
   * @param index Element index
   * @param value New value
   */
  set(index: number, value: number): void {
    this.data[index] = value;
  }

  /**
   * @brief Get the length of the vector
   * @returns Number of elements
   */
  get length(): number {
    return this.data.length;
  }

  /**
   * @brief Convert to regular JavaScript array
   * @returns Array of numbers
   */
  toArray(): number[] {
    return Array.from(this.data);
  }

  /**
   * @brief Create a copy of this vector
   * @returns New FloatVector with copied data
   */
  clone(): FloatVector {
    return new FloatVector(new Float32Array(this.data));
  }
}

/**
 * @class DoubleMatrix
 * @brief JavaScript wrapper for DoubleMatrix messages
 * 
 * Represents a 2D matrix of double-precision floating-point values
 * with row-major storage layout.
 * 
 * @example
 * ```typescript
 * const matrix = new DoubleMatrix(2, 3, [1, 2, 3, 4, 5, 6]);
 * console.log(matrix.get(1, 2)); // 6
 * 
 * matrix.set(0, 0, 10);
 * await channel.send(matrix);
 * ```
 */
export class DoubleMatrix implements DoubleMatrixMessage {
  public readonly type = 'doubleMatrix' as const;
  public readonly typeId = MessageTypes.DOUBLE_MATRIX;
  public readonly size: number;
  public readonly data: {
    rows: number;
    cols: number;
    values: Float64Array;
  };

  /**
   * @brief Create a new DoubleMatrix
   * @param rows Number of rows
   * @param cols Number of columns
   * @param data Matrix data in row-major order
   */
  constructor(rows: number, cols: number, data?: number[] | Float64Array) {
    this.data = {
      rows,
      cols,
      values: data ? (Array.isArray(data) ? new Float64Array(data) : data) 
                   : new Float64Array(rows * cols)
    };
    this.size = 8 + this.data.values.byteLength; // header + data
  }

  /**
   * @brief Create matrix from 2D array
   * @param array 2D array of numbers
   * @returns New DoubleMatrix instance
   */
  static from2DArray(array: number[][]): DoubleMatrix {
    const rows = array.length;
    const cols = rows > 0 ? array[0].length : 0;
    const data = new Float64Array(rows * cols);
    
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        data[i * cols + j] = array[i][j];
      }
    }
    
    return new DoubleMatrix(rows, cols, data);
  }

  /**
   * @brief Get element at row, column
   * @param row Row index
   * @param col Column index
   * @returns Element value
   */
  get(row: number, col: number): number {
    return this.data.values[row * this.data.cols + col];
  }

  /**
   * @brief Set element at row, column
   * @param row Row index
   * @param col Column index
   * @param value New value
   */
  set(row: number, col: number, value: number): void {
    this.data.values[row * this.data.cols + col] = value;
  }

  /**
   * @brief Get number of rows
   * @returns Row count
   */
  get rows(): number {
    return this.data.rows;
  }

  /**
   * @brief Get number of columns
   * @returns Column count
   */
  get cols(): number {
    return this.data.cols;
  }

  /**
   * @brief Convert to 2D JavaScript array
   * @returns 2D array of numbers
   */
  to2DArray(): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < this.rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.cols; j++) {
        row.push(this.get(i, j));
      }
      result.push(row);
    }
    return result;
  }

  /**
   * @brief Create a copy of this matrix
   * @returns New DoubleMatrix with copied data
   */
  clone(): DoubleMatrix {
    return new DoubleMatrix(this.rows, this.cols, new Float64Array(this.data.values));
  }
}

/**
 * @class ByteVector
 * @brief JavaScript wrapper for ByteVector messages
 * 
 * Represents raw binary data as a vector of bytes.
 * Useful for transmitting arbitrary binary data or serialized objects.
 * 
 * @example
 * ```typescript
 * const bytes = new ByteVector(Buffer.from('hello world'));
 * await channel.send(bytes);
 * 
 * // Or from Uint8Array
 * const data = new Uint8Array([1, 2, 3, 4]);
 * const bytes2 = new ByteVector(data);
 * ```
 */
export class ByteVector implements ByteVectorMessage {
  public readonly type = 'byteVector' as const;
  public readonly typeId = MessageTypes.BYTE_VECTOR;
  public readonly size: number;
  public readonly data: Uint8Array;

  /**
   * @brief Create a new ByteVector
   * @param data Buffer or Uint8Array of byte data
   */
  constructor(data: Buffer | Uint8Array) {
    if (Buffer.isBuffer(data)) {
      this.data = new Uint8Array(data);
    } else {
      this.data = data;
    }
    this.size = this.data.byteLength;
  }

  /**
   * @brief Create ByteVector from string
   * @param str String to encode
   * @param encoding Text encoding (default: utf8)
   * @returns New ByteVector instance
   */
  static fromString(str: string, encoding: BufferEncoding = 'utf8'): ByteVector {
    return new ByteVector(Buffer.from(str, encoding));
  }

  /**
   * @brief Create ByteVector from ArrayBuffer
   * @param buffer ArrayBuffer to wrap
   * @param offset Byte offset into buffer
   * @param length Number of bytes
   * @returns New ByteVector instance
   */
  static fromArrayBuffer(buffer: ArrayBuffer, offset: number = 0, length?: number): ByteVector {
    return new ByteVector(new Uint8Array(buffer, offset, length));
  }

  /**
   * @brief Convert to Node.js Buffer
   * @returns Buffer containing the byte data
   */
  toBuffer(): Buffer {
    return Buffer.from(this.data);
  }

  /**
   * @brief Convert to string
   * @param encoding Text encoding (default: utf8)
   * @returns Decoded string
   */
  toString(encoding: BufferEncoding = 'utf8'): string {
    return this.toBuffer().toString(encoding);
  }

  /**
   * @brief Get the length in bytes
   * @returns Number of bytes
   */
  get length(): number {
    return this.data.length;
  }

  /**
   * @brief Create a copy of this byte vector
   * @returns New ByteVector with copied data
   */
  clone(): ByteVector {
    return new ByteVector(new Uint8Array(this.data));
  }
}

/**
 * @class Vector3f
 * @brief JavaScript wrapper for 3D vector messages
 * 
 * Represents a 3D vector with x, y, z components using single-precision floats.
 * Commonly used for 3D graphics, physics, and spatial computations.
 * 
 * @example
 * ```typescript
 * const position = new Vector3f(1.0, 2.0, 3.0);
 * const velocity = Vector3f.fromArray([0.5, -1.0, 0.0]);
 * 
 * position.normalize();
 * const distance = position.distance(velocity);
 * ```
 */
export class Vector3f implements Vector3fMessage {
  public readonly type = 'vector3f' as const;
  public readonly typeId = MessageTypes.VECTOR_3F;
  public readonly size = 12; // 3 * 4 bytes
  public readonly data: Float32Array;

  /**
   * @brief Create a new Vector3f
   * @param x X component (or array of [x, y, z])
   * @param y Y component
   * @param z Z component
   */
  constructor(x: number | [number, number, number], y?: number, z?: number) {
    if (Array.isArray(x)) {
      this.data = new Float32Array(x);
    } else {
      this.data = new Float32Array([x, y || 0, z || 0]);
    }
  }

  /**
   * @brief Create Vector3f from array
   * @param array Array of [x, y, z] values
   * @returns New Vector3f instance
   */
  static fromArray(array: [number, number, number]): Vector3f {
    return new Vector3f(array);
  }

  /**
   * @brief Get X component
   * @returns X value
   */
  get x(): number {
    return this.data[0];
  }

  /**
   * @brief Set X component
   * @param value New X value
   */
  set x(value: number) {
    this.data[0] = value;
  }

  /**
   * @brief Get Y component
   * @returns Y value
   */
  get y(): number {
    return this.data[1];
  }

  /**
   * @brief Set Y component
   * @param value New Y value
   */
  set y(value: number) {
    this.data[1] = value;
  }

  /**
   * @brief Get Z component
   * @returns Z value
   */
  get z(): number {
    return this.data[2];
  }

  /**
   * @brief Set Z component
   * @param value New Z value
   */
  set z(value: number) {
    this.data[2] = value;
  }

  /**
   * @brief Calculate vector length
   * @returns Vector magnitude
   */
  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }

  /**
   * @brief Normalize the vector in place
   * @returns This vector for chaining
   */
  normalize(): this {
    const len = this.length();
    if (len > 0) {
      this.x /= len;
      this.y /= len;
      this.z /= len;
    }
    return this;
  }

  /**
   * @brief Calculate dot product with another vector
   * @param other Other vector
   * @returns Dot product value
   */
  dot(other: Vector3f): number {
    return this.x * other.x + this.y * other.y + this.z * other.z;
  }

  /**
   * @brief Calculate distance to another vector
   * @param other Other vector
   * @returns Distance between vectors
   */
  distance(other: Vector3f): number {
    const dx = this.x - other.x;
    const dy = this.y - other.y;
    const dz = this.z - other.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  /**
   * @brief Convert to array
   * @returns [x, y, z] array
   */
  toArray(): [number, number, number] {
    return [this.x, this.y, this.z];
  }

  /**
   * @brief Create a copy of this vector
   * @returns New Vector3f with copied data
   */
  clone(): Vector3f {
    return new Vector3f(this.x, this.y, this.z);
  }
}

// Export all message types
export {
  FloatVectorMessage,
  DoubleMatrixMessage,
  ByteVectorMessage,
  Vector3fMessage,
  Matrix4x4fMessage,
  ComplexVectorMessage,
  MLTensorMessage,
  SparseMatrixMessage
} from '../types';

// Utility functions for message handling

/**
 * @brief Create a message from a generic object
 * @param obj Object with type and data properties
 * @returns Typed message instance
 */
export function createMessage(obj: any): any {
  switch (obj.type) {
    case 'floatVector':
      return new FloatVector(obj.data);
    case 'doubleMatrix':
      return new DoubleMatrix(obj.data.rows, obj.data.cols, obj.data.values);
    case 'byteVector':
      return new ByteVector(obj.data);
    case 'vector3f':
      if (Array.isArray(obj.data)) {
        return new Vector3f(obj.data);
      } else {
        return new Vector3f(obj.data.x, obj.data.y, obj.data.z);
      }
    default:
      throw new Error(`Unknown message type: ${obj.type}`);
  }
}

/**
 * @brief Check if an object is a valid message
 * @param obj Object to check
 * @returns True if object has message structure
 */
export function isMessage(obj: any): boolean {
  return obj && 
         typeof obj.type === 'string' && 
         typeof obj.typeId === 'number' && 
         typeof obj.size === 'number' &&
         obj.data !== undefined;
}