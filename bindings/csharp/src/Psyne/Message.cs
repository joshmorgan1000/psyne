using System;
using System.Runtime.InteropServices;
using System.Text;
using Psyne.Native;

namespace Psyne
{
    /// <summary>
    /// Represents a message that can be sent or received through a Psyne channel.
    /// </summary>
    public sealed class Message : IDisposable
    {
        private readonly IntPtr _handle;
        private readonly bool _isReceived;
        private bool _disposed;

        internal Message(IntPtr handle, bool isReceived)
        {
            _handle = handle;
            _isReceived = isReceived;
        }

        /// <summary>
        /// Gets the size of the message data in bytes.
        /// </summary>
        public int Size { get; private set; }

        /// <summary>
        /// Gets the message type identifier.
        /// </summary>
        public uint Type { get; internal set; }

        /// <summary>
        /// Gets a pointer to the message data.
        /// </summary>
        internal IntPtr DataPointer { get; private set; }

        /// <summary>
        /// Gets the message data as a byte array.
        /// </summary>
        /// <returns>A copy of the message data.</returns>
        public byte[] GetData()
        {
            ThrowIfDisposed();
            EnsureDataPointer();

            var data = new byte[Size];
            if (Size > 0 && DataPointer != IntPtr.Zero)
            {
                Marshal.Copy(DataPointer, data, 0, Size);
            }
            return data;
        }

        /// <summary>
        /// Gets the message data as a string using UTF-8 encoding.
        /// </summary>
        /// <returns>The message data as a string.</returns>
        public string GetString()
        {
            ThrowIfDisposed();
            EnsureDataPointer();

            if (Size == 0 || DataPointer == IntPtr.Zero)
                return string.Empty;

            return Marshal.PtrToStringUTF8(DataPointer, Size) ?? string.Empty;
        }

        /// <summary>
        /// Copies data to the message buffer. Only valid for messages being sent.
        /// </summary>
        /// <param name="data">The data to copy.</param>
        public void SetData(byte[] data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            ThrowIfDisposed();
            if (_isReceived)
                throw new InvalidOperationException("Cannot set data on a received message");

            EnsureDataPointer();

            if (data.Length > Size)
                throw new ArgumentException($"Data size ({data.Length}) exceeds message buffer size ({Size})");

            if (DataPointer != IntPtr.Zero && data.Length > 0)
            {
                Marshal.Copy(data, 0, DataPointer, data.Length);
            }
        }

        /// <summary>
        /// Copies a string to the message buffer using UTF-8 encoding. Only valid for messages being sent.
        /// </summary>
        /// <param name="text">The string to copy.</param>
        public void SetString(string text)
        {
            if (text == null)
                throw new ArgumentNullException(nameof(text));

            var data = Encoding.UTF8.GetBytes(text);
            SetData(data);
        }

        /// <summary>
        /// Gets a span over the message data for efficient access.
        /// </summary>
        /// <returns>A span over the message data.</returns>
        public unsafe Span<byte> GetSpan()
        {
            ThrowIfDisposed();
            EnsureDataPointer();

            if (Size == 0 || DataPointer == IntPtr.Zero)
                return Span<byte>.Empty;

            return new Span<byte>(DataPointer.ToPointer(), Size);
        }

        /// <summary>
        /// Sends the message with the specified type.
        /// </summary>
        /// <param name="messageType">The message type identifier.</param>
        internal void Send(uint messageType)
        {
            ThrowIfDisposed();
            if (_isReceived)
                throw new InvalidOperationException("Cannot send a received message");

            var result = PsyneNative.psyne_message_send(_handle, messageType);
            PsyneException.ThrowIfError(result);
            
            Type = messageType;
            _disposed = true; // Message is consumed after sending
        }

        /// <summary>
        /// Cancels the message without sending it.
        /// </summary>
        internal void Cancel()
        {
            if (!_disposed && !_isReceived)
            {
                PsyneNative.psyne_message_cancel(_handle);
                _disposed = true;
            }
        }

        private void EnsureDataPointer()
        {
            if (DataPointer == IntPtr.Zero)
            {
                var result = PsyneNative.psyne_message_get_data(_handle, out var dataPtr, out var size);
                PsyneException.ThrowIfError(result);
                
                DataPointer = dataPtr;
                Size = (int)size;
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Message));
        }

        /// <summary>
        /// Releases the resources associated with this message.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_isReceived)
                {
                    PsyneNative.psyne_message_release(_handle);
                }
                else
                {
                    // Cancel unsent message
                    Cancel();
                }
                _disposed = true;
            }
        }
    }
}