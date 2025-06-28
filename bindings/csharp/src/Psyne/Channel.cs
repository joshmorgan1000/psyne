using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Psyne.Native;

namespace Psyne
{
    /// <summary>
    /// Represents a high-performance messaging channel for zero-copy communication.
    /// </summary>
    public sealed class Channel : IDisposable
    {
        private readonly IntPtr _handle;
        private readonly object _lock = new();
        private bool _disposed;
        private PsyneNative.ReceiveCallback? _receiveCallback;
        private GCHandle _callbackHandle;

        internal Channel(IntPtr handle)
        {
            _handle = handle;
        }

        /// <summary>
        /// Gets the URI of this channel.
        /// </summary>
        public string Uri
        {
            get
            {
                ThrowIfDisposed();
                const int bufferSize = 512;
                var buffer = Marshal.AllocHGlobal(bufferSize);
                try
                {
                    var result = PsyneNative.psyne_channel_get_uri(_handle, buffer, (UIntPtr)bufferSize);
                    PsyneException.ThrowIfError(result);
                    return Marshal.PtrToStringUTF8(buffer) ?? string.Empty;
                }
                finally
                {
                    Marshal.FreeHGlobal(buffer);
                }
            }
        }

        /// <summary>
        /// Gets the buffer size of this channel in bytes.
        /// </summary>
        public long BufferSize
        {
            get
            {
                ThrowIfDisposed();
                var result = PsyneNative.psyne_channel_get_buffer_size(_handle, out var size);
                PsyneException.ThrowIfError(result);
                return (long)size;
            }
        }

        /// <summary>
        /// Gets a value indicating whether this channel has been stopped.
        /// </summary>
        public bool IsStopped
        {
            get
            {
                if (_disposed) return true;
                
                var result = PsyneNative.psyne_channel_is_stopped(_handle, out var stopped);
                return result == PsyneNative.ErrorCode.Ok && stopped;
            }
        }

        /// <summary>
        /// Reserves space for a message to be sent.
        /// </summary>
        /// <param name="size">The size of the message data in bytes.</param>
        /// <returns>A Message object ready for data to be written.</returns>
        public Message ReserveMessage(int size)
        {
            if (size < 0)
                throw new ArgumentOutOfRangeException(nameof(size), "Size cannot be negative");

            ThrowIfDisposed();

            var result = PsyneNative.psyne_message_reserve(_handle, (UIntPtr)size, out var messageHandle);
            PsyneException.ThrowIfError(result);

            return new Message(messageHandle, isReceived: false);
        }

        /// <summary>
        /// Sends a message with the specified data and type.
        /// </summary>
        /// <param name="data">The data to send.</param>
        /// <param name="messageType">The message type identifier.</param>
        public void Send(byte[] data, uint messageType = 0)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            using var message = ReserveMessage(data.Length);
            message.SetData(data);
            message.Send(messageType);
        }

        /// <summary>
        /// Sends a string message using UTF-8 encoding.
        /// </summary>
        /// <param name="text">The text to send.</param>
        /// <param name="messageType">The message type identifier.</param>
        public void Send(string text, uint messageType = 0)
        {
            if (text == null)
                throw new ArgumentNullException(nameof(text));

            using var message = ReserveMessage(System.Text.Encoding.UTF8.GetByteCount(text));
            message.SetString(text);
            message.Send(messageType);
        }

        /// <summary>
        /// Sends raw data using the low-level API.
        /// </summary>
        /// <param name="data">The data to send.</param>
        /// <param name="messageType">The message type identifier.</param>
        public unsafe void SendRaw(ReadOnlySpan<byte> data, uint messageType = 0)
        {
            ThrowIfDisposed();

            fixed (byte* ptr = data)
            {
                var result = PsyneNative.psyne_send_data(_handle, (IntPtr)ptr, (UIntPtr)data.Length, messageType);
                PsyneException.ThrowIfError(result);
            }
        }

        /// <summary>
        /// Attempts to receive a message without blocking.
        /// </summary>
        /// <param name="message">The received message, or null if none available.</param>
        /// <param name="messageType">The type of the received message.</param>
        /// <returns>True if a message was received, false otherwise.</returns>
        public bool TryReceive(out Message? message, out uint messageType)
        {
            ThrowIfDisposed();

            var result = PsyneNative.psyne_message_receive(_handle, out var messageHandle, out messageType);
            
            if (result == PsyneNative.ErrorCode.NoMessage)
            {
                message = null;
                messageType = 0;
                return false;
            }

            PsyneException.ThrowIfError(result);
            message = new Message(messageHandle, isReceived: true) { Type = messageType };
            return true;
        }

        /// <summary>
        /// Receives a message with a timeout.
        /// </summary>
        /// <param name="timeoutMs">Timeout in milliseconds. Use 0 for non-blocking, -1 for infinite.</param>
        /// <returns>The received message, or null if timeout occurred.</returns>
        public Message? Receive(int timeoutMs = -1)
        {
            ThrowIfDisposed();

            var timeout = timeoutMs < 0 ? uint.MaxValue : (uint)timeoutMs;
            var result = PsyneNative.psyne_message_receive_timeout(_handle, timeout, out var messageHandle, out var messageType);
            
            if (result == PsyneNative.ErrorCode.NoMessage || result == PsyneNative.ErrorCode.Timeout)
            {
                return null;
            }

            PsyneException.ThrowIfError(result);
            return new Message(messageHandle, isReceived: true) { Type = messageType };
        }

        /// <summary>
        /// Asynchronously receives a message.
        /// </summary>
        /// <param name="cancellationToken">Token to cancel the operation.</param>
        /// <returns>The received message, or null if cancelled.</returns>
        public async Task<Message?> ReceiveAsync(CancellationToken cancellationToken = default)
        {
            return await Task.Run(() =>
            {
                while (!cancellationToken.IsCancellationRequested && !IsStopped)
                {
                    var message = Receive(100); // 100ms timeout
                    if (message != null)
                        return message;
                }
                return null;
            }, cancellationToken);
        }

        /// <summary>
        /// Receives raw data using the low-level API.
        /// </summary>
        /// <param name="buffer">Buffer to receive data into.</param>
        /// <param name="timeoutMs">Timeout in milliseconds.</param>
        /// <returns>The number of bytes received and message type, or null if timeout/no message.</returns>
        public unsafe (int BytesReceived, uint MessageType)? ReceiveRaw(Span<byte> buffer, int timeoutMs = 0)
        {
            ThrowIfDisposed();

            fixed (byte* ptr = buffer)
            {
                var result = PsyneNative.psyne_receive_data(_handle, (IntPtr)ptr, (UIntPtr)buffer.Length, 
                    out var receivedSize, out var messageType, (uint)timeoutMs);
                
                if (result == PsyneNative.ErrorCode.NoMessage || result == PsyneNative.ErrorCode.Timeout)
                {
                    return null;
                }

                PsyneException.ThrowIfError(result);
                return ((int)receivedSize, messageType);
            }
        }

        /// <summary>
        /// Stops the channel, preventing further message operations.
        /// </summary>
        public void Stop()
        {
            ThrowIfDisposed();
            var result = PsyneNative.psyne_channel_stop(_handle);
            PsyneException.ThrowIfError(result);
        }

        /// <summary>
        /// Gets the current metrics for this channel.
        /// </summary>
        /// <returns>Channel metrics.</returns>
        public Metrics GetMetrics()
        {
            ThrowIfDisposed();
            var result = PsyneNative.psyne_channel_get_metrics(_handle, out var nativeMetrics);
            PsyneException.ThrowIfError(result);
            return Metrics.FromNative(nativeMetrics);
        }

        /// <summary>
        /// Enables or disables metrics collection for this channel.
        /// </summary>
        /// <param name="enable">True to enable metrics, false to disable.</param>
        public void EnableMetrics(bool enable = true)
        {
            ThrowIfDisposed();
            var result = PsyneNative.psyne_channel_enable_metrics(_handle, enable);
            PsyneException.ThrowIfError(result);
        }

        /// <summary>
        /// Resets the metrics counters for this channel.
        /// </summary>
        public void ResetMetrics()
        {
            ThrowIfDisposed();
            var result = PsyneNative.psyne_channel_reset_metrics(_handle);
            PsyneException.ThrowIfError(result);
        }

        /// <summary>
        /// Sets a callback to be invoked when messages are received.
        /// </summary>
        /// <param name="callback">The callback to invoke, or null to disable.</param>
        public void SetReceiveCallback(Action<Message, uint>? callback)
        {
            ThrowIfDisposed();

            lock (_lock)
            {
                // Clean up existing callback
                if (_callbackHandle.IsAllocated)
                {
                    _callbackHandle.Free();
                }

                if (callback == null)
                {
                    _receiveCallback = null;
                    var result = PsyneNative.psyne_channel_set_receive_callback(_handle, null, IntPtr.Zero);
                    PsyneException.ThrowIfError(result);
                }
                else
                {
                    _receiveCallback = (messagePtr, messageType, userData) =>
                    {
                        try
                        {
                            var message = new Message(messagePtr, isReceived: true) { Type = messageType };
                            callback(message, messageType);
                        }
                        catch
                        {
                            // Swallow exceptions in callback to prevent native code issues
                        }
                    };

                    _callbackHandle = GCHandle.Alloc(_receiveCallback);
                    var result = PsyneNative.psyne_channel_set_receive_callback(_handle, _receiveCallback, IntPtr.Zero);
                    PsyneException.ThrowIfError(result);
                }
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Channel));
        }

        /// <summary>
        /// Releases the resources associated with this channel.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                lock (_lock)
                {
                    if (!_disposed)
                    {
                        if (_callbackHandle.IsAllocated)
                        {
                            _callbackHandle.Free();
                        }

                        PsyneNative.psyne_channel_destroy(_handle);
                        _disposed = true;
                    }
                }
            }
        }
    }
}